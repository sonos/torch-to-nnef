import abc
import logging
import warnings

import torch
from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.exceptions import (
    T2NErrorNotImplemented,
    T2NErrorTorchJitTraceFailed,
)
from torch_to_nnef.tensor.utils import get_named_parameters
from torch_to_nnef.utils import select_ctx_disable_torch_fn, torch_version

LOGGER = logging.getLogger(__name__)

IR_OPAQUE_NAME = "t2n::opaque_tensor_expand"

# Since pytorch 2.4 added: `torch.library.custom_op`
# we can trace with jit.trace with meta reference
#
# On Prior version we use legacy technique with
# OpaqueTensorRef that contains 'real' data
# this legacy is less optimal since it duplicate
# weights at export time between with Opaque and
# OpaqueTensorRef.
NEW_OPAQUE_TRACING_STRATEGY = torch_version() >= "2.4.0"


def maybe_custom_op(f):
    if NEW_OPAQUE_TRACING_STRATEGY:
        wrap = torch.library.custom_op(IR_OPAQUE_NAME, mutates_args=())(f)
    else:
        wrap = f
    return wrap


def find_opaque_ref_by_py_id(module: torch.nn.Module, py_id: int):
    """Allow to fetch back the opaque parameter once passed the jit 'wall'."""
    for _ in module.parameters():
        if isinstance(_, OpaqueTensorRef):
            opaque_uuid = id(_.opaque_tensor)
            if opaque_uuid == py_id:
                return _
    raise T2NErrorTorchJitTraceFailed(
        f"OpaqueTensor with id({py_id}) not found"
    )


class OpaqueTensor(torch.Tensor):
    @property
    def data(self):
        """Very important to keep access to all special attr of OpaqueTensor."""
        return self

    @data.setter
    def data(self, new_data):
        raise T2NErrorNotImplemented(
            f"Trying to alter a TensorRef.data: {self}"
        )

    def clone(self, *args, **kwargs):
        return self

    def detach(self):
        # need overwrite since nn.Parameter use it in  .__new__
        LOGGER.debug("OpaqueTensor does not support detach")
        return self

    def requires_grad_(self, mode=False):
        # need overwrite since nn.Parameter use it in .__new__
        LOGGER.debug("OpaqueTensor does not support requires_grad")
        return self

    @abc.abstractmethod
    def _to_base_tensor(self):
        raise T2NErrorNotImplemented()

    def to_base_tensor(self):
        """Wrap _to_base_tensor with jit export infos."""

        @maybe_custom_op
        def opaque_t2n_expand(py_id: int) -> torch.Tensor:
            tensor = self._to_base_tensor()
            return tensor

        return opaque_t2n_expand(id(self))


class OpaqueTensorRef(torch.Tensor):
    """Allow to pass through 'tracing'."""

    @staticmethod
    def __new__(
        cls,
        meta_tensor,
        opaque_tensor,
        *args,
        **kwargs,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=TracerWarning)
            return super().__new__(cls, meta_tensor, *args, **kwargs)

    def __init__(
        self,
        meta_tensor: torch.Tensor,
        opaque_tensor: OpaqueTensor,
    ):
        super().__init__()
        self.meta_tensor = meta_tensor
        self.opaque_tensor = opaque_tensor

    @property
    def device(self):
        return self.opaque_tensor.device

    @property
    def nnef_name(self):
        return getattr(self.opaque_tensor, "nnef_name", None)

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        raise T2NErrorNotImplemented(
            f"Trying to alter a TensorRef.data: {self}"
        )

    def clone(self, *args, **kwargs):
        return self

    def to(self, *args, **kwargs):
        self.opaque_tensor = self.opaque_tensor.to(*args, **kwargs)
        return self

    def detach(self):
        # need overwrite since nn.Paramater use it at __new__
        return self

    def requires_grad_(self, requires_grad):
        # need overwrite since nn.Paramater use it at __new__
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Custom __torch_function__.

        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.
        we modify it so it's always reference torch.Tensor.
        """
        if kwargs is None:
            kwargs = {}

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.tensor import NamedTensor

        if not all(
            issubclass(cls, t) or issubclass(NamedTensor, t) for t in types
        ):
            return NotImplemented

        with select_ctx_disable_torch_fn():
            skip_expansion = func in get_default_nowrap_functions().union(
                {cls.__repr__}
            ) or any(
                _ in str(func)
                for _ in ["'__get__'", "'__set__'", "Tensor.__reduce_ex__"]
            )
            if not skip_expansion and NEW_OPAQUE_TRACING_STRATEGY:
                args = [
                    a.opaque_tensor.to_base_tensor()
                    if isinstance(a, cls)
                    else a
                    for a in args
                ]
                kwargs = {
                    k: v.opaque_tensor.to_base_tensor()
                    if isinstance(v, cls)
                    else v
                    for k, v in kwargs.items()
                }

            ret = func(*args, **kwargs)
            if skip_expansion:
                return ret
            # important modification
            # do not propagate this qtype
            return _convert(ret, torch.Tensor)


def opaque_to_final_tensor(rtensor: torch.Tensor) -> torch.Tensor:
    """Even if OpaqueTensor are composed it exposes fully expanded tensor.

    So for example: an OffloadedTensor that contains a QTensor
    will 'load' then 'decompress' to show final fp tensor.

    """
    while isinstance(rtensor, OpaqueTensor):
        rtensor = rtensor.to_base_tensor()
    return rtensor


def set_opaque_tensor_in_params_as_ref(model: torch.nn.Module):
    """Transform OpaqueTensor Parameters into OpaqueTensorRef.

    This is applied at export time of `torch_to_nnef`
    Just before doing any tracing

    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.tensor.updater import ModTensorUpdater

    LOGGER.debug(
        "started to apply opaque tensor as reference (IR tracing friendly)"
    )
    mod_tensor_updater = ModTensorUpdater(model)
    for full_name, param in get_named_parameters(model, remove_duplicate=False):
        if not isinstance(param, OpaqueTensor):
            continue
        param.nnef_name = full_name
        LOGGER.debug("apply opaque tensor reference: %s", full_name)
        mod_tensor_updater.update_by_ref(
            param,
            OpaqueTensorRef(
                opaque_to_final_tensor(param).to(
                    "meta" if NEW_OPAQUE_TRACING_STRATEGY else "cpu"
                ),
                param,
            ),
        )
    LOGGER.debug(
        "sucessfull to apply opaque tensor as reference (IR tracing friendly)"
    )
