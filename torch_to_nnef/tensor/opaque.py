import abc
import logging
import typing as T
import warnings

import torch
from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.dtypes import dtype_is_whole_number
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


def trace_tensor_device_for_func(func) -> T.Optional[str]:
    """Which device an opaque parameter can be traced on for this op.

    ``"meta"`` keeps shape/dtype without materializing the backing data;
    ``"cpu"`` forces real decompressed values; ``None`` falls back to
    ``to_base_tensor()`` (also real values).

    Note: a ``"meta"`` op only carries shape/dtype, so its result is a meta
    tensor with no values and no real device. Two shapes of forward are thus
    unsupported for meta ops (they raise during trace instead of exporting):

    - reading concrete parameter *values* (e.g. branching on
      ``weight.view(...).argmax()``);
    - combining the meta result with a real tensor through a non-meta op
      (e.g. ``weight.view(...) * cpu_buffer``), which raises a device
      mismatch.

    Only shape-propagating uses that flow into a symbolic op
    (linear/matmul/select chains) are supported.

    The "meta" names below must correspond to genuine view/shape ops whose
    aten kind lives in ``ir_op.DERIVED_MODULE_ATTR_OPS`` (that is where the
    meta result is recognized as aliasing its constant input); keep the two
    lists in sync when adding a new view op.
    """
    if not NEW_OPAQUE_TRACING_STRATEGY:
        return None
    func_name = getattr(func, "__name__", "")
    if func_name in {"embedding", "index_select"}:
        return "cpu"
    if func_name in {
        "__getitem__",
        "bmm",
        "contiguous",
        "expand",
        "flatten",
        "linear",
        "matmul",
        "mm",
        "permute",
        "reshape",
        "select",
        "squeeze",
        "t",
        "transpose",
        "unsqueeze",
        "view",
    }:
        return "meta"
    return None


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

    def _to_trace_tensor(self, device: str):
        # A meta placeholder carries shape/dtype but no values and no real
        # device. It is only safe for float weights that flow straight into a
        # symbolic op (linear/matmul/select chains). Integer opaque params
        # (codebooks, index tables) carry values that tracing must read (e.g.
        # as gather indices or reshape sizes), so materialize them even when a
        # meta placeholder was requested. Constant-index ops (embedding,
        # index_select) also request materialization to avoid baking
        # uninitialized memory as NNEF constants.
        if device == "meta" and not dtype_is_whole_number(self.dtype):
            return torch.empty(
                tuple(self.shape), dtype=self.dtype, device=device
            )
        # Materialize on the parameter's native device rather than forcing one,
        # so a GPU-resident trace keeps the weight co-located with its runtime
        # index tensor instead of raising a device mismatch.
        return self._to_base_tensor()

    def to_trace_tensor(self, device: str):
        """Trace an opaque tensor without materializing its backing data."""

        @maybe_custom_op
        def opaque_t2n_expand(py_id: int) -> torch.Tensor:
            return self._to_trace_tensor(device)

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
                trace_device = trace_tensor_device_for_func(func)
                args = [
                    (
                        a.opaque_tensor.to_trace_tensor(trace_device)
                        if trace_device is not None
                        else a.opaque_tensor.to_base_tensor()
                    )
                    if isinstance(a, cls)
                    else a
                    for a in args
                ]
                kwargs = {
                    k: (
                        v.opaque_tensor.to_trace_tensor(trace_device)
                        if trace_device is not None
                        else v.opaque_tensor.to_base_tensor()
                    )
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
        if NEW_OPAQUE_TRACING_STRATEGY:
            meta_tensor = torch.empty(
                tuple(param.shape), dtype=param.dtype, device="meta"
            )
        else:
            meta_tensor = opaque_to_final_tensor(param).to("cpu")
        mod_tensor_updater.update_by_ref(
            param,
            OpaqueTensorRef(
                meta_tensor,
                param,
            ),
        )
    LOGGER.debug(
        "sucessfull to apply opaque tensor as reference (IR tracing friendly)"
    )
