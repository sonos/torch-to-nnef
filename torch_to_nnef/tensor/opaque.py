import abc
import logging
import warnings

from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions
import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.utils import (
    select_ctx_disable_torch_fn,
    get_parent_module_and_param_name,
)

LOGGER = logging.getLogger(__name__)

IR_OPAQUE_NAME = "t2n::opaque_tensor_expand"


def find_opaque_ref_by_py_id(module: torch.nn.Module, py_id: int):
    """Allow to fetch back the opaque parameter once passed the jit 'wall'"""
    for _ in module.parameters():
        if isinstance(_, OpaqueTensorRef):
            opaque_uuid = id(_.opaque_tensor)
            if opaque_uuid == py_id:
                return _
    raise ValueError(f"OpaqueTensor with id({py_id}) not found")


class OpaqueTensor(torch.Tensor):
    @property
    def data(self):
        """very important to keep access to all special attr of OpaqueTensor"""
        return self

    @data.setter
    def data(self, new_data):
        raise TorchToNNEFNotImplementedError(
            f"Trying to alter a TensorRef.data: {self}"
        )

    def clone(self, *args, **kwargs):
        return self

    def detach(self):
        # need overwrite since nn.Paramater use it at __new__
        LOGGER.debug("OpaqueTensor does not support detach")
        return self

    def requires_grad_(self, mode=False):
        # need overwrite since nn.Paramater use it at __new__
        LOGGER.debug("OpaqueTensor does not support requires_grad")
        return self

    @abc.abstractmethod
    def _to_base_tensor(self):
        raise NotImplementedError()

    def to_base_tensor(self):
        """wrap _to_base_tensor with jit export infos"""

        @torch.library.custom_op(IR_OPAQUE_NAME, mutates_args=())
        def opaque_t2n_expand(py_id: int) -> torch.Tensor:
            tensor = self._to_base_tensor()
            return tensor

        return opaque_t2n_expand(id(self))


class OpaqueTensorRef(torch.Tensor):
    """Allow to pass through 'tracing'"""

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
    def nnef_name(self):
        return getattr(self.opaque_tensor, "nnef_name", None)

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        raise TorchToNNEFNotImplementedError(
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
        """
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
            if not skip_expansion:
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


def set_opaque_tensor_in_params_as_ref(model: torch.nn.Module):
    """Transform OpaqueTensor Parameters into OpaqueTensorRef

    This is applied at export time of `torch_to_nnef`
    Just before doing any tracing

    """
    LOGGER.debug("started to apply opaque tensor ref with decompress")
    ids_to_qparams = {}
    for full_name, param in model.named_parameters(remove_duplicate=False):
        if not isinstance(param, OpaqueTensor):
            continue
        qid = id(param)
        if qid not in ids_to_qparams:
            param.nnef_name = full_name
            new_param = OpaqueTensorRef(
                param.to_base_tensor().to("meta"),
                param,
            )
            ids_to_qparams[qid] = new_param
        else:
            new_param = ids_to_qparams[qid]
        ref_mod, p_name = get_parent_module_and_param_name(model, full_name)

        LOGGER.debug(f"apply opaque tensor ref with decompress: {full_name}")
        setattr(
            ref_mod,
            p_name,
            torch.nn.Parameter(
                new_param,
                requires_grad=False,
            ),
        )
    LOGGER.debug("sucessfull to apply opaque tensor ref with decompress")
