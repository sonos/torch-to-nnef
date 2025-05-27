import typing as T
import logging
import warnings

import torch
from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.tensor.base import OpaqueTensor
from torch_to_nnef.utils import (
    get_parent_module_and_param_name,
    select_ctx_disable_torch_fn,
)

LOGGER = logging.getLogger(__name__)


class NamedTensor(OpaqueTensor):
    @staticmethod
    def __new__(
        cls,
        fp_tensor,
        *args,
        nnef_name,
        **kwargs,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=TracerWarning)
            return super().__new__(cls, fp_tensor, *args, **kwargs)

    def __init__(
        self,
        fp_tensor: torch.Tensor,
        nnef_name: str,
    ):
        super().__init__()
        self._fp_tensor = fp_tensor
        self.nnef_name = nnef_name

    def to_base_tensor(self):
        return self._fp_tensor

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, nnef_name='{self.nnef_name}')"

    def clone(self, *args, **kwargs):
        return self.__class__(
            super().clone(*args, **kwargs),
            nnef_name=self.nnef_name,
        )

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

        with select_ctx_disable_torch_fn():
            new_args = [a.clone() if isinstance(a, cls) else a for a in args]
            new_kwargs = {
                k: v.clone() if isinstance(v, cls) else v
                for k, v in kwargs.items()
            }
            ret = func(*new_args, **new_kwargs)
            if func in get_default_nowrap_functions():
                return ret
            # important modification
            # do not propagate this NamedTensor
            return _convert(ret, torch.Tensor)


def get_or_add_named_tensor(
    ids_to_ntensor, ref_mod, attr_name, full_name, data
):
    weight_id = id(getattr(ref_mod, attr_name))
    if weight_id in ids_to_ntensor:
        named_tensor = ids_to_ntensor[weight_id]
        LOGGER.info(
            f"detected shared weight between: {named_tensor.nnef_name} and '{full_name}'"
        )
    else:
        named_tensor = NamedTensor(data, nnef_name=full_name)
        ids_to_ntensor[weight_id] = named_tensor
    return named_tensor


def apply_name_to_tensor_in_module(model: torch.nn.Module):
    """Transform torch.Tensor or Parameters into NamedTensor

    This is applied at export time of `torch_to_nnef`
    Just before doing any tracing and allow to keep
    variable naming identical to PyTorch one

    This consistent naming unlock subsequent manipulations
    such as LORA applications @ inference or such.

    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.tensor.base import OpaqueTensor, OpaqueTensorRef

    skip_tensor_types = (OpaqueTensorRef, OpaqueTensor)

    LOGGER.debug("started to apply NamedTensor")
    ids_to_ntensor: T.Dict[int, NamedTensor] = {}
    for full_name, param in model.named_parameters(remove_duplicate=False):
        if isinstance(param.data, skip_tensor_types):
            continue
        ref_mod, pname = get_parent_module_and_param_name(model, full_name)

        LOGGER.debug(f"apply NamedTensor: {full_name}")
        named_tensor = get_or_add_named_tensor(
            ids_to_ntensor, ref_mod, pname, full_name, param.data
        )
        setattr(
            ref_mod,
            pname,
            torch.nn.Parameter(
                named_tensor,
                requires_grad=False,
            ),
        )
    # named buffers is not sufficient
    # as some tensor are not registered but on-fly generated
    for (
        named_m,
        ref_mod,
    ) in model.named_modules():
        for attr_name, attr_val in ref_mod.__dict__.items():
            if not isinstance(attr_val, torch.Tensor):
                continue
            if isinstance(attr_val.data, skip_tensor_types):
                continue
            # we need to capture every thing that is a tensor
            full_name = attr_name
            if named_m:
                full_name = f"{named_m}.{attr_name}"
            LOGGER.debug(f"apply NamedTensor: {full_name}")
            named_tensor = get_or_add_named_tensor(
                ids_to_ntensor, ref_mod, attr_name, full_name, attr_val
            )
            setattr(ref_mod, attr_name, named_tensor)

    for full_name, buffer in model.named_buffers(remove_duplicate=False):
        if isinstance(buffer, skip_tensor_types):
            continue
        ref_mod, bname = get_parent_module_and_param_name(model, full_name)
        LOGGER.debug(f"apply NamedTensor: {full_name}")
        named_tensor = get_or_add_named_tensor(
            ids_to_ntensor, ref_mod, bname, full_name, buffer
        )
        setattr(
            ref_mod,
            bname,
            named_tensor,
        )
    LOGGER.debug("sucessfull to apply NamedTensor everywhere")
