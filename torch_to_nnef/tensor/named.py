import logging
import warnings

import torch
from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.utils import (
    get_parent_module_and_param_name,
    select_ctx_disable_torch_fn,
)

LOGGER = logging.getLogger(__name__)


class NamedTensor(torch.Tensor):
    """Tensor enriched with name attribute."""

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
            try:
                return super().__new__(cls, fp_tensor, *args, **kwargs)
            except TypeError:  # legacy mode
                legacy_tensor = fp_tensor.as_subclass(cls)
                legacy_tensor.__dict__["_fp_tensor"] = fp_tensor
                legacy_tensor.__dict__["nnef_name"] = nnef_name
                assert legacy_tensor.dtype == fp_tensor.dtype
                assert legacy_tensor.shape == fp_tensor.shape
                return legacy_tensor

    def __init__(
        self,
        fp_tensor: torch.Tensor,
        nnef_name: str,
    ):
        super().__init__()
        self._fp_tensor = fp_tensor
        self.nnef_name = nnef_name

    def __hash__(self):
        return self._fp_tensor.__hash__()

    @property
    def data(self):
        """Very important to keep access to all special attr of NamedTensor."""
        return self

    @data.setter
    def data(self, new_data):
        raise T2NErrorNotImplemented(
            f"Trying to alter a TensorRef.data: {self}"
        )

    def detach(self):
        # need overwrite since nn.Paramater use it at __new__
        LOGGER.debug("NamedTensor does not support detach")
        return self

    def requires_grad_(self, mode=False):
        # need overwrite since nn.Paramater use it at __new__
        LOGGER.debug("NamedTensor does not support requires_grad")
        return self

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, nnef_name='{self.nnef_name}')"

    def clone(self, *args, **kwargs):
        return self.__class__(
            torch.Tensor.clone(self, *args, **kwargs),
            nnef_name=self.nnef_name,
        )

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
            "detected shared weight between: '%s' and '%s'",
            named_tensor.nnef_name,
            full_name,
        )
    else:
        named_tensor = NamedTensor(data, nnef_name=full_name)
        ids_to_ntensor[weight_id] = named_tensor
    return named_tensor


def apply_name_to_tensor_in_module(model: torch.nn.Module):
    """Transform torch.Tensor or Parameters into NamedTensor.

    This is applied at export time of `torch_to_nnef`
    Just before doing any tracing and allow to keep
    variable naming identical to PyTorch one

    This consistent naming unlock subsequent manipulations
    such as LORA applications @ inference or such.

    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.tensor.opaque import OpaqueTensor, OpaqueTensorRef

    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.tensor.updater import ModTensorUpdater

    skip_tensor_types = (OpaqueTensorRef, OpaqueTensor)

    mod_tensor_updater = ModTensorUpdater(
        model,
        add_buffers=True,
        add_unregistred_tensor=True,
    )
    LOGGER.debug("started to apply NamedTensor")
    for names in list(mod_tensor_updater.id_to_names.values()):
        name = list(names)[0]
        ref_mod, local_name = get_parent_module_and_param_name(model, name)
        ref = getattr(ref_mod, local_name)
        if isinstance(ref, skip_tensor_types):
            continue
        LOGGER.debug("apply NamedTensor: %s", name)
        named_tensor = NamedTensor(ref, nnef_name=name)
        mod_tensor_updater.update_by_ref(ref, named_tensor)
    LOGGER.debug("sucessfull to apply NamedTensor everywhere")
