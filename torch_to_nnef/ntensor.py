import logging

import torch
from torch._tensor import _convert
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.utils import select_ctx_disable_torch_fn

LOGGER = logging.getLogger(__name__)


class NamedTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        fp_tensor,
        *args,
        nnef_name,
        **kwargs,
    ):
        return super().__new__(cls, fp_tensor, *args, **kwargs)

    def __init__(
        self,
        fp_tensor: torch.Tensor,
        nnef_name: str,
    ):
        super().__init__()
        self.nnef_name = nnef_name

    @property
    def data(self):
        return self

    def clone(self, *args, **kwargs):
        return self.__class__(
            super().clone(*args, **kwargs),
            nnef_name=self.nnef_name,
        )

    def to(self, *args, **kwargs):
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

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

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
            # do not propagate this qtype
            return _convert(ret, torch.Tensor)


def apply_name_to_tensor_in_params(model: torch.nn.Module):
    """Transform torch.Tensor Parameters into NamedTensor

    This is applied at export time of `torch_to_nnef`
    Just before doing any tracing and allow to keep
    variable naming identical to PyTorch one

    """
    LOGGER.debug("started to apply NamedTensor")
    for named_p, param in model.named_parameters():
        ref_mod = model
        chunked_names = named_p.split(".")
        for mod_name in chunked_names[:-1]:
            ref_mod = getattr(ref_mod, mod_name)

        LOGGER.debug(f"apply NamedTensor: {named_p}")
        named_tensor = NamedTensor(param.data, nnef_name=named_p)
        setattr(
            ref_mod,
            chunked_names[-1],
            torch.nn.Parameter(
                named_tensor,
                requires_grad=False,
            ),
        )
    LOGGER.debug("sucessfull to apply NamedTensor everywhere")
