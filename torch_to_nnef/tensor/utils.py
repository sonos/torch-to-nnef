import logging
import typing as T

import torch
from torch import nn

from torch_to_nnef.utils import torch_version


def _legacy_named_parameters(
    self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
) -> T.Iterator[T.Tuple[str, nn.Parameter]]:
    """Extend legacy named_parameters to add remove_duplicate."""
    yield from self._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix,
        recurse=recurse,
        remove_duplicate=remove_duplicate,
    )


def _legacy_named_buffers(
    self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
) -> T.Iterator[T.Tuple[str, torch.Tensor]]:
    """Extend legacy named_buffers to add remove_duplicate."""
    yield from self._named_members(
        lambda module: module._buffers.items(),
        prefix=prefix,
        recurse=recurse,
        remove_duplicate=remove_duplicate,
    )


def _legacy_robust_torch_named_members(
    self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
):
    """Helper method for yielding various names + members of modules.

    Add: remove_duplicate to legacy
    Fix:
        RuntimeError: Boolean value of Tensor with more than one value
            is ambiguous
    """
    remove_duplicate = True
    memo = set()
    modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            name = module_prefix + ("." if module_prefix else "") + k
            try:
                if v is None or (remove_duplicate and v in memo):
                    continue
            except RuntimeError as exp:
                logging.debug(
                    "needed to get bellow .data in '%s': %s", name, exp
                )
                v = v.data  # try to use down data
                if v is None or (remove_duplicate and v in memo):
                    continue
            if remove_duplicate:
                memo.add(v)
            yield name, v


def get_named_parameters(
    mod: nn.Module, remove_duplicate: bool = True
) -> T.Iterator[T.Tuple[str, nn.Parameter]]:
    if torch_version() < "2.0.0":
        nn.Module._named_members = _legacy_robust_torch_named_members
        nn.Module.named_parameters = _legacy_named_parameters
    return mod.named_parameters(remove_duplicate=remove_duplicate)


def get_named_buffers(
    mod: nn.Module, remove_duplicate: bool = True
) -> T.Iterator[
    T.Tuple[str, "nn.Buffer"]
]:  # too old version of torch doesn't have Buffer
    if torch_version() < "2.0.0":
        nn.Module._named_members = _legacy_robust_torch_named_members
        nn.Module.named_buffers = _legacy_named_buffers
    return mod.named_buffers(remove_duplicate=remove_duplicate)
