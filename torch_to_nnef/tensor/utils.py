import typing as T

from torch import nn

from torch_to_nnef.utils import torch_version


def get_named_parameters(
    mod: nn.Module, remove_duplicate: bool = True
) -> T.Iterator[T.Tuple[str, nn.Parameter]]:
    if torch_version() < "2.0.0":
        return mod.named_parameters()
    return mod.named_parameters(remove_duplicate=remove_duplicate)


def get_named_buffers(
    mod: nn.Module, remove_duplicate: bool = True
) -> T.Iterator[
    T.Tuple[str, "nn.Buffer"]
]:  # legacy version of torch doesn't have Buffer
    if torch_version() < "2.0.0":
        return mod.named_buffers()
    return mod.named_buffers(remove_duplicate=remove_duplicate)
