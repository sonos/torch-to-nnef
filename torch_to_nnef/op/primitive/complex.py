import typing as T

import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import AtenOpRegistry
from torch_to_nnef.tract import tract_version

OP_REGISTRY = AtenOpRegistry()


def is_complex_dtype_and_complex_only_supported_as_lastdim(
    dtype, tract_feature_flags: T.List[str]
) -> bool:
    return (
        dtype in [torch.complex64, torch.complex128]
        and (
            tract_feature_flags is None or "complex" not in tract_feature_flags
        )
        and "0.20.0" <= tract_version()
    )


@OP_REGISTRY.register()
def view_as_complex(
    g,
    node,
    name_to_tensor,
    nnef_spec_strict,
    tract_feature_flags,
    torch_graph,
    **kwargs,
):
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError(
            "Complex not supported in vanilla spec"
        )
    # in such case we simulate complex with additional last axis being x2
    # 1 for real
    # 1 for imaginary
    # this means that rest of the flow still need to handle this design
    # decision.
    node.inputs[0].dtype = torch.complex64
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
    return []


@OP_REGISTRY.register()
def view_as_real(
    g,
    node,
    name_to_tensor,
    torch_graph,
    nnef_spec_strict,
    tract_feature_flags,
    **kwargs,
):
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError(
            "Complex not supported by vanilla NNEF"
        )
    # in such case we simulate complex with additional last axis being x2
    # 1 for real
    # 1 for imaginary
    # this means that rest of the flow still need to handle this design
    # decision.
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
    return []
