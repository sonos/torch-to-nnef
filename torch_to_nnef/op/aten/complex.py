import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import InferenceTarget, TractNNEF
from torch_to_nnef.op.helper import AtenOpRegistry

OP_REGISTRY = AtenOpRegistry()


def tract_complex_support(inference_target: InferenceTarget) -> bool:
    return (
        isinstance(inference_target, TractNNEF)
        and "complex" in inference_target.feature_flags
        and inference_target.version < "0.20.0"
    )


def is_complex_dtype_and_complex_only_supported_as_lastdim(
    dtype, inference_target: InferenceTarget
) -> bool:
    return dtype in [
        torch.complex64,
        torch.complex128,
    ] and not tract_complex_support(inference_target)


@OP_REGISTRY.register()
def view_as_complex(
    node,
    inference_target,
    torch_graph,
    **kwargs,
):
    """Map PyTorch: 'aten:view_as_complex' to NNEF."""
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
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
    node,
    torch_graph,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:view_as_real' to NNEF."""
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported by vanilla NNEF")
    # in such case we simulate complex with additional last axis being x2
    # 1 for real
    # 1 for imaginary
    # this means that rest of the flow still need to handle this design
    # decision.
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
    return []
