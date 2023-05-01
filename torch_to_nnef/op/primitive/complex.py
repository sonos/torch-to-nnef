import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
)

OP_REGISTRY = AtenOpRegistry()


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
    if tract_feature_flags is not None and "complex" in tract_feature_flags:
        input_tensor = get_or_add_tensor_variable_in_nnef(
            g, node.inputs[0], name_to_tensor
        )
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_inner_dim_to_complex",
            inputs=input_tensor,
        )
        return ["tract_core"]
    else:
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
    if tract_feature_flags is not None and "complex" in tract_feature_flags:
        # input_node, n_node, dim_node, norm_node = node.inputs
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_complex_to_inner_dim",
            inputs=get_or_add_tensor_variable_in_nnef(
                g, node.inputs[0], name_to_tensor
            ),
        )
        return ["tract_core"]
    else:
        # in such case we simulate complex with additional last axis being x2
        # 1 for real
        # 1 for imaginary
        # this means that rest of the flow still need to handle this design
        # decision.
        torch_graph.remap_node(node.outputs[0], node.inputs[0])
        return []
