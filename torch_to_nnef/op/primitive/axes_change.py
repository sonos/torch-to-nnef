from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
)
from torch_to_nnef.op.primitive.complex import (
    is_complex_dtype_and_complex_only_supported_as_lastdim,
)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def view(
    g,
    node,
    name_to_tensor,
    torch_graph,
    has_dynamic_axes,
    tract_feature_flags,
    **kwargs,
):
    (input_node, axis_node) = node.inputs
    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=has_dynamic_axes,
    )
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, tract_feature_flags
    ):
        dim_data.append(2)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


@OP_REGISTRY.register()
def transpose(g, node, name_to_tensor, tract_feature_flags, **kwargs):
    (input_node, dim0_node, dim1_node) = node.inputs
    dim0 = pick_rank(input_node, dim0_node.data)
    dim1 = pick_rank(input_node, dim1_node.data)

    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, tract_feature_flags
    ):
        raise TorchToNNEFNotImplementedError(
            "complex transpose without tract complex feature flag"
        )

    new_dims_ranks = []
    for _ in range(node.outputs[0].rank):
        if _ == dim0:
            new_dims_ranks.append(dim1)
        elif _ == dim1:
            new_dims_ranks.append(dim0)
        else:
            new_dims_ranks.append(_)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": new_dims_ranks},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def permute(g, node, name_to_tensor, **kwargs):
    (input_node, dims_node) = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, _) for _ in dims_node.data]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def unsqueeze(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs

    dim = axis_node.data
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, dim)]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def squeeze(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, dim)]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def flatten(g, node, name_to_tensor, tract_feature_flags, **kwargs):
    """
    Using NNEF:
        fragment reshape<?>(
            input: tensor<?>,
            shape: integer[],
            axis_start: integer = 0,
            axis_count: integer = -1
        ) -> ( output: tensor<?> );
    """
    (input_node, _, _) = node.inputs  # start_dim_name  # end_dim_name
    onode = node.outputs[0]
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, tract_feature_flags
    ):
        raise TorchToNNEFNotImplementedError(
            "complex flatten without tract complex feature flag"
        )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "dtype": onode.np_dtype,
            "shape": list(onode.shape),
            "axis_start": 0,
            "axis_count": -1,
        },
    )


@OP_REGISTRY.register()
def reshape(
    g,
    node,
    name_to_tensor,
    torch_graph,
    has_dynamic_axes,
    tract_feature_flags,
    **kwargs,
):
    (input_node, axis_node) = node.inputs

    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=has_dynamic_axes,
        force_none_as_tensor_ref=True,
    )
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, tract_feature_flags
    ):
        dim_data.append(2)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )
