import nnef

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.aten.complex import (
    is_complex_dtype_and_complex_only_supported_as_lastdim,
)
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    SimpleOpChainer,
    add_single_output_op,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def view(
    g,
    node,
    name_to_tensor,
    torch_graph,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:view' to NNEF."""
    (input_node, axis_node) = node.inputs
    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=inference_target.has_dynamic_axes,
    )
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
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
def unflatten(
    g,
    node,
    name_to_tensor,
    torch_graph,
    op_helper,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:unflatten' to NNEF."""
    (input_node, axis_node, new_shape_chunk_node) = node.inputs
    assert isinstance(axis_node, PythonConstant), (
        "axis is supposed to be static"
    )

    rank_data = pick_axis(input_node, axis_node.data)

    partial_dim_data = get_list_of_int(
        new_shape_chunk_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=inference_target.has_dynamic_axes,
    )

    if inference_target.has_dynamic_axes:
        # TODO: fix me now
        soc = SimpleOpChainer(
            op_helper=op_helper, input_data_nodes=[input_node]
        )
        shape_tensor_name = f"{input_node.export_name}_shape"
        soc = soc.chain(
            "tract_core_shape_of",
            force_full_output_tensor_name=shape_tensor_name,
        )
        dim_data = []
        for dim in range(rank_data):
            index_tensor_name = f"{input_node.export_name}_dim{dim}"
            if index_tensor_name not in op_helper.name_to_tensor:
                soc.chain(
                    "slice",
                    attrs={
                        "axes": [0],
                        "begin": [dim],
                        "end": [dim + 1],
                        "stride": [1],
                    },
                    output_tensor_name_suffix=f"sliced{dim}",
                ).chain(
                    "squeeze",
                    attrs={
                        "axes": [0],
                    },
                    force_full_output_tensor_name=index_tensor_name,
                )
            dim_size = nnef.Identifier(index_tensor_name)
            dim_data.append(dim_size)
    else:
        dim_data = input_node.shape[:rank_data]
    dim_data = dim_data + partial_dim_data + input_node.shape[rank_data + 1 :]
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
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
def transpose(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:transpose' to NNEF."""
    (input_node, dim0_node, dim1_node) = node.inputs
    dim0 = pick_axis(input_node, dim0_node.data)
    dim1 = pick_axis(input_node, dim1_node.data)

    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        raise T2NErrorNotImplemented(
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
    """Map PyTorch: 'aten:permute' to NNEF."""
    (input_node, dims_node) = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_axis(input_node, _) for _ in dims_node.data]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def unsqueeze(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:unsqueeze' to NNEF."""
    (input_node, axis_node) = node.inputs

    axis = pick_axis(input_node, axis_node.data)
    if axis_node.data < 0:
        axis += 1
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [axis]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def squeeze(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:squeeze' to NNEF."""
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
        attrs={"axes": [pick_axis(input_node, dim)]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def flatten(g, node, name_to_tensor, inference_target, **kwargs):
    """Translate operator: `aten::flatten` to NNEF.

    Using NNEF:.
    fragment reshape<?>(
        input: tensor<?>,
        shape: integer[],
        axis_start: integer = 0,
        axis_count: integer = -1
    ) -> ( output: tensor<?> );
    """
    (input_node, start_dim, end_dim) = node.inputs
    onode = node.outputs[0]
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        raise T2NErrorNotImplemented(
            "complex flatten without tract complex feature flag"
        )
    axis_start = start_dim.data or 0
    axis_end = end_dim.data or -1
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
            "shape": [-1],
            "axis_start": axis_start,
            "axis_count": axis_end,
        },
    )


@OP_REGISTRY.register()
def reshape(
    g,
    node,
    name_to_tensor,
    torch_graph,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:reshape' to NNEF."""
    (input_node, axis_node) = node.inputs

    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        force_none_as_tensor_ref=True,
    )
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
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
