import nnef
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten.complex import (
    is_complex_dtype_and_complex_only_supported_as_lastdim,
)
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
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
        dim_data = []
        for dim in range(rank_data):
            # Reuse centralized dynamic axis extraction helper
            get_tract_dyn_axis_size_soc(op_helper, input_node, dim)
            dim_data.append(
                nnef.Identifier(f"{input_node.export_name}_dim{dim}")
            )
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

    PyTorch ``flatten(start_dim, end_dim)`` flattens dims in
    ``[start_dim, end_dim]`` *inclusive*; NNEF reshape uses ``axis_count``
    (number of axes to replace), so convert as
    ``axis_count = end_dim - start_dim + 1`` after normalizing negative
    indices via :func:`pick_axis`.

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
    raw_start = start_dim.data if start_dim.data is not None else 0
    raw_end = end_dim.data if end_dim.data is not None else -1
    axis_start = pick_axis(input_node, raw_start)
    axis_end = pick_axis(input_node, raw_end)
    axis_count = axis_end - axis_start + 1
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
            "axis_count": axis_count,
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


@OP_REGISTRY.register()
def flip(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:flip' to NNEF.

    Decomposes to a chain of `tract_core_gather` calls, one per axis,
    with a constant reversed-index tensor `[N-1, ..., 0]`. Static-shape
    only: a dynamic axis size raises `T2NErrorNotImplemented` (would
    require building the index at runtime via `tract_core_range` over
    `tract_core_shape_of(input)[axis]`).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "flip requires `tract_core_gather` (TractNNEF target)"
        )
    input_node, dims_node = node.inputs
    raw_dims = list(dims_node.data) if dims_node.data is not None else []

    seen = set()
    axes = []
    for d in raw_dims:
        a = pick_axis(input_node, d)
        if a not in seen:
            seen.add(a)
            axes.append(a)

    cur_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)

    if not axes:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=cur_ref,
            attrs={"shape": list(input_node.shape)},
        )
        return []

    for axis in axes:
        if not isinstance(input_node.shape[axis], int):
            raise T2NErrorNotImplemented(
                f"flip on dynamic axis {axis} not yet supported"
            )

    for i, axis in enumerate(axes):
        n = input_node.shape[axis]
        idx_const = PythonConstant(
            name=f"{node.outputs[0].export_name}_flip_idx_{axis}",
            data=torch.arange(n - 1, -1, -1, dtype=torch.int64),
        )
        idx_ref = get_or_add_tensor_variable_in_nnef(
            g, idx_const, name_to_tensor
        )
        is_last = i == len(axes) - 1
        cur_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_gather",
            inputs=[cur_ref, idx_ref],
            attrs={"axis": axis},
            force_consistent_inputs_shapes=False,
            output_tensor_name_suffix=("" if is_last else f"_flip_axis{axis}"),
        )
    return ["tract_core"]
