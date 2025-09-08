from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
    pick_index_in_axis,
)
from torch_to_nnef.torch_graph import FixedTensorList

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def cat(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:cat' to NNEF."""
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise T2NErrorNotImplemented(f"cat with input_item: {input_item}")
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)

    # edge case with zero sized tensors
    axis = max(pick_axis(inode, dim) for inode in input_node.data)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "concat",
        inputs=inputs,
        attrs={"axis": axis},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def stack(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:stack' to NNEF."""
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise T2NErrorNotImplemented(f"stack with input_item: {input_item}")
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "stack",
        inputs=inputs,
        attrs={"axis": pick_axis(input_node, dim)},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def vstack(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:vstack' to NNEF."""
    input_node = node.inputs[0]
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise T2NErrorNotImplemented(
                f"vstack with input_item: {input_item}"
            )
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "concat",
        inputs=inputs,
        attrs={"axis": 0},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def hstack(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:hstack' to NNEF."""
    input_node = node.inputs[0]
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise T2NErrorNotImplemented(
                f"vstack with input_item: {input_item}"
            )
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "concat",
        inputs=inputs,
        attrs={"axis": 1},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def roll(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:roll' to NNEF."""
    input_node, shifts_node, dims_node = node.inputs
    shifts = shifts_node.data
    dims = dims_node.data
    assert len(shifts) == len(dims), "shifts and dims need to be sample size"
    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if inference_target.has_dynamic_axes and not isinstance(
        inference_target, TractNNEF
    ):
        raise T2NErrorNotImplemented(inference_target)
    custom_fragments = []
    for i, _ in enumerate(shifts):
        tensor_chunks = []
        dim = dims[i]
        shift = shifts[i]
        begin = pick_index_in_axis(input_node, dim, -shift)
        if inference_target.has_dynamic_axes:
            shape_out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "dyn_slice_begin",
                inputs=input_tensor,
                attrs={
                    "axis": pick_axis(input_node, dim),
                    "begin": begin,
                    "stride": 1,
                },
                output_tensor_name_suffix=f"roll_l{i}_p1",
            )
            custom_fragments.append("dyn_slice_begin")
        else:
            maxsize = input_node.shape[dim]
            end = pick_index_in_axis(input_node, dim, maxsize)
            shape_out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=input_tensor,
                attrs={
                    "axes": [pick_axis(input_node, dim)],
                    "begin": [begin],
                    "end": [end],
                    "stride": [1],
                },
                output_tensor_name_suffix=f"roll_l{i}_p1",
            )
        tensor_chunks.append(shape_out)
        if inference_target.has_dynamic_axes:
            shape_out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "dyn_slice",
                inputs=input_tensor,
                attrs={
                    "axis": pick_axis(input_node, dim),
                    "begin": 0,
                    "end": -shift,
                    "stride": 1,
                },
                output_tensor_name_suffix=f"roll_l{i}_p2",
            )
            custom_fragments.append("dyn_slice")
        else:
            shape_out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=input_tensor,
                attrs={
                    "axes": [pick_axis(input_node, dim)],
                    "begin": [0],
                    "end": [pick_index_in_axis(input_node, dim, -shift)],
                    "stride": [1],
                },
                output_tensor_name_suffix=f"roll_l{i}_p2",
            )
        tensor_chunks.append(shape_out)
        # result = g.op("Concat", *shapes, axis_i=dims[i])
        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "concat",
            inputs=tensor_chunks,
            attrs={"axis": pick_axis(input_node, dim)},
            ensure_tuple=False,
            output_tensor_name_suffix=""
            if i + 1 == len(shifts)
            else f"roll_{i}",
        )
    return custom_fragments
