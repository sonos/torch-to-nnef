from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.aten.base import (
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
            raise TorchToNNEFNotImplementedError(
                f"cat with input_item: {input_item}"
            )
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    axis = pick_axis(input_node.data[0], dim)
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
            raise TorchToNNEFNotImplementedError(
                f"stack with input_item: {input_item}"
            )
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
    input_node = node.inputs[0]
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise TorchToNNEFNotImplementedError(
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
    input_node = node.inputs[0]
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise TorchToNNEFNotImplementedError(
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
    input_node, shifts_node, dims_node = node.inputs
    shifts = shifts_node.data
    dims = dims_node.data
    assert len(shifts) == len(dims), "shifts and dims need to be sample size"
    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    for i, _ in enumerate(shifts):
        tensor_chunks = []
        dim = dims[i]
        shift = shifts[i]
        if not inference_target.has_dynamic_axes:
            maxsize = input_node.shape[dim]
        else:
            raise TorchToNNEFNotImplementedError("Should use shape_of")
        shape_out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=input_tensor,
            attrs={
                "axes": [pick_axis(input_node, dim)],
                "begin": [pick_index_in_axis(input_node, dim, -shift)],
                "end": [pick_index_in_axis(input_node, dim, maxsize)],
                "stride": [1],
            },
            output_tensor_name_suffix=f"roll_l{i}_p1",
        )
        tensor_chunks.append(shape_out)
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
    return []
