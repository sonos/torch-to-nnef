from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_multi_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
)
from torch_to_nnef.torch_graph import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def split_with_sizes(g, node, name_to_tensor, **kwargs):
    """Translate `aten::split_with_sizes` to NNEF.

    We are aware that.
    split<?>(
        value: tensor<?>,
        axis: integer,
        ratios: integer[]
    ) -> ( values: tensor<?>[] )

    exists but since tract does not support it, we reexpress it with slice
    """
    (input_node, ratio_node, axis_node) = node.inputs
    assert isinstance(axis_node, PythonConstant)
    assert isinstance(ratio_node, PythonConstant)
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node, n_elements in zip(node.outputs, ratio_node.data):
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if n_elements <= 0:
            raise T2NErrorNotImplemented("unexpected n_elements<=0")
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [pick_axis(input_node, axis_node.data)],
                "begin": [current_dim_elm_idx],
                "end": [current_dim_elm_idx + n_elements],
                "stride": [1],
            },
        )
        if inputs.quant:
            out.quant = inputs.quant
        current_dim_elm_idx += n_elements


@OP_REGISTRY.register()
def unbind(g, node, name_to_tensor, **kwargs):
    """Unbind is `unstack` in NNEF."""
    input_node, axis_node = node.inputs
    add_multi_output_op(
        g,
        node,
        name_to_tensor,
        "unstack",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axis": pick_axis(input_node, axis_node.data)},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def chunk(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:chunk' to NNEF."""
    (input_node, n_chunk_node, axis_node) = node.inputs
    assert n_chunk_node.data == len(node.outputs)
    assert len({tuple(o.shape) for o in node.outputs}) == 1, (
        "all chunk are not equal"
    )
    n_elements = node.outputs[0].shape[axis_node.data]
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node in node.outputs:
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [pick_axis(input_node, axis_node.data)],
                "begin": [current_dim_elm_idx],
                "end": [current_dim_elm_idx + n_elements],
                "stride": [1],
            },
        )
        current_dim_elm_idx += n_elements
