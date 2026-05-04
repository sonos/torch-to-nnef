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

    NNEF spec has a `split` op (value, axis, ratios -> tensor[]) but tract
    does not register it, so we re-express each output as a `slice`.

    ``ratio_node`` may be a ``PythonConstant`` (literal sizes from the trace)
    or a ``TensorVariable`` whose data is shape-derived (e.g. fused-qkv
    splits like ``x.shape[-1] // 3``); both cases are unwrapped to plain ints.
    """
    (input_node, ratio_node, axis_node) = node.inputs
    assert isinstance(axis_node, PythonConstant)
    # ratio_node may be a PythonConstant or a TensorVariable whose data was
    # shape-derived (e.g. ``x.shape[-1] // 3``); what matters is that we can
    # pull concrete integer sizes out of it at trace time.
    if ratio_node.data is None:
        raise T2NErrorNotImplemented(
            "split_with_sizes requires statically-known sizes"
        )
    ratio_data = ratio_node.data
    if hasattr(ratio_data, "tolist"):
        ratio_data = ratio_data.tolist()

    def _as_int(x):
        if hasattr(x, "data"):
            x = x.data
        if hasattr(x, "item"):
            x = x.item()
        return int(x)

    ratio_data = [_as_int(x) for x in ratio_data]
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node, n_elements in zip(node.outputs, ratio_data, strict=False):
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
