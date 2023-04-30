import numpy as np
import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    OpRegistry,
    add_single_output_op,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
)

OP_REGISTRY = OpRegistry()


@OP_REGISTRY.register()
def pad(node, **kwargs):
    kind = node.inputs.pop(2)
    if kind.data == "constant":
        return constant_pad_nd(node=node, **kwargs)
    if kind.data in ["reflection", "reflect"]:  # pre 1.12.0  # post 1.12.0
        node.inputs = node.inputs[:2]
        return reflection_padnd(node=node, **kwargs)
    if kind.data == "replicate":
        node.inputs = node.inputs[:2]
        return replication_padnd(node=node, **kwargs)
    raise TorchToNNEFNotImplementedError(
        f"pad kind={kind.data} not implemented"
    )


@OP_REGISTRY.register()
def reflection_padnd(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    (input_node, pads_node) = node.inputs
    pads = get_list_of_int(
        pads_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    assert isinstance(pads, list)
    assert all(isinstance(_, int) for _ in pads)
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"padding": pads, "border": "reflect"},
    )


@OP_REGISTRY.register()
def replication_padnd(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    (input_node, pads_node) = node.inputs
    pads = get_list_of_int(
        pads_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    assert isinstance(pads, list)
    assert all(isinstance(_, int) for _ in pads)
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"padding": pads, "border": "replicate"},
    )


@OP_REGISTRY.register()
def constant_pad_nd(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    (input_node, pads_node, value_node) = node.inputs
    pads = get_list_of_int(
        pads_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    assert isinstance(pads, list)
    assert all(isinstance(_, int) for _ in pads)
    value = value_node.data
    if value is None:
        value = 0  # add default value if not set
    # ensure cast to same dtype as output
    value = torch.tensor(value, dtype=node.outputs[0].dtype).tolist()
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"padding": pads, "value": value},
    )
