import numpy as np
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def pad(node, **kwargs):
    """Map PyTorch: 'aten:pad' to NNEF."""
    kind = node.inputs.pop(2)
    if kind.data == "constant":
        return constant_pad_nd(node=node, **kwargs)
    if kind.data in ["reflection", "reflect"]:  # pre 1.12.0  # post 1.12.0
        node.inputs = node.inputs[:2]
        return reflection_padnd(node=node, **kwargs)
    if kind.data == "replicate":
        node.inputs = node.inputs[:2]
        return replication_padnd(node=node, **kwargs)
    raise T2NErrorNotImplemented(f"pad kind={kind.data} not implemented")


def _pad_format(pads, node):
    pads_r = pads[:]
    pads = np.zeros(len(pads)).reshape(-1, 2).tolist()
    for idx, pad_val in enumerate(pads_r[::-1]):
        left_idx = idx // 2
        right_idx = (idx + 1) % 2
        pads[left_idx][right_idx] = pad_val

    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    return pads


@OP_REGISTRY.register(
    torch_op_ids=[
        "reflection_pad1d",
        "reflection_pad2d",
        "reflection_pad3d",
        "reflection_padnd",
    ]
)
def reflection_padnd(
    g, node, name_to_tensor, torch_graph, inference_target, **kwargs
):
    """Map PyTorch: 'aten:reflection_pad{1,2,3,n}d' to NNEF."""
    (input_node, pads_node) = node.inputs
    pads = _pad_format(
        get_list_of_int(
            pads_node,
            torch_graph,
            name_to_tensor=name_to_tensor,
            has_dynamic_axes=inference_target.has_dynamic_axes,
        ),
        node,
    )
    assert isinstance(pads, list)
    # assert all(isinstance(_, int) for _ in pads)
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


@OP_REGISTRY.register(
    torch_op_ids=[
        "replication_pad1d",
        "replication_pad2d",
        "replication_pad3d",
        "replication_padnd",
    ]
)
def replication_padnd(
    g, node, name_to_tensor, torch_graph, inference_target, **kwargs
):
    """Map PyTorch: 'aten:replication_pad{1,2,3,n}d' to NNEF."""
    (input_node, pads_node) = node.inputs
    pads = _pad_format(
        get_list_of_int(
            pads_node,
            torch_graph,
            name_to_tensor=name_to_tensor,
            has_dynamic_axes=inference_target.has_dynamic_axes,
        ),
        node,
    )

    assert isinstance(pads, list)
    # assert all(isinstance(_, int) for _ in pads)
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


@OP_REGISTRY.register(torch_op_ids=["constant_pad1d", "constant_pad_nd"])
def constant_pad_nd(
    g, node, name_to_tensor, torch_graph, inference_target, **kwargs
):
    """Map PyTorch: 'aten:constant_pad_{1,n}d' to NNEF."""
    (input_node, pads_node, value_node) = node.inputs
    pads = _pad_format(
        get_list_of_int(
            pads_node,
            torch_graph,
            name_to_tensor=name_to_tensor,
            has_dynamic_axes=inference_target.has_dynamic_axes,
        ),
        node,
    )
    assert isinstance(pads, list)
    # assert all(isinstance(_, int) for _ in pads)
    value = value_node.data
    if value is None:
        value = 0  # add default value if not set
    # ensure cast to same dtype as output
    value = torch.tensor(value, dtype=node.outputs[0].dtype).tolist()

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
