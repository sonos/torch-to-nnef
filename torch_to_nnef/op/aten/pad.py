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
    g, node, name_to_tensor, torch_graph, inference_target, op_helper, **kwargs
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

    # PyTorch's constant_pad_nd allows negative entries (cropping the
    # corresponding side). NNEF `pad` only accepts non-negative
    # paddings, so split: emit a `slice` to absorb the negative parts
    # first, then a `pad` for the positive remainder. Common path
    # (all-non-negative) stays a single `pad`.
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    custom_fragments: list = []
    has_negative = any(left < 0 or right < 0 for (left, right) in pads)
    if has_negative:
        # Decompose per-axis: each axis with a negative pad becomes its
        # own slice. This keeps streaming-axis crops on `dyn_slice_begin`
        # (which preserves the symbolic streaming shape) and lets the
        # static axes use the regular `slice` op. A single multi-axis
        # `slice` with a concrete `end` would otherwise collapse the
        # streaming axis to a fixed size and break pulse mode downstream.
        pos_pads = []
        for axis_idx, (left, right) in enumerate(pads):
            crop_left = -left if left < 0 else 0
            crop_right = -right if right < 0 else 0
            pos_pads.append((max(left, 0), max(right, 0)))
            if not (crop_left or crop_right):
                continue
            dim_size = input_node.shape[axis_idx]
            # When the export carries any dynamic axes, prefer the
            # streaming-friendly `dyn_slice_begin` op for left-crops:
            # t2n's IR uses concrete trace shapes everywhere so we
            # can't tell from `dim_size` alone whether *this* axis is
            # the streaming one. `dyn_slice_begin` works correctly on
            # static axes too (it just slices to end-of-axis), and the
            # caller pulse-pass requires it on the streaming axis to
            # keep the symbolic dim intact.
            if inference_target.has_dynamic_axes:
                if crop_right > 0:
                    raise T2NErrorNotImplemented(
                        "negative pad on the right side of a dynamic axis "
                        "is not supported (would need a symbolic `end`)"
                    )
                # `dyn_slice_begin` preserves the symbolic dim on this axis.
                inp_ref = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "dyn_slice_begin",
                    inputs=inp_ref,
                    attrs={
                        "axis": axis_idx,
                        "begin": crop_left,
                        "stride": 1,
                    },
                    output_tensor_name_suffix=f"_pad_crop_axis{axis_idx}",
                )
                custom_fragments.extend(
                    ["dyn_slice_begin", "within_bound_index"]
                )
            else:
                inp_ref = add_single_output_op(
                    g,
                    node,
                    name_to_tensor,
                    nnef_op_type="slice",
                    inputs=inp_ref,
                    attrs={
                        "axes": [axis_idx],
                        "begin": [crop_left],
                        "end": [dim_size - crop_right],
                        "stride": [1],
                    },
                    output_tensor_name_suffix=f"_pad_crop_axis{axis_idx}",
                )
        pads = pos_pads

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=inp_ref,
        attrs={"padding": pads, "value": value},
    )
    return custom_fragments or None
