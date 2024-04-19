import logging

import nnef
import numpy as np

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
    pick_value_in_rank,
)

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register(torch_op_ids=["slice"])
def slice_(
    g,
    node,
    name_to_tensor,
    torch_graph,
    nnef_spec_strict,
    has_dynamic_axes,
    **kwargs,
):
    input_node, axis_node, begin_node, end_node, stride_node = node.inputs

    # we assert for now all node except first are all constant
    dim = axis_node.data

    has_concrete_values = True
    # we use this since by default pytorch generate max int32 value for end
    if begin_node.data is not None:
        begin = pick_value_in_rank(input_node, dim, begin_node.data)
    else:
        has_concrete_values = False
        begin = nnef.Identifier(begin_node.export_name)

    if end_node.data is not None:
        end = pick_value_in_rank(input_node, dim, end_node.data)
    else:
        has_concrete_values = False
        end = nnef.Identifier(end_node.export_name)

    if (
        begin == 0
        and end in [input_node.shape[dim], np.iinfo(np.int64).max]
        and stride_node.data == 1
    ):
        LOGGER.debug("Slice is not needed since it have not effect")
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return []

    if has_concrete_values:
        assert begin < end

    if end_node.data is not None:
        if (
            has_dynamic_axes
            and not nnef_spec_strict
            and end >= input_node.shape[dim]
        ):
            # NOTE: since we can't ensure used dimension is not symbolic
            # we use `tract_core_shape_of`
            input_tensor = get_or_add_tensor_variable_in_nnef(
                g, input_node, name_to_tensor
            )
            shape_tensor_name = f"{input_tensor.name}_shape"
            index_tensor_name = f"{shape_tensor_name}_{dim}"
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_shape_of",
                inputs=input_tensor,
                force_full_output_tensor_name=shape_tensor_name,
            )
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=out,
                attrs={
                    "axes": [0],
                    "begin": [dim],
                    "end": [dim + 1],
                    "stride": [1],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
            end = nnef.Identifier(out.name)
        else:
            end = min(
                end,
                input_node.shape[dim],
            )

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "axes": [pick_rank(input_node, dim)],
            "begin": [begin],
            "end": [end],
            "stride": [stride_node.data],
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def where(g, node, name_to_tensor, **kwargs):
    (condition_node, true_value_node, false_value_node) = node.inputs

    inputs = []
    for snode in [condition_node, true_value_node, false_value_node]:
        name = snode.export_name
        if name in name_to_tensor:
            inputs.append(name_to_tensor[name])
        else:
            snode_ref = add_tensor_variable_node_as_nnef_tensor(
                name_suffix=name,
                node=snode,
                g=g,
                name_to_tensor=name_to_tensor,
            )
            inputs.append(snode_ref)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="select",
        inputs=inputs,
    )


@OP_REGISTRY.register()
def narrow(
    g, node, name_to_tensor, nnef_spec_strict, has_dynamic_axes, **kwargs
):
    """Fancy slice made in PyTorch

    torch.narrow(input, dim, start, length)

    example:

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> torch.narrow(x, 0, 0, 2)
        tensor([[ 1,  2,  3],
                [ 4,  5,  6]])

    """
    input_node, axis_node, start_node, length_node = node.inputs

    assert isinstance(axis_node.data, int)
    assert isinstance(start_node.data, int)
    assert isinstance(length_node.data, int)
    assert length_node.data > 0

    start_idx = pick_value_in_rank(input_node, axis_node.data, start_node.data)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "axes": [pick_rank(input_node, axis_node.data)],
            "begin": [start_idx],
            "end": [start_idx + length_node.data],
            "stride": [1],
        },
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def select(g, node, name_to_tensor, **kwargs):
    input_node, axis_node, index_node = node.inputs
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "axes": [pick_rank(input_node, axis_node.data)],
            "begin": [
                pick_value_in_rank(input_node, axis_node.data, index_node.data)
            ],
            "end": [
                pick_value_in_rank(
                    input_node, axis_node.data, index_node.data + 1
                )
            ],
            "stride": [1],
        },
        output_tensor_name_suffix="_select",
        pass_quantization_params=True,
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=out,
        attrs={"axes": [pick_rank(input_node, axis_node.data)]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register(torch_op_ids=["index"])
def index_(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    """
    fragment gather<?>(
        input: tensor<?>,                 # the tensor to gather from
        indices: tensor<integer>,         # the indices to gather at
        axis: integer = 0 )               # the axis to gather at
    -> ( output: tensor<?> )
    """
    # gather
    input_node, indexes_node = node.inputs
    # input_node = TensorVariable([?], shape=(169,4))
    # indexes_node = FixedTensorList (data=[TensorVariable([?], shape=(2401,))])
    if len(indexes_node.data) > 1:
        raise TorchToNNEFNotImplementedError("index dim>1 not implemented")

    custom_fragments = []
    if nnef_spec_strict:
        op_name = "gather"
    else:
        op_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        op_name,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            get_or_add_tensor_variable_in_nnef(
                g, indexes_node.data[0], name_to_tensor
            ),
        ],
        attrs={
            "axis": 0,
        },
    )
    return custom_fragments


@OP_REGISTRY.register()
def embedding(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        weight_node,
        indices_node,
        _,  # padding_idx_node
        _,  # scale_grad_by_freq_node
        _,  # sparse_node
    ) = node.inputs

    weight_tensor = get_or_add_tensor_variable_in_nnef(
        g, weight_node, name_to_tensor
    )
    indices_tensor = get_or_add_tensor_variable_in_nnef(
        g, indices_node, name_to_tensor
    )
    custom_fragments = []
    if nnef_spec_strict:
        fragment_name = "gather"
    else:
        fragment_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        fragment_name,
        inputs=(weight_tensor, indices_tensor),
        attrs={"axis": 0},
    )
    return custom_fragments


@OP_REGISTRY.register()
def masked_fill(
    g, node, name_to_tensor, nnef_spec_strict, has_dynamic_axes, **kwargs
):
    input_node, mask_node, value_node = node.inputs

    false_value_node = input_node
    false_nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, false_value_node, name_to_tensor
    )
    if not nnef_spec_strict and has_dynamic_axes:
        # repeats on non const not working in tract<=0.21.3
        # so while correct graph notation, tract will fail
        true_value_node = value_node.into_tensor_variable()
        true_value_node.data = true_value_node.data.to(false_value_node.dtype)
        out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_shape_of",
            inputs=false_nnef_tensor,
            output_tensor_name_suffix="shape_of_false",
        )
        true_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tile",
            inputs=get_or_add_tensor_variable_in_nnef(
                g, true_value_node, name_to_tensor, name_suffix="true_scalar"
            ),
            attrs={"repeats": nnef.Identifier(str(out.name))},
            output_tensor_name_suffix="true_expanded",
        )
    else:
        # Static expansion
        true_value_node = value_node.into_tensor_variable()
        true_value_node.data = true_value_node.data.to(
            false_value_node.dtype
        ).repeat(false_value_node.shape)
        true_value_node.dtype = false_value_node.dtype
        true_nnef_tensor = get_or_add_tensor_variable_in_nnef(
            g, true_value_node, name_to_tensor
        )

    # tract need float where ?
    # mask_node.data = mask_node.data.float()
    # mask_node.dtype = mask_node.data.dtype
    condition_node = mask_node

    inputs = [
        get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
        for _ in [condition_node]
    ]
    inputs += [true_nnef_tensor, false_nnef_tensor]

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="select",
        inputs=inputs,
    )
