import logging

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
    pick_value_in_rank,
)

LOGGER = logging.getLogger(__name__)


def slice_(g, node, name_to_tensor, torch_graph, **kwargs):
    input_node, axis_node, begin_node, end_node, stride_node = node.inputs

    # we assert for now all node except first are all constant
    dim = axis_node.data

    # we use this since by default pytorch generate max int32 value for end
    begin = pick_value_in_rank(input_node, dim, begin_node.data)
    end = min(
        pick_value_in_rank(input_node, dim, end_node.data),
        input_node.shape[dim],
    )
    assert begin < end

    if (
        begin_node.data == 0
        and end == input_node.shape[dim]
        and stride_node.data == 1
    ):
        LOGGER.debug("Slice is not needed since it have not effect")
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return
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
