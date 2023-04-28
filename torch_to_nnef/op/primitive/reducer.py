from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
    unary_output_op_without_params,
)
from torch_to_nnef.torch_graph import PythonConstant


def _reducer(aten_op_name: str, g, node, name_to_tensor, output_idx: int = 0):
    (input_node, axis_node, keep_dim_node) = node.inputs

    keep_dim = keep_dim_node.data

    onode = node.outputs[output_idx]
    out = add_tensor_variable_node_as_nnef_tensor(
        g,
        onode,
        name_to_tensor,
        prevent_variable=True,
    )
    op_reduce_out = None
    if not keep_dim:
        # apply squeeze
        op_reduce_out_name = f"{onode.export_name}_{aten_op_name}"
        op_reduce_out = NTensor(
            g,
            op_reduce_out_name,
            dtype=onode.np_dtype,
            shape=onode.shape,
        )
        name_to_tensor[op_reduce_out_name] = op_reduce_out

    # can be either 1 or n axes {
    if isinstance(axis_node.data, int):
        axes = [pick_rank(input_node, axis_node.data)]
    else:
        axes = [pick_rank(input_node, _) for _ in axis_node.data]
    #  }
    attribs = {"axes": axes}
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=aten_op_name,
        name=f"{onode.export_name}_{aten_op_name}",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        outputs=out if keep_dim else op_reduce_out,
        attribs=attribs,
    )
    if not keep_dim:
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="squeeze",
            name=f"{onode.export_name}_squeeze",
            inputs=op_reduce_out,
            outputs=out,
            attribs=attribs,
        )


def mean(g, node, name_to_tensor, **kwargs):
    _reducer("mean_reduce", g, node, name_to_tensor)


def reduce_sum(g, node, name_to_tensor, **kwargs):
    _reducer("sum_reduce", g, node, name_to_tensor)


def argmax(g, node, name_to_tensor, **kwargs):
    _reducer("argmax_reduce", g, node, name_to_tensor)


def argmin(g, node, name_to_tensor, **kwargs):
    _reducer("argmin_reduce", g, node, name_to_tensor)


def reduce_any(g, node, name_to_tensor, **kwargs):
    assert len(node.outputs) == 1
    _reducer("any_reduce", g, node, name_to_tensor)


def reduce_all(g, node, name_to_tensor, **kwargs):
    assert len(node.outputs) == 1
    _reducer("all_reduce", g, node, name_to_tensor)


def reduce_max(g, node, name_to_tensor, **kwargs):
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise TorchToNNEFNotImplementedError(
            f"unknown 'max' variant with {n_outputs} outputs used"
        )
    _reducer("max_reduce", g, node, name_to_tensor)
    if n_outputs == 2:
        _reducer("argmax_reduce", g, node, name_to_tensor, output_idx=1)


def reduce_min(g, node, name_to_tensor, **kwargs):
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise TorchToNNEFNotImplementedError(
            f"unknown 'min' variant with {n_outputs} outputs used"
        )
    _reducer("min_reduce", g, node, name_to_tensor)
    if n_outputs == 2:
        _reducer("argmin_reduce", g, node, name_to_tensor, output_idx=1)


def max_(g, node, name_to_tensor, null_ref, **kwargs):
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_max(g, node, name_to_tensor)
    return unary_output_op_without_params(
        nnef_op_type="max",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


def min_(g, node, name_to_tensor, null_ref, **kwargs):
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_min(g, node, name_to_tensor)
    return unary_output_op_without_params(
        nnef_op_type="min",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )
