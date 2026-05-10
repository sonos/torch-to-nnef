import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    cast_and_add_nnef_operation,
    pick_axis,
)
from torch_to_nnef.torch_graph import PythonConstant

OP_REGISTRY = AtenOpRegistry()


def reducer_helper(aten_op_name: str, node, op_helper, output_idx: int = 0):
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor

    # PyTorch's reduction aten ops have variants with different arity:
    # - `aten::sum(input)`: 1 input
    # - `aten::sum.dim_IntList(input, dim)`: 2 inputs
    # - `aten::sum.dim_IntList(input, dim, keepdim)`: 3 inputs
    # - `aten::prod.dim_int(input, dim, keepdim, *, dtype=None)`: 4
    # We only need `input`, `dim`, and `keepdim`; the trailing
    # `dtype` (when present) is honored by PyTorch upstream and the
    # exported graph already carries the post-cast output dtype, so we
    # can safely ignore it here.
    n_inputs = len(node.inputs)
    if n_inputs == 2:
        (input_node, axis_node) = node.inputs
        keep_dim = False
    elif n_inputs >= 3:
        input_node, axis_node, keep_dim_node = node.inputs[:3]
        keep_dim = keep_dim_node.data
    else:
        raise T2NErrorNotImplemented(
            f"reducer with {n_inputs} inputs (expected 2 or >=3)"
        )

    onode = node.outputs[output_idx]
    out = op_helper.get_or_add_tensor_variable_in_nnef(
        onode,
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
        axes = [pick_axis(input_node, axis_node.data)]
    else:
        if axis_node.data is None:
            axes = [pick_axis(input_node, _) for _ in range(input_node.rank)]
        else:
            axes = [pick_axis(input_node, _) for _ in axis_node.data]
    #  }
    tensor_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if (
        input_node.dtype == torch.bool
        and isinstance(op_helper.inference_target, TractNNEF)
        and aten_op_name in ["sum_reduce", "mean_reduce"]
    ):
        dtype_str = TORCH_DTYPE_TO_TRACT_STR[torch.int64]
        tensor_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_cast",
            inputs=[tensor_ref],
            attrs={
                "to": dtype_str,
            },
            output_tensor_name_suffix=f"as_{dtype_str}",
        )
    attribs = {"axes": axes}
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=aten_op_name,
        name=f"{onode.export_name}_{aten_op_name}",
        inputs=tensor_ref,
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


@OP_REGISTRY.register()
def mean(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:mean' to NNEF."""
    reducer_helper("mean_reduce", node, op_helper)


@OP_REGISTRY.register(torch_op_ids=["reduce_sum", "sum"])
def reduce_sum(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:reduce_sum', 'aten:sum' to NNEF."""
    reducer_helper("sum_reduce", node, op_helper)


@OP_REGISTRY.register()
def argmax(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:argmax' to NNEF."""
    reducer_helper("argmax_reduce", node, op_helper)


@OP_REGISTRY.register()
def argmin(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:argmin' to NNEF."""
    reducer_helper("argmin_reduce", node, op_helper)


@OP_REGISTRY.register(torch_op_ids=["reduce_any", "any"])
def reduce_any(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:reduce_any', 'aten:any' to NNEF."""
    assert len(node.outputs) == 1
    reducer_helper("any_reduce", node, op_helper)


@OP_REGISTRY.register(torch_op_ids=["reduce_all", "all"])
def reduce_all(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:reduce_all', 'aten:all' to NNEF."""
    assert len(node.outputs) == 1
    reducer_helper("all_reduce", node, op_helper)


@OP_REGISTRY.register(torch_op_ids=["reduce_max", "amax"])
def reduce_max(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:reduce_max', 'aten:amax' to NNEF."""
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise T2NErrorNotImplemented(
            f"unknown 'max' variant with {n_outputs} outputs used"
        )
    reducer_helper("max_reduce", node, op_helper)
    if n_outputs == 2:
        reducer_helper("argmax_reduce", node, op_helper, output_idx=1)


@OP_REGISTRY.register(torch_op_ids=["reduce_min", "amin"])
def reduce_min(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:reduce_min', 'aten:amin' to NNEF."""
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise T2NErrorNotImplemented(
            f"unknown 'min' variant with {n_outputs} outputs used"
        )
    reducer_helper("min_reduce", node, op_helper)
    if n_outputs == 2:
        reducer_helper("argmin_reduce", node, op_helper, output_idx=1)


@OP_REGISTRY.register(torch_op_ids=["max"])
def max_(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:max' to NNEF."""
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_max(node, op_helper)
    return op_helper.unary_output_op_without_attr(nnef_op_type="max", node=node)


@OP_REGISTRY.register(torch_op_ids=["min"])
def min_(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:min' to NNEF."""
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_min(node, op_helper)
    return op_helper.unary_output_op_without_attr(nnef_op_type="min", node=node)


@OP_REGISTRY.register()
def prod(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:prod' to NNEF."""
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    reducer_helper("tract_core_product_reduce", node, op_helper)
    return ["tract_core"]


@OP_REGISTRY.register()
def count_nonzero(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:count_nonzero' to NNEF.

    `count_nonzero(input, dim=None)` returns the number of non-zero
    elements in `input` along `dim` (or globally when `dim=None`) as
    an int64 scalar / reduced tensor. Decomposed as
    `ne(x, 0) -> tract_core_cast(i64) -> sum_reduce(axes) -> squeeze`.

    Intermediate NTensors are built explicitly with their kept-dim
    shapes (rather than going through `add_single_output_op_from_nnef_tensors`
    which inherits `node.outputs[0].shape`). The shared helper
    declares the rank-0 final shape on every intermediate, which then
    trips the rank-align pass: `ne(input, scalar_zero)` sees both
    operands as "scalar-like" and squeezes the rank-1 input to scalar
    before evaluating, panicking the downstream `sum_reduce`.
    """
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor
    input_node = node.inputs[0]
    dim_node = node.inputs[1] if len(node.inputs) > 1 else None
    rank = input_node.rank
    if dim_node is None or dim_node.data is None:
        axes = list(range(rank))
    elif isinstance(dim_node.data, int):
        axes = [pick_axis(input_node, dim_node.data)]
    else:
        axes = [pick_axis(input_node, d) for d in dim_node.data]

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    onode = node.outputs[0]
    base = onode.export_name
    # The dtype on intermediate NTensors is informational; tract reads
    # the actual datum type from the op output. Stamping the input's
    # dtype is fine -- the cast op below switches it to i64.
    np_dtype = input_node.np_dtype

    def _intermediate(name, shape):
        t = NTensor(g, name, dtype=np_dtype, shape=tuple(shape))
        name_to_tensor[name] = t
        return t

    zero_const = PythonConstant(
        name=f"{base}_cnz_zero",
        data=torch.zeros((), dtype=input_node.dtype),
    )
    zero_ref = op_helper.get_or_add_tensor_variable_in_nnef(zero_const)
    full_shape = list(input_node.shape)
    mask = _intermediate(f"{base}_cnz_mask", full_shape)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="ne",
        name=f"{mask.name}_op",
        inputs=(inp_ref, zero_ref),
        outputs=(mask,),
        attribs={},
    )
    int_t = _intermediate(f"{base}_cnz_int", full_shape)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="tract_core_cast",
        name=f"{int_t.name}_op",
        inputs=(mask,),
        outputs=(int_t,),
        attribs={"to": TORCH_DTYPE_TO_TRACT_STR[torch.int64]},
    )
    out_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        onode, prevent_variable=True
    )
    if rank == 0:
        # Edge case: rank-0 input. `sum_reduce` with empty axes is a
        # no-op identity; emit a reshape to materialize the final
        # NTensor with the right name.
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="reshape",
            name=f"{base}_cnz_reshape",
            inputs=(int_t,),
            outputs=(out_ref,),
            attribs={"shape": []},
        )
        return ["tract_core"]
    kd_shape = list(full_shape)
    for ax in axes:
        kd_shape[ax] = 1
    kd = _intermediate(f"{base}_cnz_kd", kd_shape)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="sum_reduce",
        name=f"{kd.name}_op",
        inputs=(int_t,),
        outputs=(kd,),
        attribs={"axes": axes},
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="squeeze",
        name=f"{base}_cnz_squeeze",
        inputs=(kd,),
        outputs=(out_ref,),
        attribs={"axes": axes},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def aminmax(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:aminmax' to NNEF.

    `aminmax(input, dim=None, keepdim=False)` returns a `(min, max)`
    tuple. Decomposed into two independent reductions: `min_reduce`
    into `outputs[0]` and `max_reduce` into `outputs[1]`. Squeeze
    handling is shared with the rest of the reducer family via
    `reducer_helper`.
    """
    assert len(node.outputs) == 2
    reducer_helper("min_reduce", node, op_helper, output_idx=0)
    reducer_helper("max_reduce", node, op_helper, output_idx=1)
