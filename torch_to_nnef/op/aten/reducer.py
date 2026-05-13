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


def _emit_nan_replace(input_node, op_helper, node, replacement, suffix):
    """Emit `select(isnan(x), replacement, x)`.

    Uses the IEEE-754 invariant `NaN != NaN` for the NaN test so the
    decomposition stays in NNEF stdlib (no `tract_core_is_nan`
    dependency).
    """
    out_dtype = input_node.dtype or torch.float32
    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    replacement_const = PythonConstant(
        name=f"{node.outputs[0].export_name}_{suffix}_const",
        data=torch.tensor(float(replacement), dtype=out_dtype),
    )
    replacement_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        replacement_const
    )
    is_nan = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "ne",
        inputs=[inp_ref, inp_ref],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix=f"_{suffix}_isnan",
    )
    return op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "select",
        inputs=[is_nan, replacement_ref, inp_ref],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix=f"_{suffix}_clean",
    ), is_nan


def _resolve_reduce_axes(input_node, axis_node):
    """Mirror `reducer_helper`'s axis resolution -- int / list / None."""
    if isinstance(axis_node.data, int):
        return [pick_axis(input_node, axis_node.data)]
    if axis_node.data is None:
        return [pick_axis(input_node, i) for i in range(input_node.rank)]
    return [pick_axis(input_node, i) for i in axis_node.data]


def _nan_reduce(node, op_helper, mode: str):
    """Shared core for `nansum` / `nanmean`.

    aten signature: `(self, dim?, keepdim=False, *, dtype=None)`.
    Decomposes to `sum_reduce` on a NaN-replaced copy of the input,
    plus (for `nanmean`) a `sum_reduce` of the non-NaN mask and a
    division.
    """
    n_inputs = len(node.inputs)
    if n_inputs not in (2, 3, 4):
        raise T2NErrorNotImplemented(
            f"nan{mode} with {n_inputs} inputs (expected 2-4)"
        )
    input_node, axis_node = node.inputs[:2]
    keep_dim = (
        node.inputs[2].data
        if n_inputs >= 3 and isinstance(node.inputs[2].data, bool)
        else False
    )
    axes = _resolve_reduce_axes(input_node, axis_node)

    clean_ref, is_nan_ref = _emit_nan_replace(
        input_node, op_helper, node, replacement=0.0, suffix=f"nan{mode}"
    )

    onode = node.outputs[0]
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor

    # The sum stage writes the final output for `nansum`; for
    # `nanmean` it lands on an intermediate that gets divided next.
    needs_div = mode == "mean"
    sum_out_suffix = "_nanmean_sum" if needs_div else ""
    sum_target = op_helper.get_or_add_tensor_variable_in_nnef(
        onode,
        prevent_variable=True,
        name_suffix=sum_out_suffix,
    )
    sum_pre_squeeze = sum_target
    if not keep_dim:
        sum_pre_name = f"{onode.export_name}_nan{mode}_reduce_unsqueezed"
        sum_pre_squeeze = NTensor(
            g,
            sum_pre_name,
            dtype=onode.np_dtype,
            shape=onode.shape,
        )
        name_to_tensor[sum_pre_name] = sum_pre_squeeze
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="sum_reduce",
        name=f"{onode.export_name}_nan{mode}_reduce",
        inputs=clean_ref,
        outputs=sum_pre_squeeze if not keep_dim else sum_target,
        attribs={"axes": axes},
    )
    if not keep_dim:
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="squeeze",
            name=f"{onode.export_name}_nan{mode}_squeeze",
            inputs=sum_pre_squeeze,
            outputs=sum_target,
            attribs={"axes": axes},
        )
    if not needs_div:
        return

    # nanmean: divide the sum by the per-axis count of non-NaN inputs.
    # The mask is `1 - is_nan` cast to float; sum-reduce gives the
    # count.
    one_const = PythonConstant(
        name=f"{onode.export_name}_nanmean_one",
        data=torch.tensor(1.0, dtype=input_node.dtype or torch.float32),
    )
    zero_const = PythonConstant(
        name=f"{onode.export_name}_nanmean_zero",
        data=torch.tensor(0.0, dtype=input_node.dtype or torch.float32),
    )
    one_ref = op_helper.get_or_add_tensor_variable_in_nnef(one_const)
    zero_ref = op_helper.get_or_add_tensor_variable_in_nnef(zero_const)
    mask_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "select",
        inputs=[is_nan_ref, zero_ref, one_ref],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_nanmean_mask",
    )
    count_target_name = f"{onode.export_name}_nanmean_count"
    count_pre = NTensor(
        g, count_target_name, dtype=onode.np_dtype, shape=onode.shape
    )
    name_to_tensor[count_target_name] = count_pre
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="sum_reduce",
        name=f"{onode.export_name}_nanmean_count_reduce",
        inputs=mask_ref,
        outputs=count_pre,
        attribs={"axes": axes},
    )
    if not keep_dim:
        count_squeezed_name = f"{onode.export_name}_nanmean_count_squeezed"
        count_squeezed = NTensor(
            g, count_squeezed_name, dtype=onode.np_dtype, shape=onode.shape
        )
        name_to_tensor[count_squeezed_name] = count_squeezed
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="squeeze",
            name=f"{onode.export_name}_nanmean_count_squeeze",
            inputs=count_pre,
            outputs=count_squeezed,
            attribs={"axes": axes},
        )
        count_pre = count_squeezed
    final_target = op_helper.get_or_add_tensor_variable_in_nnef(
        onode, prevent_variable=True
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="div",
        name=f"{onode.export_name}_nanmean_div",
        inputs=(sum_target, count_pre),
        outputs=final_target,
        attribs={},
    )


@OP_REGISTRY.register()
def nansum(node, op_helper, **kwargs):
    """Map PyTorch: `aten::nansum` -> NaN-skipping sum.

    Decomposed as `sum_reduce(select(isnan(x), 0, x))`. NaN detection
    via `ne(x, x)` (IEEE-754 invariant) so the decomposition only
    touches NNEF stdlib ops.
    """
    _nan_reduce(node, op_helper, mode="sum")


@OP_REGISTRY.register()
def nanmean(node, op_helper, **kwargs):
    """Map PyTorch: `aten::nanmean` -> NaN-skipping mean.

    Sum of NaN-replaced input divided by the count of non-NaN inputs
    along the reduce axes.
    """
    _nan_reduce(node, op_helper, mode="mean")


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
