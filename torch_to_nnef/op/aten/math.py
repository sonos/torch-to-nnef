import logging

import numpy as np
import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR, dtype_is_whole_number
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten.complex import tract_complex_support
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    cast_and_add_nnef_operation,
    pick_axis,
    unary_input_output_op_with_constant,
)
from torch_to_nnef.torch_graph import PythonConstant
from torch_to_nnef.torch_graph.ir_data import TensorVariable

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


def _div_constant_result(numerator, divisor, output_dtype) -> torch.Tensor:
    """Return `numerator / divisor` as a tensor of `output_dtype`.

    Both inputs may be Python scalars (after frozen-graph constant
    folding); their division returns a Python scalar that has no `.to`,
    so wrap before casting.
    """
    result = numerator / divisor
    if not isinstance(result, torch.Tensor):
        return torch.tensor(result, dtype=output_dtype)
    return result.to(output_dtype)


@OP_REGISTRY.register()
def div(node, op_helper, inference_target, torch_graph, **kwargs):
    """Map PyTorch: 'aten:div' to NNEF."""
    input_node = node.inputs[0]
    divisor_node = node.inputs[1]
    suffix_div_op_output = ""
    rounding_mode = None

    if input_node.data is not None and divisor_node.data is not None:
        node.outputs[0].set_data(
            _div_constant_result(
                input_node.data, divisor_node.data, node.outputs[0].dtype
            )
        )
        return []

    if remap_if_neutral_op(torch_graph, node, divisor_node, input_node):
        return []

    used_custom_fragment = []

    for c_node in [input_node, divisor_node]:
        if (  # in case mixing precision
            any(
                not isinstance(nod, PythonConstant)
                and nod.dtype.is_floating_point
                for nod in [input_node, divisor_node]
            )
            and len({input_node.dtype, divisor_node.dtype}) == 2
        ):
            LOGGER.warning(
                "div: Mixing input of 2 different dtype: %s force cast to f32",
                (input_node.dtype, divisor_node.dtype),
            )
            c_node.cast_float_inplace()

    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    divisor_tensor = op_helper.get_or_add_tensor_variable_in_nnef(divisor_node)
    io_casting_with_dtype = None

    if isinstance(inference_target, TractNNEF):
        if dtype_is_whole_number(input_tensor.dtype):
            input_tensor, cf = op_helper.cast_to_if_not_dtype_and_variable(
                node,
                input_tensor,
                cast_to=np.float32,
                suffix="input_forced_cast",
            )
            used_custom_fragment.extend(cf)
        if dtype_is_whole_number(divisor_tensor.dtype):
            divisor_tensor, cf = op_helper.cast_to_if_not_dtype_and_variable(
                node,
                divisor_tensor,
                cast_to=np.float32,
                suffix="divisor_forced_cast",
            )
            used_custom_fragment.extend(cf)

    if len(node.inputs) == 3:
        rounding_mode = node.inputs[2].data
        # `rounding_mode` may be None even with 3 inputs (PyTorch passes the
        # literal None when called as `div(a, b, rounding_mode=None)`).
        # In that case the result is true float division and we must NOT
        # cast the output to int64.
        if rounding_mode is not None and isinstance(
            inference_target, TractNNEF
        ):
            # PyTorch preserves input dtype across rounded division:
            # `div(float, float, trunc) -> float`,
            # `div(int, int, trunc) -> int64`.
            # We cast to int64 only when the traced output is integer
            # (originally added to dodge U64 propagation for dim math).
            output_torch_dtype = node.outputs[0].dtype
            if not output_torch_dtype.is_floating_point:
                io_casting_with_dtype = np.int64
                suffix_div_op_output = "div"

    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix=suffix_div_op_output,
    )

    if rounding_mode:
        out = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            rounding_mode,
            inputs=out,
            output_tensor_name_suffix=""
            if io_casting_with_dtype is None
            else rounding_mode,
        )
        if rounding_mode == "trunc":
            used_custom_fragment.append(rounding_mode)

    if io_casting_with_dtype is not None:
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(
                "What NNEF compliance mean in such case ?", inference_target
            )
        _, custom_fragments = op_helper.cast_to_if_not_dtype_and_variable(
            node=node,
            nnef_tensor=out,
            cast_to=io_casting_with_dtype,
        )
        used_custom_fragment += custom_fragments
    return list(set(used_custom_fragment))


@OP_REGISTRY.register()
def floor_divide(node, op_helper, inference_target, torch_graph, **kwargs):
    """Map PyTorch: 'aten:floor_divide' to NNEF."""
    input_node, divisor_node = node.inputs
    if (
        input_node.data
        and divisor_node.data
        and not inference_target.has_dynamic_axes
    ):
        # avoid graph computation since static
        idata = float(
            input_node.data.tolist()
            if isinstance(input_node, TensorVariable)
            else input_node.data
        )
        ddata = float(
            divisor_node.data.tolist()
            if isinstance(divisor_node, TensorVariable)
            else divisor_node.data
        )
        torch_graph.remap_node(
            node.outputs[0],
            PythonConstant(name=node.outputs[0].name, data=idata // ddata),
        )
        return []
    # for c_node in [input_node, divisor_node]:
    #     c_node.cast_float_inplace()

    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    divisor_tensor = op_helper.get_or_add_tensor_variable_in_nnef(divisor_node)

    need_floor = not (
        dtype_is_whole_number(input_tensor.dtype)
        and dtype_is_whole_number(divisor_tensor.dtype)
    )

    suffix = ""
    if need_floor:
        suffix = "div"
    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix=suffix,
    )
    if need_floor:
        op_helper.add_single_output_op_from_nnef_tensors(
            node, "floor", inputs=out
        )
    return []


@OP_REGISTRY.register()
def trunc(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:trunc' to NNEF."""
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "trunc",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0]),
    )
    return ["trunc"]


@OP_REGISTRY.register()
def outer(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:outer' to NNEF.

    `torch.outer(a, b)` over 1-D inputs is `a[:, None] * b[None, :]`.
    Lower to two unsqueezes and a broadcasting `mul`.

    Axes are kept positive. Tract's NNEF unsqueeze deserializer
    (`tract_core::ops::change_axes::AxisOp::change_shape`) does not
    normalize negative axes and panics with `smallvec: index exceeds
    length`; verified across tract 0.20.22 through 0.23.0-dev.5. This
    matches the wider t2n convention: the dedicated `unsqueeze` op
    handler also normalizes via `pick_axis`.
    """
    a_node, b_node = node.inputs
    a = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    a_col = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=a,
        attrs={"axes": [1]},
        output_tensor_name_suffix="_col",
    )
    b_row = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=b,
        attrs={"axes": [0]},
        output_tensor_name_suffix="_row",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=(a_col, b_row),
    )


@OP_REGISTRY.register(torch_op_ids=["pow"])
def pow_(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:pow' to NNEF."""
    (input_node, exponent_node) = node.inputs
    inputs = [op_helper.get_or_add_tensor_variable_in_nnef(input_node)]
    # Scalar 2 / -2 only: isinstance check skips the truthiness branch on
    # tensor-valued exponents (which would raise "ambiguous").
    exp_data = exponent_node.data
    if isinstance(exp_data, (int, float)) and exp_data in (2, -2):
        op_type = "sqr" if exp_data == 2 else "rsqr"
    else:
        op_type = "pow"
        inputs += [op_helper.get_or_add_tensor_variable_in_nnef(exponent_node)]

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        op_type,
        inputs=inputs,
    )


@OP_REGISTRY.register(torch_op_ids=["round"])
def round_(inference_target, **kwargs):
    """Map PyTorch: 'aten:round' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        LOGGER.warning(
            "round: Spec definition of round in NNEF does not follow IEEE, "
            "so it will not be exactly same behavior"
        )
        unary_input_output_op_with_constant("round", **kwargs)
        return []
    unary_input_output_op_with_constant("tract_core_round_even", **kwargs)
    return ["tract_core"]


def remap_if_neutral_op(torch_graph, node, a, b):
    if a.data is not None and (a.into_tensor_variable().data == 1.0).all():
        torch_graph.remap_node(node.outputs[0], b)
        return True
    return False


@OP_REGISTRY.register()
def mul(node, op_helper, torch_graph, **kwargs):
    """Map PyTorch: 'aten:mul' to NNEF."""
    input_node = node.inputs[0]
    other_node = node.inputs[1]

    if input_node.data is not None and other_node.data is not None:
        # When one operand is a float scalar (e.g. 1/sqrt(d) attention scaling)
        # and the other an int64 shape value, torch promotes to float, but the
        # traced output dtype may still be int64: set_data's dtype validation
        # would then fail. Cast to the declared dtype for tensor results;
        # Python scalars (int * int -> int) are passed through as-is.
        result = input_node.data * other_node.data
        if isinstance(result, torch.Tensor):
            result = result.to(node.outputs[0].dtype)
        node.outputs[0].set_data(result)
        return
    if remap_if_neutral_op(
        torch_graph, node, input_node, other_node
    ) or remap_if_neutral_op(torch_graph, node, other_node, input_node):
        return

    inputs = []
    for c_node in [input_node, other_node]:
        if isinstance(c_node, PythonConstant):
            # because torch.ops.aten.mul(float, tensor(float))
            # give complex number
            c_node = c_node.into_tensor_variable()
        inputs.append(op_helper.get_or_add_tensor_variable_in_nnef(c_node))
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=inputs,
    )


@OP_REGISTRY.register()
def remainder(node, op_helper, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:remainder' to NNEF."""
    input_node, other_node = node.inputs
    if all(
        isinstance(node, PythonConstant) for node in [input_node, other_node]
    ):
        torch_graph.remap_node(
            from_node=node.outputs[0],
            to_node=PythonConstant(
                name=node.outputs[0].export_name,
                data=input_node.data % other_node.data,
            ),
        )
        return []
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "remainder",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(_)
            for _ in [input_node, other_node]
        ],
    )
    return ["remainder"]


def _resolve_operand(op_helper, c_node):
    """Materialize an `aten:add` / `aten:sub` operand as an NNEF tensor."""
    if isinstance(c_node, PythonConstant):
        c_node = c_node.into_tensor_variable()
    return op_helper.get_or_add_tensor_variable_in_nnef(c_node)


def _alpha_is_default(alpha_node) -> bool:
    """Return True when alpha is absent or equals 1.0 (the PyTorch default)."""
    if alpha_node is None:
        return True
    if not isinstance(alpha_node, PythonConstant):
        return False
    return alpha_node.data is not None and float(alpha_node.data) == 1.0


def _emit_alpha_scaled_other(op_helper, node, other_tensor, alpha_node):
    """Return `other * alpha` as a fresh NNEF tensor.

    Declares the intermediate NNEF tensor with `other`'s shape explicitly:
    `add_single_output_op_from_nnef_tensors` would reuse
    `node.outputs[0].shape` (which is the FINAL broadcast shape of the
    add/sub), and tract refuses to broadcast a tensor of `other.shape`
    declared with that final-broadcast shape.
    """
    if isinstance(alpha_node, PythonConstant):
        alpha_node.set_data(float(alpha_node.data))
        alpha_node = alpha_node.into_tensor_variable()
    alpha_tensor = op_helper.get_or_add_tensor_variable_in_nnef(alpha_node)
    scaled_name = f"{node.outputs[0].export_name}_alpha_scaled"
    scaled_other = NTensor(
        op_helper.g,
        scaled_name,
        dtype=other_tensor.dtype,
        shape=other_tensor.shape,
    )
    op_helper.name_to_tensor[scaled_name] = scaled_other
    cast_and_add_nnef_operation(
        graph=op_helper.g,
        name_to_tensor=op_helper.name_to_tensor,
        type="mul",
        name=scaled_name,
        inputs=(other_tensor, alpha_tensor),
        outputs=(scaled_other,),
        attribs={},
    )
    return scaled_other


def _add_or_sub_with_alpha(nnef_op_name: str, node, op_helper, **_):
    """Shared body for `aten:add` and `aten:sub` (both honor `alpha`).

    PyTorch's signatures are::

        add(input, other, *, alpha=1) -> input + alpha * other
        sub(input, other, *, alpha=1) -> input - alpha * other

    For the default `alpha == 1` we emit a single NNEF `add` / `sub`
    op. For non-default alpha we decompose to `mul(other, alpha)` then
    `nnef_op_name(input, scaled_other)`: this avoids needing a custom
    NNEF op variant that takes `alpha` as an attribute.
    """
    if len(node.inputs) == 3:
        input_node, other_node, alpha_node = node.inputs
    else:
        # Some aten variants don't carry alpha (e.g. add.Scalar without it
        # being explicitly emitted).
        input_node, other_node = node.inputs
        alpha_node = None

    input_tensor = _resolve_operand(op_helper, input_node)
    other_tensor = _resolve_operand(op_helper, other_node)

    if _alpha_is_default(alpha_node):
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            nnef_op_name,
            inputs=(input_tensor, other_tensor),
        )
        return

    scaled_other = _emit_alpha_scaled_other(
        op_helper, node, other_tensor, alpha_node
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_name,
        inputs=(input_tensor, scaled_other),
    )


@OP_REGISTRY.register(torch_op_ids=["add"])
def add(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:add' to NNEF, honoring the `alpha` parameter."""
    _add_or_sub_with_alpha("add", node, op_helper, **kwargs)


@OP_REGISTRY.register(torch_op_ids=["sub"])
def sub(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:sub' to NNEF, honoring the `alpha` parameter."""
    _add_or_sub_with_alpha("sub", node, op_helper, **kwargs)


@OP_REGISTRY.register()
def rsub(node, op_helper, torch_graph, **kwargs):
    """Map PyTorch: 'aten:rsub' to NNEF."""
    input_node, other_node, alpha_node = node.inputs
    if all(
        isinstance(_, PythonConstant)
        for _ in [input_node, other_node, alpha_node]
    ):
        LOGGER.debug("Slice is not needed since it have not effect")
        torch_graph.remap_node(
            from_node=node.outputs[0],
            to_node=PythonConstant(
                name=node.outputs[0].export_name,
                data=int(
                    input_node.data * -1.0 * alpha_node.data + other_node.data
                ),
            ),
        )
        return []
    if isinstance(alpha_node, PythonConstant):
        alpha_node.set_data(float(alpha_node.data))
    inputs = [
        op_helper.get_or_add_tensor_variable_in_nnef(_)
        for _ in [input_node, other_node]
    ]
    for idx, inp in enumerate(inputs):
        inputs[idx] = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_cast",
            inputs=[inp],
            attrs={"to": "f32"},
            force_full_output_tensor_name=f"{inp.name}_as_f32",
        )

    out_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "rsub",
        inputs=inputs,
        attrs={"alpha": alpha_node.data},
        output_tensor_name_suffix="rsub",
    )
    o_dtype = node.outputs[0].dtype
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_cast",
        inputs=[out_ref],
        attrs={"to": TORCH_DTYPE_TO_TRACT_STR[o_dtype]},
    )

    return ["rsub"]


@OP_REGISTRY.register(torch_op_ids=["abs"])
def _abs(
    node,
    op_helper,
    inference_target,
    torch_graph,
    **kwargs,
):
    """Map PyTorch: 'aten:abs' to NNEF."""
    if node.inputs[0].dtype in [torch.complex64, torch.complex128]:
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(
                "NNEF compliance does not allow complex"
            )
        input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(
            node.inputs[0]
        )
        # to real, pow(2), slice both, add 2 tensors, rsqr
        if tract_complex_support(inference_target):
            input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "tract_core_complex_to_inner_dim",
                inputs=input_tensor,
                output_tensor_name_suffix="complex_abs_to_real",
            )

        input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "sqr",
            inputs=input_tensor,
            output_tensor_name_suffix="complex_abs_sqr",
        )
        input_tensor_real = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "slice",
            inputs=input_tensor,
            attrs={
                "axes": [len(input_tensor.shape)],
                "begin": [0],
                "end": [1],
                "stride": [1],
            },
            output_tensor_name_suffix="complex_abs_slice_real",
        )
        input_tensor_imag = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "slice",
            inputs=input_tensor,
            attrs={
                "axes": [len(input_tensor.shape)],
                "begin": [1],
                "end": [2],
                "stride": [1],
            },
            output_tensor_name_suffix="complex_abs_slice_imag",
        )

        input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "add",
            inputs=[input_tensor_real, input_tensor_imag],
            output_tensor_name_suffix="complex_abs_add",
        )
        input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "sqrt",
            inputs=input_tensor,
            output_tensor_name_suffix="complex_abs_sqrt",
        )
        input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "squeeze",
            inputs=input_tensor,
            attrs={"axes": [len(input_tensor.shape)]},
        )
        return []
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type="abs",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(_)
            if _ and not (isinstance(_.data, str) and _.data == "none")
            else op_helper.null_ref
            for _ in node.inputs
        ],
    )
    return []


@OP_REGISTRY.register()
def log10(node, op_helper, **kwargs):
    """Mul val may not be good enough."""
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    # maybe better puting this in the graph to avoid precision loss
    mul_val = 1 / np.log(10)
    input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "log",
        inputs=input_tensor,
        output_tensor_name_suffix="pre_log10",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=input_tensor,
        attrs={"y": mul_val},
    )


@OP_REGISTRY.register()
def log1p(node, op_helper, **kwargs):
    """aten::log1p."""
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "log1p",
        inputs=input_tensor,
    )
    return ["log1p"]


@OP_REGISTRY.register()
def atan2(node, op_helper, **kwargs):
    """aten::atan2."""
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    other_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "atan2",
        inputs=(input_tensor, other_tensor),
    )
    return ["atan2"]


@OP_REGISTRY.register()
def expm1(node, op_helper, **kwargs):
    """aten::exp1m."""
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "expm1",
        inputs=input_tensor,
    )
    return ["expm1"]


@OP_REGISTRY.register()
def square(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:square' to NNEF.

    `x.square()` is `x * x`; the dedicated NNEF `sqr` op exists in some
    runtimes but the simplest portable form is a `mul` with the same
    tensor on both sides.
    """
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=[input_tensor, input_tensor],
        force_consistent_inputs_shapes=False,
    )
    return []


@OP_REGISTRY.register()
def nan_to_num(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:nan_to_num' to NNEF.

    Decomposed to pure NNEF stdlib (`ne` for NaN via the IEEE-754
    ``NaN != NaN`` invariant, `gt`/`lt` against the dtype's finite
    range for +/-inf, plus `select`). No tract extension needed.

    Defaults match torch: NaN -> 0; +inf -> finfo.max; -inf -> finfo.min.
    """
    input_node, nan_node, posinf_node, neginf_node = node.inputs
    out_dtype = input_node.dtype or torch.float32
    finfo = torch.finfo(out_dtype)
    nan_val = float(nan_node.data) if nan_node.data is not None else 0.0
    posinf_val = (
        float(posinf_node.data) if posinf_node.data is not None else finfo.max
    )
    neginf_val = (
        float(neginf_node.data) if neginf_node.data is not None else finfo.min
    )

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

    def _scalar(value: float, suffix: str):
        const = PythonConstant(
            name=f"{node.outputs[0].name}_n2n_{suffix}",
            data=torch.tensor(value, dtype=out_dtype),
        )
        return op_helper.get_or_add_tensor_variable_in_nnef(const)

    nan_t = _scalar(nan_val, "nan")
    posinf_t = _scalar(posinf_val, "posinf")
    neginf_t = _scalar(neginf_val, "neginf")
    finfo_max_t = _scalar(finfo.max, "finfo_max")
    finfo_min_t = _scalar(finfo.min, "finfo_min")

    # NaN check via the IEEE-754 ``NaN != NaN`` invariant: pure NNEF
    # stdlib so this works on every TractNNEF version (vs the dev-only
    # ``tract_core_is_nan`` extension).
    is_nan_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "ne",
        inputs=[inp_ref, inp_ref],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_n2n_isnan",
    )
    no_nan_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "select",
        inputs=[is_nan_ref, nan_t, inp_ref],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_n2n_no_nan",
    )

    # +inf is the only post-NaN-replace value strictly greater than the
    # dtype's max finite, so `gt(no_nan, finfo.max)` flags +inf alone.
    is_posinf_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "gt",
        inputs=[no_nan_ref, finfo_max_t],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_n2n_is_posinf",
    )
    no_posinf_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "select",
        inputs=[is_posinf_ref, posinf_t, no_nan_ref],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_n2n_no_posinf",
    )

    is_neginf_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "lt",
        inputs=[no_posinf_ref, finfo_min_t],
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_n2n_is_neginf",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "select",
        inputs=[is_neginf_ref, neginf_t, no_posinf_ref],
        force_consistent_inputs_shapes=False,
    )
    return []


@OP_REGISTRY.register()
def cosine_similarity(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:cosine_similarity' to NNEF via a fragment.

    The fragment lives in `op/fragment/cosine_similarity.nnef` and is
    composed only of NNEF stdlib ops, so no tract-side change is needed.

    Negative `dim` is normalized to a non-negative index via
    `pick_axis` before reaching the fragment: under dynamic-axes
    mode tract's reduce path crashes (index out of bounds at
    core/src/ops/nn/reduce.rs) when handed a negative axis against a
    symbolic-rank tensor.
    """
    input_a_node = node.inputs[0]
    input_a = op_helper.get_or_add_tensor_variable_in_nnef(input_a_node)
    input_b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    dim_node = node.inputs[2]
    axis = pick_axis(input_a_node, dim_node.data)
    eps_node = node.inputs[3] if len(node.inputs) > 3 else None
    eps_val = (
        float(eps_node.data)
        if eps_node is not None and eps_node.data is not None
        else 1e-8
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "cosine_similarity",
        inputs=[input_a, input_b],
        attrs={"axis": axis, "eps": eps_val},
        force_consistent_inputs_shapes=False,
    )
    return ["cosine_similarity"]


@OP_REGISTRY.register()
def cumsum(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:cumsum' to NNEF using a scan fragment (Tract).

    - Implemented via `tract_core_scan` inside `fragment cumsum` (axis=0).
    - For arbitrary dim, transpose input to bring that axis to 0, apply
      fragment, then transpose back.
    """
    input_node, dim_node = node.inputs[:2]
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("cumsum need `TractNNEF` inference target")

    axis = pick_axis(input_node, dim_node.data)
    x = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

    # build zero init slice with shape [1, ...]
    first = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=x,
        attrs={
            "axes": [axis],
            "begin": [0],
            "end": [1],
            "stride": [1],
        },
        output_tensor_name_suffix="cumsum_first",
        pass_quantization_params=True,
    )
    init = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "sub",
        inputs=[first, first],
        output_tensor_name_suffix="cumsum_init",
        pass_quantization_params=True,
    )

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_cumsum",
        inputs=[x, init],
        attrs={
            "axis": axis,
        },
        pass_quantization_params=True,
    )

    # Ensure final op (possibly the last transpose) maps to node output
    # op_helper already wires node.outputs[0] to the last added op.
    return ["cumsum"]


@OP_REGISTRY.register()
def fmod(node, op_helper, **kwargs):
    """aten::fmod.

    equivalent:
        a - a.div(b, rounding_mode="trunc") * b
    """
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "fmod",
        inputs=(a, b),
    )
    return ["fmod", "trunc"]


def _parse_var_inputs(node, input_tensor_rank):
    """Parse `(input, dims, correction, keepdim)` for the var family.

    Covers the `aten::var` / `std` / `var_mean` / `std_mean` overloads.
    Returns `(input_node, axes, correction, keepdim)` with axes
    normalized to non-negative ints. Falls back to legacy `unbiased`
    booleans on the 2-arg overload (pre-correction torch).
    """
    if len(node.inputs) >= 4:
        input_node, dnode, cornode, kdnode = node.inputs[:4]
        keepdim = bool(kdnode.data)
    elif len(node.inputs) == 3:
        input_node, dnode, cornode = node.inputs[:3]
        keepdim = False
    elif len(node.inputs) == 2:  # legacy pytorch
        input_node, unbiased_node = node.inputs
        cor_val = 1 if unbiased_node.data else 0
        cornode = PythonConstant(
            name=f"{node.outputs[0].name}_corr", data=cor_val
        )
        dnode = PythonConstant(name=f"{node.outputs[0].name}_dims", data=None)
        keepdim = False
    else:
        raise T2NErrorNotImplemented(len(node.inputs))
    raw_axes = dnode.data
    if raw_axes is None or raw_axes == []:
        axes = list(range(input_tensor_rank))
    else:
        axes = [a if a >= 0 else input_tensor_rank + a for a in raw_axes]
    correction = int(cornode.data)
    return input_node, axes, correction, keepdim


def _static_n_along_axes(input_node, axes):
    """Static product of shape sizes along the given axes."""
    n = 1
    for axis in axes:
        dim = input_node.shape[axis]
        if not isinstance(dim, int):
            raise T2NErrorNotImplemented(
                f"var/std needs static shape on reduced axis {axis}; got {dim}"
            )
        n *= dim
    return n


def _make_intermediate_ntensor(g, name_to_tensor, name, shape, np_dtype):
    """NTensor with an explicit shape, registered in `name_to_tensor`.

    The shared `add_single_output_op_from_nnef_tensors` helper inherits
    `node.outputs[0].shape` for every intermediate it emits, which is
    the *post*-squeeze shape for the var/std family. The kept-dim
    intermediates have a different rank, and tract's auto rank-align
    relies on declared shapes -- if we declare a (2, 4) shape on a
    tensor that's actually (2, 1, 4), align inserts unsqueeze on the
    wrong axis and broadcast then misaligns the reduced axis.
    Building NTensors explicitly with the kept-dim shape avoids that
    trap.
    """
    tensor = NTensor(g, name, dtype=np_dtype, shape=tuple(shape))
    name_to_tensor[name] = tensor
    return tensor


def _emit_op_with_explicit_output(
    op_helper, *, op_type, inputs, output, attribs=None
):
    cast_and_add_nnef_operation(
        name_to_tensor=op_helper.name_to_tensor,
        graph=op_helper.g,
        type=op_type,
        name=f"{output.name}_op",
        inputs=tuple(inputs) if isinstance(inputs, list) else (inputs,),
        outputs=(output,),
        attribs=attribs or {},
    )
    return output


def _kept_dim_shape(input_node, axes):
    """Input shape with each reduced axis collapsed to size 1."""
    shape = list(input_node.shape)
    for ax in axes:
        shape[ax] = 1
    return shape


def _emit_var_or_std_with_optional_mean(
    node, op_helper, *, take_sqrt, also_emit_mean
):
    """Shared backbone for `var`, `std`, `var_mean`, `std_mean`.

    Pipeline (all kept-dim shapes, finalised by an explicit squeeze
    when `keepdim=False`):

        mean_kd = mean_reduce(x, axes)
        sq      = sqr(x - mean_kd)
        var_kd  = mean_reduce(sq, axes)                    # correction=0
                  | sum_reduce(sq, axes) * 1/(N-correction) # correction>0
        std_kd  = sqrt(var_kd)                              # for std/std_mean

    `var_mean` / `std_mean` reuse the single `mean_kd` for both outputs.
    """
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor
    input_node = node.inputs[0]
    rank = input_node.rank
    _, axes, correction, keepdim = _parse_var_inputs(node, rank)
    n = _static_n_along_axes(input_node, axes)
    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

    np_dtype = node.outputs[0].np_dtype
    base = node.outputs[0].export_name
    kd_shape = _kept_dim_shape(input_node, axes)

    mean_kd = _make_intermediate_ntensor(
        g, name_to_tensor, f"{base}_vs_mean_kd", kd_shape, np_dtype
    )
    _emit_op_with_explicit_output(
        op_helper,
        op_type="mean_reduce",
        inputs=[inp_ref],
        output=mean_kd,
        attribs={"axes": axes},
    )
    centered = _make_intermediate_ntensor(
        g,
        name_to_tensor,
        f"{base}_vs_centered",
        list(input_node.shape),
        np_dtype,
    )
    _emit_op_with_explicit_output(
        op_helper,
        op_type="sub",
        inputs=[inp_ref, mean_kd],
        output=centered,
    )
    sq = _make_intermediate_ntensor(
        g,
        name_to_tensor,
        f"{base}_vs_sq",
        list(input_node.shape),
        np_dtype,
    )
    _emit_op_with_explicit_output(
        op_helper, op_type="sqr", inputs=[centered], output=sq
    )

    if correction == 0:
        var_kd = _make_intermediate_ntensor(
            g, name_to_tensor, f"{base}_vs_var_kd", kd_shape, np_dtype
        )
        _emit_op_with_explicit_output(
            op_helper,
            op_type="mean_reduce",
            inputs=[sq],
            output=var_kd,
            attribs={"axes": axes},
        )
    else:
        denom = n - correction
        if denom <= 0:
            raise T2NErrorNotImplemented(
                f"var/std with correction={correction} ill-defined for N={n}"
            )
        ssq = _make_intermediate_ntensor(
            g, name_to_tensor, f"{base}_vs_ssq", kd_shape, np_dtype
        )
        _emit_op_with_explicit_output(
            op_helper,
            op_type="sum_reduce",
            inputs=[sq],
            output=ssq,
            attribs={"axes": axes},
        )
        inv_denom = PythonConstant(
            name=f"{base}_vs_inv_denom",
            data=torch.tensor(1.0 / float(denom), dtype=torch.float32),
        )
        inv_ref = op_helper.get_or_add_tensor_variable_in_nnef(inv_denom)
        var_kd = _make_intermediate_ntensor(
            g, name_to_tensor, f"{base}_vs_var_kd", kd_shape, np_dtype
        )
        _emit_op_with_explicit_output(
            op_helper,
            op_type="mul",
            inputs=[ssq, inv_ref],
            output=var_kd,
        )

    if take_sqrt:
        std_kd = _make_intermediate_ntensor(
            g, name_to_tensor, f"{base}_vs_std_kd", kd_shape, np_dtype
        )
        _emit_op_with_explicit_output(
            op_helper, op_type="sqrt", inputs=[var_kd], output=std_kd
        )
        primary_kd = std_kd
    else:
        primary_kd = var_kd

    _finalize_reduce_to_output(
        op_helper, node, primary_kd, axes, keepdim, output_idx=0
    )
    if also_emit_mean:
        _finalize_reduce_to_output(
            op_helper, node, mean_kd, axes, keepdim, output_idx=1
        )


def _finalize_reduce_to_output(
    op_helper, node, kept_dim_ref, axes, keepdim, output_idx
):
    """Squeeze (or reshape) `kept_dim_ref` into `node.outputs[output_idx]`.

    NNEF reductions keep the reduced axis as size-1; torch's keepdim=False
    drops it, so we emit an explicit squeeze. For keepdim=True the kept-dim
    shape already matches the torch output, so a no-op reshape suffices to
    materialize the final-named NTensor.
    """
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor
    onode = node.outputs[output_idx]
    out_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        onode, prevent_variable=True
    )
    op_type = "squeeze" if not keepdim else "reshape"
    attribs = {"axes": axes} if not keepdim else {"shape": list(onode.shape)}
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=op_type,
        name=f"{onode.export_name}_finalize",
        inputs=kept_dim_ref,
        outputs=out_ref,
        attribs=attribs,
    )


@OP_REGISTRY.register()
def var(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:var' to NNEF.

    Centered second moment along `dim` with arbitrary `correction`
    (denominator = N - correction). Lowered to mean_reduce + sub + sqr
    + (sum_reduce / mean_reduce) so any correction value works without
    relying on NNEF's `var` fragment (which always squeezes and so
    can't honor `keepdim=True`).
    """
    _emit_var_or_std_with_optional_mean(
        node, op_helper, take_sqrt=False, also_emit_mean=False
    )


@OP_REGISTRY.register()
def std(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:std' (sqrt of var) to NNEF."""
    _emit_var_or_std_with_optional_mean(
        node, op_helper, take_sqrt=True, also_emit_mean=False
    )


@OP_REGISTRY.register()
def var_mean(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:var_mean' (returns `(var, mean)`) to NNEF."""
    assert len(node.outputs) == 2
    _emit_var_or_std_with_optional_mean(
        node, op_helper, take_sqrt=False, also_emit_mean=True
    )


@OP_REGISTRY.register()
def std_mean(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:std_mean' (returns `(std, mean)`) to NNEF."""
    assert len(node.outputs) == 2
    _emit_var_or_std_with_optional_mean(
        node, op_helper, take_sqrt=True, also_emit_mean=True
    )


@OP_REGISTRY.register(["logical_xor"])
def logical_xor(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:logical_xor' to NNEF."""
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    op_helper.unary_output_op_without_attr(
        nnef_op_type="tract_core_xor", node=node
    )
    return ["tract_core"]


@OP_REGISTRY.register(["bitwise_xor"])
def bitwise_xor(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:bitwise_xor' to NNEF."""
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    op_helper.unary_output_op_without_attr(
        nnef_op_type="tract_core_bitxor", node=node
    )
    return ["tract_core"]


@OP_REGISTRY.register(["bitwise_and", "bitwise_cpu"])
def bitwise_and(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:bitwise_and', 'aten:bitwise_cpu' to NNEF."""
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    op_helper.unary_output_op_without_attr(
        nnef_op_type="tract_core_bitand", node=node
    )
    return ["tract_core"]


@OP_REGISTRY.register(["bitwise_not", "bitwise_not_cpu"])
def bitwise_not(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:bitwise_not', 'aten:bitwise_not_cpu' to NNEF.

    On bool inputs, PyTorch's `~` is semantically a logical not, so we emit
    the standard NNEF `not` op (keeps the graph portable and self-documenting,
    rather than relying on tract's bitnot happening to do the right thing on
    bool). For integer inputs, emit `tract_core_bitnot` for true bitwise
    inversion.
    """
    assert len(node.outputs) == 1
    input_node = node.inputs[0]
    if getattr(input_node, "dtype", None) == torch.bool:
        op_helper.unary_output_op_without_attr(nnef_op_type="not", node=node)
        return []
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    op_helper.unary_output_op_without_attr(
        nnef_op_type="tract_core_bitnot", node=node
    )
    return ["tract_core"]


@OP_REGISTRY.register(["bitwise_or"])
def bitwise_or(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:bitwise_or' to NNEF."""
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    op_helper.unary_output_op_without_attr(
        nnef_op_type="tract_core_bitor", node=node
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def addcmul(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:addcmul' to the `addcmul` fragment."""
    out_node, x_node, y_node, value_node = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "addcmul",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(out_node),
            op_helper.get_or_add_tensor_variable_in_nnef(x_node),
            op_helper.get_or_add_tensor_variable_in_nnef(y_node),
        ],
        attrs={"value": float(value_node.data)},
        force_consistent_inputs_shapes=False,
    )
    return ["addcmul"]


@OP_REGISTRY.register()
def lerp(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:lerp' to the `lerp` fragment."""
    start_node, end_node, weight_node = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "lerp",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(start_node),
            op_helper.get_or_add_tensor_variable_in_nnef(end_node),
            op_helper.get_or_add_tensor_variable_in_nnef(weight_node),
        ],
        force_consistent_inputs_shapes=False,
    )
    return ["lerp"]


@OP_REGISTRY.register()
def logit(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:logit' to the `logit` fragment."""
    x_node = node.inputs[0]
    eps_val = 0.0
    if len(node.inputs) > 1 and node.inputs[1].data is not None:
        eps_val = float(node.inputs[1].data)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "logit",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(x_node),
        attrs={"eps": eps_val},
    )
    return ["logit"]


@OP_REGISTRY.register()
def log_sigmoid(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:log_sigmoid' to the `log_sigmoid` fragment."""
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "log_sigmoid",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0]),
    )
    return ["log_sigmoid"]


@OP_REGISTRY.register()
def isfinite(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:isfinite' to the `isfinite` fragment."""
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "isfinite",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0]),
    )
    return ["isfinite"]


@OP_REGISTRY.register()
def hardshrink(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:hardshrink' to the `hardshrink` fragment."""
    x_node, lambd_node = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "hardshrink",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(x_node),
        attrs={"lambd": float(lambd_node.data)},
    )
    return ["hardshrink"]


@OP_REGISTRY.register()
def softshrink(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:softshrink' to the `softshrink` fragment."""
    x_node, lambd_node = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "softshrink",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(x_node),
        attrs={"lambd": float(lambd_node.data)},
    )
    return ["softshrink"]


@OP_REGISTRY.register()
def celu(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:celu' to the `celu` fragment."""
    x_node, alpha_node = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "celu",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(x_node),
        attrs={"alpha": float(alpha_node.data)},
    )
    return ["celu"]


@OP_REGISTRY.register()
def logsumexp(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:logsumexp' to the `logsumexp` fragment.

    PyTorch signature: ``logsumexp(input, dim, keepdim=False)``.
    The fragment always reduces the named axis (no keepdim); when the
    user asks for keepdim=True we follow up with an `unsqueeze` on
    the same axis to reinstate it.
    """
    input_node, dim_node, keepdim_node = node.inputs[:3]
    dim = dim_node.data
    if isinstance(dim, list):
        if len(dim) != 1:
            raise T2NErrorNotImplemented(
                f"logsumexp over multiple axes not yet supported: {dim}"
            )
        dim = dim[0]
    keepdim = bool(keepdim_node.data)
    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if keepdim:
        reduced = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "logsumexp",
            inputs=inp_ref,
            attrs={"axis": dim},
            output_tensor_name_suffix="_lse",
        )
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=reduced,
            attrs={"axes": [dim]},
        )
    else:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "logsumexp",
            inputs=inp_ref,
            attrs={"axis": dim},
        )
    return ["logsumexp"]
