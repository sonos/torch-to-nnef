import logging
import math

import numpy as np
import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import (
    TORCH_DTYPE_TO_TRACT_STR,
    TORCH_TO_NUMPY_DTYPE,
    dtype_is_whole_number,
)
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
        # When the traced output is integer we deliberately keep the
        # division (and its rounding fragment) in float and cast the
        # result back to int below. Letting the generic implicit-cast
        # realign the f32 operands to the int output dtype would run the
        # rounding fragment (e.g. `trunc`'s `select(x < 0.0, ...)`) on
        # integers, which tract cannot type-resolve.
        maybe_cast_align_tract=io_casting_with_dtype is None,
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


@OP_REGISTRY.register(torch_op_ids=["floor_divide", "floordiv"])
def floor_divide(node, op_helper, inference_target, torch_graph, **kwargs):
    """Map PyTorch: 'aten::floor_divide' / 'aten::floordiv' to NNEF.

    JIT records `aten::floordiv` for Python `//`; upstream
    `normalize_ops.cpp` does not bridge it to `floor_divide`, so we
    alias it on our side.
    """
    input_node, divisor_node = node.inputs
    if input_node.data is not None and divisor_node.data is not None:
        # both operands concrete -> fold (a genuinely-dynamic operand has
        # data=None here, so this never bakes a symbolic dim, even under
        # dynamic_axes). Preserve int-ness: shape arithmetic (e.g.
        # head_dim // 2 feeding a slice bound) must stay integer, else a float
        # bound clashes with tract's symbolic (TDim) axis size.
        def _scalar(n):
            return n.data.tolist() if isinstance(n, TensorVariable) else n.data

        idata, ddata = _scalar(input_node), _scalar(divisor_node)
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
        inputs[idx] = op_helper.add_cast_nnef_tensor(
            inp,
            cast_to=np.float32,
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
    tensor on both sides. `square` preserves dtype, so no int->float
    promotion is needed.
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
def pairwise_distance(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:pairwise_distance' to NNEF via a fragment.

    `pairwise_distance(a, b, p, eps, keepdim)` computes
    `(sum(|a - b + eps|^p, axis=-1))^(1/p)`. Torch keeps the reduced
    last axis only when `keepdim=True`; the fragment squeezes it
    unconditionally and we re-`unsqueeze` after the call when needed.

    Pure NNEF stdlib (sub / abs / pow / sum_reduce / squeeze).
    """
    a_node, b_node, p_node, eps_node, keepdim_node = node.inputs
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    axis = a_node.rank - 1
    p_val = float(p_node.data)
    eps_val = float(eps_node.data)
    keepdim = bool(keepdim_node.data)
    if keepdim:
        squeezed = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "pairwise_distance",
            inputs=[a_ref, b_ref],
            attrs={"axis": axis, "p": p_val, "eps": eps_val},
            force_consistent_inputs_shapes=False,
            output_tensor_name_suffix="pwd_squeezed",
        )
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=squeezed,
            attrs={"axes": [axis]},
        )
    else:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "pairwise_distance",
            inputs=[a_ref, b_ref],
            attrs={"axis": axis, "p": p_val, "eps": eps_val},
            force_consistent_inputs_shapes=False,
        )
    return ["pairwise_distance"]


def _tensordot_einsum_expr(ra, rb, dims_a, dims_b):
    """Build the einsum expression for `tensordot(a, b, dims_a, dims_b)`.

    Each axis of `a` gets a unique label; each contracted axis of `b`
    reuses its paired `a` label (forcing the contraction); each
    non-contracted axis of `b` gets a fresh label. Output spec is
    `non-contracted a labels` followed by `non-contracted b labels`,
    matching torch's tensordot ordering.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    if ra + rb - len(dims_a) > len(letters):
        raise T2NErrorNotImplemented(
            f"tensordot rank budget exceeded (ra={ra}, rb={rb}, "
            f"contracted={len(dims_a)}, max={len(letters)})"
        )
    a_labels = list(letters[:ra])
    pos = ra
    a_to_b = {dims_a[k]: a_labels[dims_a[k]] for k in range(len(dims_a))}
    b_labels = []
    for j in range(rb):
        if j in dims_b:
            idx = dims_b.index(j)
            b_labels.append(a_to_b[dims_a[idx]])
        else:
            b_labels.append(letters[pos])
            pos += 1
    a_out = [a_labels[i] for i in range(ra) if i not in dims_a]
    b_out = [b_labels[j] for j in range(rb) if j not in dims_b]
    return f"{''.join(a_labels)},{''.join(b_labels)}->{''.join(a_out + b_out)}"


@OP_REGISTRY.register()
def tensordot(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:tensordot' to NNEF via `tract_core_einsum`.

    `tensordot(a, b, dims_a, dims_b)` contracts the paired axes (same
    size on each side) and produces an output of rank
    `a.rank + b.rank - 2 * len(dims_a)`. The einsum expression is
    built so each contracted axis-pair shares a label.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "tensordot requires `tract_core_einsum` (TractNNEF target)"
        )
    a_node, b_node, dims_a_node, dims_b_node = node.inputs
    ra = a_node.rank
    rb = b_node.rank
    dims_a_raw = dims_a_node.data
    dims_b_raw = dims_b_node.data
    if hasattr(dims_a_raw, "tolist"):
        dims_a_raw = dims_a_raw.tolist()
    if hasattr(dims_b_raw, "tolist"):
        dims_b_raw = dims_b_raw.tolist()
    dims_a = [pick_axis(a_node, int(d)) for d in dims_a_raw]
    dims_b = [pick_axis(b_node, int(d)) for d in dims_b_raw]
    expr = _tensordot_einsum_expr(ra, rb, dims_a, dims_b)
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_einsum",
        inputs=[a_ref, b_ref],
        ensure_tuple=False,
        force_consistent_inputs_shapes=False,
        attrs={
            "expr": expr,
            "acc": "f32",
            "output": "",
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def frexp(node, op_helper, inference_target, **kwargs):
    """Map `aten::frexp(input) -> (mantissa, exponent)` to NNEF.

    The exponent output is `int32`; the fragment uses `tract_core_cast`
    so this handler is tract-only.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "frexp requires `tract_core_cast` (TractNNEF target)"
        )
    input_node = node.inputs[0]
    assert len(node.outputs) == 2
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    op_helper.add_multi_output_op_from_nnef_tensors(node, "frexp", inputs=[inp])
    return ["frexp"]


@OP_REGISTRY.register()
def dist(node, op_helper, **kwargs):
    """Map PyTorch: `aten::dist(input, other, p)` to NNEF via fragment.

    Scalar `(sum_{all axes} |a - b|^p)^(1/p)` between two broadcastable
    tensors. p must be finite and > 0; the inf / -inf / 0 norms would
    need `max_reduce` / `min_reduce` / `count_nonzero` branches (raise
    `T2NErrorNotImplemented` for those).
    """
    a_node, b_node, p_node = node.inputs[:3]
    p_val = float(p_node.data) if p_node.data is not None else 2.0
    if p_val <= 0 or not math.isfinite(p_val):
        raise T2NErrorNotImplemented(
            f"dist with p={p_val} not supported (require finite p > 0)"
        )
    rank = max(a_node.rank, b_node.rank)
    if rank == 0:
        raise T2NErrorNotImplemented("dist: scalar inputs not supported")
    axes = list(range(rank))
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "dist",
        inputs=[a_ref, b_ref],
        attrs={"axes": axes, "p": p_val},
        force_consistent_inputs_shapes=False,
    )
    return ["dist"]


@OP_REGISTRY.register()
def cdist(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:cdist' to NNEF via a fragment.

    `cdist(a, b, p)` computes the pairwise distance matrix between
    rows of `a` (shape `(..., M, D)`) and `b` (shape `(..., N, D)`):
    `out[..., i, j] = (sum(|a[..., i, :] - b[..., j, :]|^p))^(1/p)`.

    The fragment broadcasts via `unsqueeze` (one new axis on each
    input) and reduces along the trailing feature axis. Pure stdlib.
    """
    a_node, b_node, p_node = node.inputs[:3]
    p_val = float(p_node.data) if p_node.data is not None else 2.0
    if p_val <= 0:
        raise T2NErrorNotImplemented(
            f"cdist with p={p_val} not supported (require p > 0; "
            "p=inf would need max_reduce, separate code path)"
        )
    if a_node.rank < 2 or b_node.rank < 2 or a_node.rank != b_node.rank:
        raise T2NErrorNotImplemented(
            f"cdist needs equal-rank rank>=2 inputs; "
            f"got a.rank={a_node.rank}, b.rank={b_node.rank}"
        )
    rank = a_node.rank
    # a (..., M, D) -> a_exp (..., M, 1, D)  via unsqueeze at index rank-1
    a_axis = rank - 1
    # b (..., N, D) -> b_exp (..., 1, N, D)  via unsqueeze at index rank-2
    b_axis = rank - 2
    # reduce the trailing D axis on the rank-(rank+1) broadcast tensor
    reduce_axis = rank
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "cdist",
        inputs=[a_ref, b_ref],
        attrs={
            "a_axis": a_axis,
            "b_axis": b_axis,
            "reduce_axis": reduce_axis,
            "p": p_val,
        },
        force_consistent_inputs_shapes=False,
    )
    return ["cdist"]


@OP_REGISTRY.register()
def pdist(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: `aten::pdist(self, p)` to NNEF.

    `torch.pdist(x, p)` returns a 1-D tensor of length `N*(N-1)/2`
    holding the pairwise p-norm distances between rows of the
    rank-2 input `x` (shape `(N, D)`). We reuse the existing `cdist`
    fragment to build the `(N, N)` distance matrix against `x` itself,
    flatten it, and gather the strict upper-triangle entries
    `i*N + j` for `i < j`. Static `N` only (the upper-triangle index
    constant is baked at export time).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "pdist requires `tract_core_gather` (TractNNEF target)"
        )
    input_node, p_node = node.inputs[:2]
    p_val = float(p_node.data) if p_node.data is not None else 2.0
    if p_val <= 0:
        raise T2NErrorNotImplemented(f"pdist with p={p_val} not supported")
    if input_node.rank != 2:
        raise T2NErrorNotImplemented(
            f"pdist expects rank-2 input; got rank={input_node.rank}"
        )
    n = input_node.shape[0]
    if not isinstance(n, int):
        raise T2NErrorNotImplemented(
            "pdist on dynamic N not yet supported (upper-tri index "
            "constant is baked at export time)"
        )

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    dist_matrix = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "cdist",
        inputs=[inp_ref, inp_ref],
        attrs={"a_axis": 1, "b_axis": 0, "reduce_axis": 2, "p": p_val},
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="_pdist_matrix",
    )
    flat = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=[dist_matrix],
        attrs={"shape": [n * n]},
        output_tensor_name_suffix="_pdist_flat",
    )
    upper_indices = [i * n + j for i in range(n) for j in range(i + 1, n)]
    idx_const = PythonConstant(
        name=f"{node.outputs[0].export_name}_pdist_idx",
        data=torch.tensor(upper_indices, dtype=torch.int64),
    )
    idx_ref = op_helper.get_or_add_tensor_variable_in_nnef(idx_const)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather",
        inputs=[flat, idx_ref],
        attrs={"axis": 0},
        force_consistent_inputs_shapes=False,
    )
    return ["cdist", "tract_core"]


@OP_REGISTRY.register(torch_op_ids=["cross", "linalg_cross"])
def cross(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:cross' to NNEF via a fragment.

    `cross(a, b, dim)` is the 3-D vector cross product along `dim`.
    The fragment slices each input along `dim` into its three
    components and computes the standard `(a1*b2 - a2*b1, a2*b0 -
    a0*b2, a0*b1 - a1*b0)` triplet. Pure stdlib.
    """
    a_node, b_node, dim_node = node.inputs
    axis = pick_axis(a_node, dim_node.data)
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "cross",
        inputs=[a_ref, b_ref],
        attrs={"axis": axis},
        force_consistent_inputs_shapes=False,
    )
    return ["cross"]


@OP_REGISTRY.register()
def special_entr(node, op_helper, **kwargs):
    """aten::special_entr -> `-x * log(x)` (0 at x=0).

    See `special_entr.nnef` for the eps-clamped formulation that keeps
    `log(0)` from poisoning the discarded `select` branch.
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "special_entr", inputs=inp
    )
    return ["special_entr"]


@OP_REGISTRY.register()
def special_xlog1py(node, op_helper, **kwargs):
    """aten::special_xlog1py -> `x * log(1 + y)` (0 at x=0).

    See `special_xlog1py.nnef` for the eps-clamped log argument.
    """
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "special_xlog1py",
        inputs=[a, b],
        force_consistent_inputs_shapes=False,
    )
    return ["special_xlog1py"]


@OP_REGISTRY.register(torch_op_ids=["trapezoid", "trapz"])
def trapezoid(node, op_helper, **kwargs):
    """Map `aten::trapezoid(y, dx_or_x, dim)` (alias `trapz`) to NNEF.

    Uniform-`dx` case only: the second arg must be a scalar float
    constant (the `dx` overload). The tensor-`x` overload is rejected
    for now (would need a `(x[1:] - x[:-1])` multiply that broadcasts
    against `y[1:] + y[:-1]`).
    """
    y_node = node.inputs[0]
    dx_node = node.inputs[1] if len(node.inputs) >= 2 else None
    dim_node = node.inputs[2] if len(node.inputs) >= 3 else None

    if (
        dx_node is None
        or not isinstance(dx_node, PythonConstant)
        or not isinstance(dx_node.data, (int, float))
    ):
        raise T2NErrorNotImplemented(
            "trapezoid: only the uniform-dx overload is supported "
            "(tensor x would need extra broadcasting)"
        )
    dx = float(dx_node.data)
    raw_dim = (
        dim_node.data
        if dim_node is not None and dim_node.data is not None
        else -1
    )
    axis = pick_axis(y_node, raw_dim)
    size = y_node.shape[axis]
    if not isinstance(size, int) or size < 2:
        raise T2NErrorNotImplemented(
            f"trapezoid: axis {axis} needs static size >= 2 (got {size})"
        )
    y = op_helper.get_or_add_tensor_variable_in_nnef(y_node)
    left = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=y,
        attrs={
            "axes": [axis],
            "begin": [0],
            "end": [size - 1],
            "stride": [1],
        },
        output_tensor_name_suffix="_trapz_left",
    )
    right = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=y,
        attrs={
            "axes": [axis],
            "begin": [1],
            "end": [size],
            "stride": [1],
        },
        output_tensor_name_suffix="_trapz_right",
    )
    summed = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "add",
        inputs=[left, right],
        output_tensor_name_suffix="_trapz_sum",
    )
    scaled = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=summed,
        attrs={"y": 0.5 * dx},
        output_tensor_name_suffix="_trapz_scaled",
    )
    reduced = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "sum_reduce",
        inputs=scaled,
        attrs={"axes": [axis]},
        output_tensor_name_suffix="_trapz_reduce",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "squeeze", inputs=reduced, attrs={"axes": [axis]}
    )


@OP_REGISTRY.register()
def diff(node, op_helper, **kwargs):
    """Map `aten::diff(input, n, dim, prepend?, append?)` to NNEF.

    n-th order finite differences along `dim`: each step replaces
    `x` with `x[1:] - x[:-1]` along `dim`. `prepend` / `append` are
    not supported (raise `T2NErrorNotImplemented`).
    """
    input_node = node.inputs[0]
    n_node = node.inputs[1] if len(node.inputs) >= 2 else None
    dim_node = node.inputs[2] if len(node.inputs) >= 3 else None
    prepend_node = node.inputs[3] if len(node.inputs) >= 4 else None
    append_node = node.inputs[4] if len(node.inputs) >= 5 else None

    has_n = n_node is not None and n_node.data is not None
    n = int(n_node.data) if has_n else 1
    if n < 1:
        raise T2NErrorNotImplemented(f"diff: n must be >= 1, got {n}")
    raw_dim = (
        dim_node.data
        if dim_node is not None and dim_node.data is not None
        else -1
    )
    axis = pick_axis(input_node, raw_dim)
    for opt_node, label in (
        (prepend_node, "prepend"),
        (append_node, "append"),
    ):
        if opt_node is not None and getattr(opt_node, "data", None) is not None:
            raise T2NErrorNotImplemented(f"diff: {label} != None not supported")

    size = input_node.shape[axis]
    if not isinstance(size, int):
        raise T2NErrorNotImplemented(
            f"diff: dynamic size on axis {axis} not supported"
        )
    if n >= size:
        raise T2NErrorNotImplemented(
            f"diff: n={n} >= axis-{axis} size {size}; "
            "would produce an empty tensor"
        )

    acc = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    cur_size = size
    for step in range(n):
        is_final = step == n - 1
        suffix_left = "" if is_final else f"_diff_step_{step}_left"
        left = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "slice",
            inputs=acc,
            attrs={
                "axes": [axis],
                "begin": [0],
                "end": [cur_size - 1],
                "stride": [1],
            },
            output_tensor_name_suffix=f"_diff_step_{step}_left",
        )
        right = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "slice",
            inputs=acc,
            attrs={
                "axes": [axis],
                "begin": [1],
                "end": [cur_size],
                "stride": [1],
            },
            output_tensor_name_suffix=f"_diff_step_{step}_right",
        )
        del suffix_left
        acc = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "sub",
            inputs=[right, left],
            output_tensor_name_suffix="" if is_final else f"_diff_step_{step}",
        )
        cur_size -= 1


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
def cumprod(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:cumprod' to NNEF using a scan fragment.

    Mirror of `cumsum` with a `mul` scan body and an init of `1` (built
    pointwise via `mul(first, 0) + 1` to keep init shape-matching).
    """
    input_node, dim_node = node.inputs[:2]
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "cumprod need `TractNNEF` inference target"
        )

    axis = pick_axis(input_node, dim_node.data)
    x = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

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
        output_tensor_name_suffix="cumprod_first",
        pass_quantization_params=True,
    )
    # init = 1, shape-matched to `first`. Build via `0 * first + 1`
    # where the `1` is a scalar PythonConstant broadcast into `first`'s
    # shape by `add`.
    zeroed = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "sub",
        inputs=[first, first],
        output_tensor_name_suffix="cumprod_zero",
        pass_quantization_params=True,
    )
    one_const = PythonConstant(
        name=f"{node.outputs[0].export_name}_cumprod_one",
        data=torch.tensor(1.0, dtype=input_node.dtype or torch.float32),
    )
    one_ref = op_helper.get_or_add_tensor_variable_in_nnef(one_const)
    init = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "add",
        inputs=[zeroed, one_ref],
        output_tensor_name_suffix="cumprod_init",
        pass_quantization_params=True,
    )

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_cumprod",
        inputs=[x, init],
        attrs={"axis": axis},
        pass_quantization_params=True,
    )
    return ["cumprod"]


def _emit_cum_minmax(node, op_helper, inference_target, op_label: str):
    """Shared body for `cummax` / `cummin`.

    PyTorch returns `(values, indices)` so the IR node has two outputs.
    We run two single-output scans (mirror of `cumsum`):

    * `tract_cum{max,min}_values` returns running max / min values.
    * `tract_cum{max,min}_indices` keeps a `(running_val, running_idx)`
      pair in its state but emits only the index, so we don't need to
      teach `tract_core_scan` about multi-output deserialisation.

    `gt` / `lt` (strict) preserve the first-occurrence tie-break that
    PyTorch uses.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            f"{op_label} need `TractNNEF` inference target"
        )
    input_node, dim_node = node.inputs[:2]
    axis = pick_axis(input_node, dim_node.data)
    if not isinstance(input_node.shape[axis], int):
        raise T2NErrorNotImplemented(
            f"{op_label} on dynamic axis {axis} not yet supported"
        )
    k = input_node.shape[axis]
    input_dtype = input_node.dtype or torch.float32
    finfo = torch.finfo(input_dtype)
    # `init_val` is `-inf` for cummax (so the first step's input always
    # wins the strict-gt compare) and `+inf` for cummin.
    sentinel = finfo.min if op_label == "cummax" else finfo.max
    base = node.outputs[0].export_name

    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor

    x = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

    # `node` has two outputs (values, indices), so the
    # `add_single_output_op_from_nnef_tensors` helper isn't usable here
    # (it asserts single-output). Build intermediates manually.
    slice_shape = list(input_node.shape)
    slice_shape[axis] = 1
    np_dtype = TORCH_TO_NUMPY_DTYPE[input_dtype]

    def _emit_intermediate(op_type, inputs_, suffix, attrs=None, shape=None):
        out_name = f"{base}_{suffix}"
        out_t = NTensor(g, out_name, dtype=np_dtype, shape=shape or slice_shape)
        name_to_tensor[out_name] = out_t
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type=op_type,
            name=f"{out_name}_op",
            inputs=tuple(inputs_),
            outputs=(out_t,),
            attribs=attrs or {},
        )
        return out_t

    first = _emit_intermediate(
        "slice",
        [x],
        f"{op_label}_first",
        attrs={
            "axes": [axis],
            "begin": [0],
            "end": [1],
            "stride": [1],
        },
    )
    zeroed = _emit_intermediate("sub", [first, first], f"{op_label}_zero")
    sentinel_const = PythonConstant(
        name=f"{base}_{op_label}_sentinel",
        data=torch.tensor(float(sentinel), dtype=input_dtype),
    )
    sentinel_ref = op_helper.get_or_add_tensor_variable_in_nnef(sentinel_const)
    init_val = _emit_intermediate(
        "add", [zeroed, sentinel_ref], f"{op_label}_init_val"
    )

    # 1st scan: running values into `node.outputs[0]`.
    values_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        node.outputs[0], prevent_variable=True
    )
    cast_and_add_nnef_operation(
        name_to_tensor=op_helper.name_to_tensor,
        graph=op_helper.g,
        type=f"tract_{op_label}_values",
        name=f"{node.outputs[0].export_name}_{op_label}_values",
        inputs=(x, init_val),
        outputs=(values_ref,),
        attribs={"axis": axis},
    )

    # `idx_full`: tensor of shape `input_node.shape` with `arange(K)`
    # broadcast along `axis`. Each scan step picks the per-position
    # index without needing a separate step counter.
    idx_shape = [1] * input_node.rank
    idx_shape[axis] = k
    idx_arange = torch.arange(k, dtype=torch.int64).reshape(idx_shape)
    idx_data = idx_arange.expand(list(input_node.shape)).contiguous()
    idx_const = PythonConstant(
        name=f"{node.outputs[1].export_name}_{op_label}_idx_full",
        data=idx_data,
    )
    idx_full_ref = op_helper.get_or_add_tensor_variable_in_nnef(idx_const)

    # `init_idx`: zeros (any int constant works; the first iteration
    # always overwrites it).
    init_idx_const = PythonConstant(
        name=f"{node.outputs[1].export_name}_{op_label}_init_idx",
        data=torch.zeros(
            [s if i != axis else 1 for i, s in enumerate(input_node.shape)],
            dtype=torch.int64,
        ),
    )
    init_idx_ref = op_helper.get_or_add_tensor_variable_in_nnef(init_idx_const)

    # 2nd scan: running indices into `node.outputs[1]`.
    indices_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        node.outputs[1], prevent_variable=True
    )
    cast_and_add_nnef_operation(
        name_to_tensor=op_helper.name_to_tensor,
        graph=op_helper.g,
        type=f"tract_{op_label}_indices",
        name=f"{node.outputs[1].export_name}_{op_label}_indices",
        inputs=(x, idx_full_ref, init_val, init_idx_ref),
        outputs=(indices_ref,),
        attribs={"axis": axis},
    )
    return [op_label]


@OP_REGISTRY.register()
def cummax(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: `aten::cummax(self, dim) -> (values, indices)`."""
    return _emit_cum_minmax(node, op_helper, inference_target, "cummax")


@OP_REGISTRY.register()
def cummin(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: `aten::cummin(self, dim) -> (values, indices)`."""
    return _emit_cum_minmax(node, op_helper, inference_target, "cummin")


@OP_REGISTRY.register()
def exp2(node, op_helper, **kwargs):
    """aten::exp2 -> `exp2` fragment (`exp(x * ln 2)`)."""
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "exp2", inputs=inp)
    return ["exp2"]


@OP_REGISTRY.register()
def hypot(node, op_helper, **kwargs):
    """aten::hypot -> `hypot` fragment (`sqrt(a^2 + b^2)`)."""
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "hypot", inputs=[a, b]
    )
    return ["hypot"]


@OP_REGISTRY.register()
def xlogy(node, op_helper, **kwargs):
    """aten::xlogy -> `xlogy` fragment (`x * log(y)` with x==0 -> 0)."""
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    y = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "xlogy", inputs=[x, y]
    )
    return ["xlogy"]


@OP_REGISTRY.register()
def heaviside(node, op_helper, **kwargs):
    """aten::heaviside -> `heaviside` fragment (step with tie-breaker)."""
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    v = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "heaviside", inputs=[x, v]
    )
    return ["heaviside"]


@OP_REGISTRY.register()
def fmax(node, op_helper, **kwargs):
    """aten::fmax -> NaN-skipping max."""
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "fmax", inputs=[a, b]
    )
    return ["fmax"]


@OP_REGISTRY.register()
def fmin(node, op_helper, **kwargs):
    """aten::fmin -> NaN-skipping min."""
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "fmin", inputs=[a, b]
    )
    return ["fmin"]


@OP_REGISTRY.register()
def logaddexp(node, op_helper, **kwargs):
    """aten::logaddexp -> numerically-stable `log(exp a + exp b)`."""
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "logaddexp", inputs=[a, b]
    )
    return ["logaddexp"]


@OP_REGISTRY.register()
def logaddexp2(node, op_helper, **kwargs):
    """aten::logaddexp2 -> base-2 variant."""
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "logaddexp2", inputs=[a, b]
    )
    return ["logaddexp2"]


@OP_REGISTRY.register()
def copysign(node, op_helper, **kwargs):
    """aten::copysign -> `|a| * sign(b)`."""
    a = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    b = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "copysign", inputs=[a, b]
    )
    return ["copysign"]


@OP_REGISTRY.register()
def sinc(node, op_helper, **kwargs):
    """aten::sinc -> normalised `sin(pi x)/(pi x)` with 0 -> 1."""
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "sinc", inputs=inp)
    return ["sinc"]


@OP_REGISTRY.register()
def isclose(node, op_helper, **kwargs):
    """aten::isclose -> `|a - b| <= atol + rtol * |b|`.

    Reads optional `rtol` / `atol` from the trace (defaults match
    torch's 1e-5 / 1e-8 via the fragment defaults). `equal_nan=True`
    is not yet handled -- rare in real traces; raises if set.
    """
    input_a, input_b, *opt = node.inputs
    attrs = {}
    # Trace order: `aten::isclose(self, other, rtol, atol, equal_nan)`.
    if len(opt) >= 1 and opt[0].data is not None:
        attrs["rtol"] = float(opt[0].data)
    if len(opt) >= 2 and opt[1].data is not None:
        attrs["atol"] = float(opt[1].data)
    if len(opt) >= 3 and opt[2].data:
        raise T2NErrorNotImplemented(
            "isclose with `equal_nan=True` not yet supported"
        )
    a = op_helper.get_or_add_tensor_variable_in_nnef(input_a)
    b = op_helper.get_or_add_tensor_variable_in_nnef(input_b)
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "isclose", inputs=[a, b], attrs=attrs
    )
    return ["isclose"]


@OP_REGISTRY.register()
def frac(node, op_helper, **kwargs):
    """aten::frac -> `x - trunc(x)` (sign-of-x fractional part)."""
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "frac", inputs=inp)
    return ["trunc", "frac"]


@OP_REGISTRY.register()
def signbit(node, op_helper, **kwargs):
    """aten::signbit -> `x < 0` (bool).

    NNEF can't see the IEEE-754 sign bit, so `signbit(-0.0)` returns
    False here (vs PyTorch's True); same caveat as `copysign` /
    `atan2`.
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "signbit", inputs=inp
    )
    return ["signbit"]


@OP_REGISTRY.register()
def erfc(node, op_helper, **kwargs):
    """aten::erfc -> `1 - erf(x)`."""
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "erfc", inputs=inp)
    return ["erf", "erfc"]


@OP_REGISTRY.register()
def tanhshrink(node, op_helper, **kwargs):
    """aten::tanhshrink -> `x - tanh(x)`."""
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "tanhshrink", inputs=inp
    )
    return ["tanhshrink"]


@OP_REGISTRY.register()
def ldexp(node, op_helper, **kwargs):
    """aten::ldexp -> `x * 2^exp`."""
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    e = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "ldexp", inputs=[x, e]
    )
    return ["exp2", "ldexp"]


@OP_REGISTRY.register()
def logcumsumexp(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: `aten::logcumsumexp(self, dim)`.

    Numerically-stable scan with two state vars `(running_max,
    running_sum_shifted)`. Init: `finfo.min` and `0` so the first
    step naturally produces `out[0] = input[0]`.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "logcumsumexp need `TractNNEF` inference target"
        )
    input_node, dim_node = node.inputs[:2]
    axis = pick_axis(input_node, dim_node.data)
    input_dtype = input_node.dtype or torch.float32
    finfo = torch.finfo(input_dtype)
    base = node.outputs[0].export_name

    x = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
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
        output_tensor_name_suffix="_lcsex_first",
        pass_quantization_params=True,
    )
    init_sum = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "sub",
        inputs=[first, first],
        output_tensor_name_suffix="_lcsex_init_sum",
        pass_quantization_params=True,
    )
    neg_inf_const = PythonConstant(
        name=f"{base}_lcsex_neg_inf",
        data=torch.tensor(float(finfo.min), dtype=input_dtype),
    )
    neg_inf_ref = op_helper.get_or_add_tensor_variable_in_nnef(neg_inf_const)
    init_max = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "add",
        inputs=[init_sum, neg_inf_ref],
        output_tensor_name_suffix="_lcsex_init_max",
        pass_quantization_params=True,
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_logcumsumexp",
        inputs=[x, init_max, init_sum],
        attrs={"axis": axis},
    )
    return ["logcumsumexp"]


@OP_REGISTRY.register(torch_op_ids=["i0", "special_i0"])
def i0(node, op_helper, **kwargs):
    """aten::i0 / aten::special_i0 -> Bessel `I_0(x)`.

    Abramowitz & Stegun polynomial approximation; two branches at
    `|x| = 3.75`.
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "i0", inputs=inp)
    return ["i0"]


@OP_REGISTRY.register(torch_op_ids=["special_i0e"])
def special_i0e(node, op_helper, **kwargs):
    """aten::special_i0e -> `exp(-|x|) * I_0(x)`.

    Same Abramowitz & Stegun polynomial branches as `i0` but the
    large-x branch drops `exp(|x|)` so the result stays finite for
    arbitrarily large `|x|`.
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "i0e", inputs=inp)
    return ["i0e"]


@OP_REGISTRY.register()
def lgamma(node, op_helper, **kwargs):
    """aten::lgamma -> log-Gamma via Lanczos.

    Numerical Recipes Lanczos approximation (g = 5, N = 6). Valid for
    `x > 0.5`; smaller arguments need a reflection branch (left as a
    follow-up).
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "lgamma", inputs=inp)
    return ["lgamma"]


@OP_REGISTRY.register(torch_op_ids=["digamma", "special_digamma"])
def digamma(node, op_helper, **kwargs):
    """aten::digamma -> `psi(x) = (d/dx) log(Gamma(x))`.

    Asymptotic series after shifting `x` up by 6 via the recurrence
    `psi(x+1) = psi(x) + 1/x`. Valid for `x > 0`; negative inputs would
    need the reflection formula (left as a follow-up).
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "digamma", inputs=inp
    )
    return ["digamma"]


@OP_REGISTRY.register(torch_op_ids=["i1", "special_i1"])
def i1(node, op_helper, **kwargs):
    """aten::i1 / aten::special_i1 -> Bessel `I_1(x)`.

    Abramowitz & Stegun 9.8.3 / 9.8.4 polynomial branches; same
    structure as `i0` with the sign carried explicitly so the function
    remains odd.
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "i1", inputs=inp)
    return ["i1"]


@OP_REGISTRY.register(torch_op_ids=["special_i1e"])
def special_i1e(node, op_helper, **kwargs):
    """aten::special_i1e -> `exp(-|x|) * I_1(x)`.

    Same polynomial branches as `i1`; the large-|x| branch drops
    `exp(|x|)` so the result stays finite for arbitrarily large `|x|`.
    """
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(node, "i1e", inputs=inp)
    return ["i1e"]


@OP_REGISTRY.register()
def mvlgamma(node, op_helper, **kwargs):
    """aten::mvlgamma -> multivariate log-Gamma.

    `mvlgamma(x, p) = (p*(p-1)/4) * log(pi)
                    + sum_{i=1..p} lgamma(x + (1-i)/2)`.

    `p` is a static int from the trace; we unroll the sum into `p`
    `lgamma` fragment calls plus a single constant offset. Inherits
    `lgamma`'s domain restriction (each shifted argument must stay
    `> 0.5`).
    """
    input_node, p_node = node.inputs
    if not (
        isinstance(p_node, PythonConstant) and isinstance(p_node.data, int)
    ):
        raise T2NErrorNotImplemented("mvlgamma: dynamic p not supported")
    p = int(p_node.data)
    if p < 1:
        raise T2NErrorNotImplemented(f"mvlgamma: p must be >= 1, got {p}")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    terms = []
    for i in range(1, p + 1):
        offset = -0.5 * (i - 1)
        if offset == 0.0:
            shifted = inp
        else:
            shifted = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "add",
                inputs=inp,
                attrs={"y": offset},
                output_tensor_name_suffix=f"_mvlg_shift_{i}",
            )
        term = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "lgamma",
            inputs=shifted,
            output_tensor_name_suffix=f"_mvlg_term_{i}",
        )
        terms.append(term)
    summed = terms[0]
    for idx, t in enumerate(terms[1:], start=2):
        is_final = idx == len(terms)
        summed = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "add",
            inputs=[summed, t],
            output_tensor_name_suffix=""
            if (is_final and p > 1)
            else f"_mvlg_acc_{idx}",
        )
    const = (p * (p - 1) / 4.0) * math.log(math.pi)
    if const != 0.0 or p == 1:
        # `p == 1` collapses the sum to a single lgamma call; we still
        # need to land the result in node.outputs[0] (the inner term
        # currently sits in an intermediate without a final-name slot).
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "add",
            inputs=summed,
            attrs={"y": const},
        )
    return ["lgamma"]


@OP_REGISTRY.register()
def addcdiv(node, op_helper, **kwargs):
    """aten::addcdiv -> `self + value * (t1 / t2)`."""
    input_node, a_node, b_node, *opt = node.inputs
    attrs = {}
    if opt and opt[0].data is not None:
        attrs["value"] = float(opt[0].data)
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    a = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "addcdiv", inputs=[inp, a, b], attrs=attrs
    )
    return ["addcdiv"]


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


def _emit_bitwise_shift(node, op_helper, inference_target, *, nnef_op_type):
    """Emit one of tract's shift ops (`tract_shl` / `tract_shr`).

    PythonConstant shift counts (the `<< 2` form) traced as
    `prim::Constant[int]` lower to NNEF integer literals. tract's
    NNEF reader binds bare integer literals as TDim (its shape-arith
    type), and `ShiftLeft` / `ShiftRight` reject TDim at evaluation.
    Force the constant through a `tract_core_cast` to the data
    input's dtype so the shift sees a concrete int tensor.
    """
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    a_node, b_node = node.inputs
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    if isinstance(b_node, PythonConstant):
        b_node = b_node.into_tensor_variable()
        b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
        b_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_cast",
            inputs=b_ref,
            attrs={"to": TORCH_DTYPE_TO_TRACT_STR[a_node.dtype]},
            output_tensor_name_suffix="shift_count",
        )
    else:
        b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type,
        inputs=[a_ref, b_ref],
    )
    return ["tract_core"]


@OP_REGISTRY.register(torch_op_ids=["bitwise_left_shift", "__lshift__"])
def bitwise_left_shift(node, op_helper, inference_target, **kwargs):
    """Map `aten:bitwise_left_shift` / `<<` op to NNEF -> `tract_shl`.

    Tract's shift ops live under the `tract_core` extension despite
    the bare `tract_shl` / `tract_shr` names in the registry.
    """
    return _emit_bitwise_shift(
        node, op_helper, inference_target, nnef_op_type="tract_shl"
    )


@OP_REGISTRY.register(torch_op_ids=["bitwise_right_shift", "__rshift__"])
def bitwise_right_shift(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:bitwise_right_shift' / `>>` op -> `tract_shr`."""
    return _emit_bitwise_shift(
        node, op_helper, inference_target, nnef_op_type="tract_shr"
    )


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
