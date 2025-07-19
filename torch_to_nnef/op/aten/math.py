import logging

import numpy as np
import torch

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR, dtype_is_whole_number
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten import other
from torch_to_nnef.op.aten.complex import tract_complex_support
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    unary_input_output_op_with_constant,
)
from torch_to_nnef.torch_graph import PythonConstant
from torch_to_nnef.torch_graph.ir_data import TensorVariable

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def div(node, op_helper, inference_target, torch_graph, **kwargs):
    input_node = node.inputs[0]
    divisor_node = node.inputs[1]
    suffix_div_op_output = ""
    rounding_mode = None

    if input_node.data is not None and divisor_node.data is not None:
        node.outputs[0].data = input_node.data / divisor_node.data
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
                "div: Mixing input of 2 different dtype:"
                f" {(input_node.dtype, divisor_node.dtype)}"
                " force cast to f32"
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
        if isinstance(inference_target, TractNNEF):
            io_casting_with_dtype = np.uint64
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
            raise TorchToNNEFNotImplementedError(
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
    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix="div",
    )
    op_helper.add_single_output_op_from_nnef_tensors(node, "floor", inputs=out)
    return []


@OP_REGISTRY.register()
def trunc(node, op_helper, **kwargs):
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "trunc",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0]),
    )
    return ["trunc"]


@OP_REGISTRY.register(torch_op_ids=["pow"])
def pow_(node, op_helper, **kwargs):
    (input_node, exponent_node) = node.inputs
    inputs = [op_helper.get_or_add_tensor_variable_in_nnef(input_node)]
    if exponent_node.data:
        exponent = exponent_node.data
        if exponent == 2:
            op_type = "sqr"
        elif exponent == -2:
            op_type = "rsqr"
        else:
            op_type = "pow"
            inputs += [
                op_helper.get_or_add_tensor_variable_in_nnef(exponent_node)
            ]
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
    input_node = node.inputs[0]
    other_node = node.inputs[1]

    if input_node.data is not None and other_node.data is not None:
        node.outputs[0].data = input_node.data * other_node.data
        return
    if remap_if_neutral_op(
        torch_graph, node, input_node, other_node
    ) or remap_if_neutral_op(torch_graph, node, other_node, input_node):
        return

    inputs = []
    for c_node in [input_node, other_node]:
        if isinstance(c_node, PythonConstant):
            # because torch.ops.aten.mul(float, tensor(float)) give complex number
            c_node = c_node.into_tensor_variable()
        inputs.append(op_helper.get_or_add_tensor_variable_in_nnef(c_node))
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=inputs,
    )


@OP_REGISTRY.register()
def remainder(node, op_helper, torch_graph, inference_target, **kwargs):
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


@OP_REGISTRY.register()
def rsub(node, op_helper, torch_graph, **kwargs):
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
        alpha_node.data = float(alpha_node.data)
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
    if node.inputs[0].dtype in [torch.complex64, torch.complex128]:
        if not isinstance(inference_target, TractNNEF):
            raise TorchToNNEFNotImplementedError(
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
    """mul val may not be good enough"""
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
    """aten::log1p"""
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "log1p",
        inputs=input_tensor,
    )
    return ["log1p"]


@OP_REGISTRY.register()
def atan2(node, op_helper, **kwargs):
    """aten::atan2"""
    input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    other_tensor = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[1])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "atan2",
        inputs=(input_tensor, other_tensor),
    )
    return ["atan2"]
