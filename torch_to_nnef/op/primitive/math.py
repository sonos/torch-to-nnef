import logging

import numpy as np
import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    cast_to_if_not_dtype_and_variable,
    get_or_add_tensor_variable_in_nnef,
    unary_input_output_op_with_constant,
    unary_output_op_without_params,
)
from torch_to_nnef.torch_graph import PythonConstant
from torch_to_nnef.torch_graph.ir_data import TensorVariable

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def div(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    input_node = node.inputs[0]
    divisor_node = node.inputs[1]
    suffix_div_op_output = ""
    rounding_mode = None

    used_custom_fragment = []

    for c_node in [input_node, divisor_node]:
        c_node.cast_float_inplace()

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    divisor_tensor = get_or_add_tensor_variable_in_nnef(
        g, divisor_node, name_to_tensor
    )
    io_casting_with_dtype = None

    int_types = (torch.int8, torch.int16, torch.int32, torch.int64)
    if hasattr(input_node, "dtype") and input_node.dtype in int_types:
        io_casting_with_dtype = input_node.np_dtype
        if nnef_spec_strict:
            raise TorchToNNEFNotImplementedError(
                "What NNEF compliance mean in such case ?"
            )
        input_tensor, custom_fragments = cast_to_if_not_dtype_and_variable(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            nnef_tensor=input_tensor,
            cast_to=np.float32,
            suffix="casted",
        )
        used_custom_fragment += custom_fragments

    if len(node.inputs) == 3:
        rounding_mode = node.inputs[2].data

    if len(node.inputs) == 3 or io_casting_with_dtype is not None:
        suffix_div_op_output = "div"

    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix=suffix_div_op_output,
    )

    if rounding_mode:
        out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            rounding_mode,
            inputs=out,
            output_tensor_name_suffix=""
            if io_casting_with_dtype is None
            else rounding_mode,
        )
        if rounding_mode == "trunc":
            used_custom_fragment.append(rounding_mode)

    if io_casting_with_dtype is not None:
        if nnef_spec_strict:
            raise TorchToNNEFNotImplementedError(
                "What NNEF compliance mean in such case ?"
            )
        _, custom_fragments = cast_to_if_not_dtype_and_variable(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            nnef_tensor=out,
            cast_to=io_casting_with_dtype,
        )
        used_custom_fragment += custom_fragments
    return list(set(used_custom_fragment))


@OP_REGISTRY.register()
def floor_divide(
    g, node, name_to_tensor, has_dynamic_axes, torch_graph, **kwargs
):
    input_node, divisor_node = node.inputs
    if input_node.data and divisor_node.data and not has_dynamic_axes:
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

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    divisor_tensor = get_or_add_tensor_variable_in_nnef(
        g, divisor_node, name_to_tensor
    )
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix="div",
    )
    add_single_output_op(g, node, name_to_tensor, "floor", inputs=out)
    return []


@OP_REGISTRY.register()
def trunc(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "trunc",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, node.inputs[0], name_to_tensor
        ),
    )
    return ["trunc"]


@OP_REGISTRY.register(torch_op_ids=["pow"])
def pow_(g, node, name_to_tensor, **kwargs):
    (input_node, exponent_node) = node.inputs
    inputs = [get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)]
    if exponent_node.data:
        exponent = exponent_node.data
        if exponent == 2:
            op_type = "sqr"
        elif exponent == -2:
            op_type = "rsqr"
        else:
            op_type = "pow"
            inputs += [
                get_or_add_tensor_variable_in_nnef(
                    g, exponent_node, name_to_tensor
                )
            ]
    else:
        op_type = "pow"
        inputs += [
            get_or_add_tensor_variable_in_nnef(g, exponent_node, name_to_tensor)
        ]

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        op_type,
        inputs=inputs,
    )


@OP_REGISTRY.register(torch_op_ids=["round"])
def round_(nnef_spec_strict, **kwargs):
    if nnef_spec_strict:
        LOGGER.warning(
            "round: Spec definition of round in NNEF does not follow IEEE, "
            "so it will not be exactly same behavior"
        )
        unary_input_output_op_with_constant("round", **kwargs)
        return []
    unary_input_output_op_with_constant("tract_core_round_even", **kwargs)
    return ["tract_core"]


@OP_REGISTRY.register()
def mul(g, node, name_to_tensor, **kwargs):
    input_node = node.inputs[0]
    other_node = node.inputs[1]

    inputs = []
    for c_node in [input_node, other_node]:
        if isinstance(c_node, PythonConstant):
            # because torch.ops.aten.mul(float, tensor(float)) give complex number
            c_node = c_node.into_tensor_variable()
        if any(nod.dtype.is_floating_point for nod in [input_node, other_node]):
            c_node.cast_float_inplace()
        inputs.append(
            get_or_add_tensor_variable_in_nnef(g, c_node, name_to_tensor)
        )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "mul",
        inputs=inputs,
    )


@OP_REGISTRY.register()
def remainder(g, node, name_to_tensor, torch_graph, **kwargs):
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
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "remainder",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            for _ in [input_node, other_node]
        ],
    )
    return ["remainder"]


@OP_REGISTRY.register()
def rsub(g, node, name_to_tensor, torch_graph, **kwargs):
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
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "rsub",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            for _ in [input_node, other_node]
        ],
        attrs={"alpha": alpha_node.data},
    )
    return ["rsub"]


@OP_REGISTRY.register(torch_op_ids=["abs"])
def _abs(
    g,
    node,
    name_to_tensor,
    null_ref,
    nnef_spec_strict,
    tract_feature_flags,
    torch_graph,
    **kwargs,
):
    if node.inputs[0].dtype in [torch.complex64, torch.complex128]:
        if nnef_spec_strict:
            raise TorchToNNEFNotImplementedError(
                "NNEF compliance does not allow complex"
            )
        input_tensor = get_or_add_tensor_variable_in_nnef(
            g, node.inputs[0], name_to_tensor
        )
        # to real, pow(2), slice both, add 2 tensors, rsqr
        if tract_feature_flags is not None and "complex" in tract_feature_flags:
            input_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_complex_to_inner_dim",
                inputs=input_tensor,
                output_tensor_name_suffix="complex_abs_to_real",
            )

        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "sqr",
            inputs=input_tensor,
            output_tensor_name_suffix="complex_abs_sqr",
        )
        input_tensor_real = add_single_output_op(
            g,
            node,
            name_to_tensor,
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
        input_tensor_imag = add_single_output_op(
            g,
            node,
            name_to_tensor,
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

        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "add",
            inputs=[input_tensor_real, input_tensor_imag],
            output_tensor_name_suffix="complex_abs_add",
        )
        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "sqrt",
            inputs=input_tensor,
            output_tensor_name_suffix="complex_abs_sqrt",
        )
        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=input_tensor,
            attrs={"axes": [len(input_tensor.shape)]},
        )
        return []
    return unary_output_op_without_params(
        nnef_op_type="abs",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


@OP_REGISTRY.register()
def log10(g, node, name_to_tensor, **kwargs):
    """mul val may not be good enough"""
    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, node.inputs[0], name_to_tensor
    )
    # maybe better puting this in the graph to avoid precision loss
    mul_val = 1 / np.log(10)
    input_tensor = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "log",
        inputs=input_tensor,
        output_tensor_name_suffix="pre_log10",
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "mul",
        inputs=[input_tensor],
        attrs={"y": mul_val},
    )
