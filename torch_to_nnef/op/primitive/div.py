import numpy as np
import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    add_single_output_op,
    cast_to_if_not_dtype_and_variable,
    get_or_add_tensor_variable_in_nnef,
)


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


def floor_divide(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    input_node, divisor_node = node.inputs
    for c_node in [input_node, divisor_node]:
        c_node.cast_float_inplace()

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
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "trunc",
        inputs=out,
    )
    return ["trunc"]


def trunc(g, node, name_to_tensor, **kwargs):
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
