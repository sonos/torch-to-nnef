# pylint: disable=too-many-lines
import logging
import typing as T

import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.activation import (
    clamp,
    clamp_max,
    clamp_min,
    elu,
    erf,
    gelu,
    glu,
    hardtanh,
    leaky_relu,
    log_softmax,
    prelu,
    selu,
    silu,
    softmax,
    softplus,
)
from torch_to_nnef.op.primitive.base import (
    add_multi_output_op,
    add_nnef_operation,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
    pick_value_in_rank,
    unary_input_output_op_with_constant,
    unary_output_op_without_params,
    weight_bias_and_output_tensor,
)
from torch_to_nnef.op.primitive.complex import view_as_complex, view_as_real
from torch_to_nnef.op.primitive.div import div, floor_divide, trunc
from torch_to_nnef.op.primitive.expand import expand
from torch_to_nnef.op.primitive.fft import fft_fft, fft_ifft, stft
from torch_to_nnef.op.primitive.norm import (
    batch_norm,
    group_norm,
    layer_norm,
    norm,
)
from torch_to_nnef.op.primitive.pad import pad
from torch_to_nnef.op.primitive.pool import (
    adaptive_avg_pool2d,
    avg_pool1d,
    avg_pool2d,
    max_pool1d,
    max_pool2d,
)
from torch_to_nnef.op.primitive.qops import dequantize, quantize_per_tensor
from torch_to_nnef.op.primitive.reducer import (
    argmax,
    argmin,
    max_,
    mean,
    min_,
    reduce_all,
    reduce_any,
    reduce_max,
    reduce_min,
    reduce_sum,
)
from torch_to_nnef.op.primitive.selector import (
    index_,
    narrow,
    select,
    slice_,
    where,
)
from torch_to_nnef.op.primitive.tensor_build import (
    arange,
    new_zeros,
    ones,
    zeros,
    zeros_like,
)
from torch_to_nnef.torch_graph import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.tract import tract_version_lower_than

# silence pyflakes F401 {
assert view_as_complex
assert view_as_real
assert fft_fft
assert fft_ifft
assert stft

assert softmax
assert softplus
assert elu
assert leaky_relu
assert prelu
assert selu
assert silu
assert gelu
assert erf
assert hardtanh
assert log_softmax
assert glu
assert clamp
assert clamp_min
assert clamp_max

assert norm
assert batch_norm
assert group_norm
assert layer_norm

assert max_pool1d
assert avg_pool1d
assert max_pool2d
assert avg_pool2d
assert adaptive_avg_pool2d

assert arange
assert ones
assert zeros_like
assert new_zeros
assert zeros

assert pad

assert expand
assert slice_
assert where
assert narrow
assert select
assert index_

assert div
assert floor_divide
assert trunc

assert quantize_per_tensor
assert dequantize

assert mean
assert reduce_sum
assert argmax
assert argmin
assert reduce_any
assert reduce_all
assert reduce_max
assert reduce_min
assert max_
assert min_
# }


LOGGER = logging.getLogger(__name__)

REMAP_ATEN_OP_NAMES = {
    "_relu": "relu",
    "reciprocal": "rcp",
    "clone": "copy",
    "bitwise_not": "not",
    "bitwise_not_cpu": "not",
    "bitwise_cpu": "and",
    "__and_": "and",
    "__or_": "or",
    "less": "lt",
    "greater": "gt",
    "less_equal": "le",
    "greater_equal": "ge",
    "reflection_pad1d": "reflection_padnd",
    "replication_pad1d": "replication_padnd",
    "constant_pad1d": "constant_padnd",
    # avoid to ovewrite python builtin {
    "any": "reduce_any",
    "all": "reduce_all",
    "sum": "reduce_sum",
    "pow": "pow_",
    "max": "max_",
    "min": "min_",
    "slice": "slice_",
    "round": "round_",
    "index": "index_",
    # }
    "bmm": "matmul",  # since NNEF matmul does not care about rank
    "amax": "reduce_max",
}

GENERIC_UNARY_OUTPUT_ATEN_OP_NAMES = [
    "relu",
    "sigmoid",
    "log",
    "exp",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sign",
    "neg",
    "floor",
    "ceil",
    "sqrt",
    "rsqrt",
    "log2",
    "rcp",
    "not",
    "eq",
    "ne",
    "add",
    "sub",
    "lt",
    "gt",
    "le",
    "ge",
    "and",
    "or",
]


def external(
    g: NGraph, node: TensorVariable, name_to_tensor: T.Dict[str, NTensor]
):
    """Add External NNEF Operation in graph"""
    nnef_tensor_ref = add_tensor_variable_node_as_nnef_tensor(
        g, node, name_to_tensor, prevent_variable=True
    )
    add_nnef_operation(
        graph=g,
        type="external",
        inputs=None,
        outputs=nnef_tensor_ref,
        attribs={
            "shape": list(nnef_tensor_ref.shape),
            "dtype": nnef_tensor_ref.dtype,
        },
    )
    return nnef_tensor_ref


def mul(g, node, name_to_tensor, **kwargs):
    input_node = node.inputs[0]
    other_node = node.inputs[1]

    inputs = []
    for c_node in [input_node, other_node]:
        if isinstance(c_node, PythonConstant):
            # because torch.ops.aten.mul(float, tensor(float)) give complex number
            c_node = c_node.into_tensor_variable()
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


def _convolution(g, node, name_to_tensor, null_ref, **kwargs):
    (
        input_node,
        weight_node,
        bias_node,
        stride_node,
        padding_node,
        dilation_node,
        transposed_node,
        _,  # output_padding_name
        groups_node,
        _,  # benchmark_name
        _,  # deterministic_name
        _,  # cuda_enabled
        _,  # allow_tf32
    ) = node.inputs

    stride = stride_node.data
    dilation = dilation_node.data
    padding = padding_node.data
    groups = groups_node.data
    transposed = transposed_node.data

    if transposed:
        weight_node.data = weight_node.data.transpose(1, 0)

    # expand in stored variables export to avoid unsqueeze guessing in graph {
    params_nodes = [weight_node]
    if bias_node.data is not None and tract_version_lower_than("0.18.1"):
        params_nodes.append(bias_node)
    for param_node in params_nodes:
        for _ in range(input_node.rank - param_node.rank):
            param_node.data = param_node.data.unsqueeze(0)
            param_node.shape = list(param_node.data.shape)
    # }

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="deconv" if transposed else "conv",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            weight_ref,
            bias_ref,
        ),
        outputs=output_tensor,
        attribs={
            "dilation": list(dilation),
            "padding": [
                (pad, pad) if isinstance(pad, int) else pad for pad in padding
            ],
            "stride": list(stride),
            "groups": groups,
            "border": "constant",
        },
        force_consistent_inputs_shapes=False,
    )


def linear(g, node, name_to_tensor, null_ref, **kwargs):
    (
        input_node,
        weight_node,
        bias_node,
    ) = node.inputs

    # expand in stored variable export to avoid adding unsqueeze in graph {

    if weight_node.data is not None:
        for _ in range(input_node.rank - weight_node.rank):
            weight_node.data = weight_node.data.unsqueeze(0)
            weight_node.shape = list(weight_node.data.shape)

    if bias_node.data is not None:
        for _ in range(input_node.rank - bias_node.rank):
            bias_node.data = bias_node.data.unsqueeze(0)
            bias_node.shape = list(bias_node.data.shape)
    # }

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="linear",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            weight_ref,
            bias_ref,
        ),
        outputs=output_tensor,
        attribs={},
    )


def dropout(node, torch_graph, **kwargs):
    (
        input_node,
        _,  # probability
        is_active_node,
    ) = node.inputs
    # should wire directly input_node to output without intermediate
    if is_active_node.data:
        raise TorchToNNEFNotImplementedError("dropout active at inference")

    # this replace order is important for graph of single nodes or starting with
    torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


def detach(node, torch_graph, **kwargs):
    """This does not translate to any operation"""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


def contiguous(node, torch_graph, **kwargs):
    """This does not translate to any operation"""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


def view(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    (input_node, axis_node) = node.inputs
    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=has_dynamic_axes,
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


def flatten(g, node, name_to_tensor, **kwargs):
    """
    Using NNEF:
        fragment reshape<?>(
            input: tensor<?>,
            shape: integer[],
            axis_start: integer = 0,
            axis_count: integer = -1
        ) -> ( output: tensor<?> );
    """
    (input_node, _, _) = node.inputs  # start_dim_name  # end_dim_name
    onode = node.outputs[0]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "dtype": onode.np_dtype,
            "shape": list(onode.shape),
            "axis_start": 0,
            "axis_count": -1,
        },
    )


def einsum(g, node, name_to_tensor, **kwargs):
    raise TorchToNNEFNotImplementedError(
        "einsum operator is not supported by `NNEF` or `tract-nnef` and"
        "breaking it down to primite ops may be tricky"
    )


def to(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        input_node,
        *_,  # dtype_name, non_blocking_name, copy_name, memory_format_name
    ) = node.inputs

    onode = node.outputs[0]
    LOGGER.warning(
        "convert .to() with tract custom operator since it can express "
        "all torch type (contrary to vanilla cast NNEF operator)"
    )
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError("`to` with nnef_spec_strict ?")
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "to": TORCH_DTYPE_TO_TRACT_STR[onode.dtype],
            # "shape": list(onode.shape),
        },
    )
    return ["tract_core"]


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


def transpose(g, node, name_to_tensor, **kwargs):
    (input_node, dim0_node, dim1_node) = node.inputs
    dim0 = pick_rank(input_node, dim0_node.data)
    dim1 = pick_rank(input_node, dim1_node.data)

    new_dims_ranks = []
    for _ in range(node.outputs[0].rank):
        if _ == dim0:
            new_dims_ranks.append(dim1)
        elif _ == dim1:
            new_dims_ranks.append(dim0)
        else:
            new_dims_ranks.append(_)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": new_dims_ranks},
        pass_quantization_params=True,
    )


def permute(g, node, name_to_tensor, **kwargs):
    (input_node, dims_node) = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, _) for _ in dims_node.data]},
        pass_quantization_params=True,
    )


def cat(g, node, name_to_tensor, torch_graph, **kwargs):
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise TorchToNNEFNotImplementedError(
                f"cat with input_item: {input_item}"
            )
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    axis = pick_rank(input_node.data[0], dim)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "concat",
        inputs=inputs,
        attrs={"axis": axis},
        ensure_tuple=False,
    )


def stack(g, node, name_to_tensor, torch_graph, **kwargs):
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    assert isinstance(input_node, FixedTensorList)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise TorchToNNEFNotImplementedError(
                f"stack with input_item: {input_item}"
            )
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "stack",
        inputs=inputs,
        attrs={"axis": pick_rank(input_node, dim)},
        ensure_tuple=False,
    )


def unbind(g, node, name_to_tensor, **kwargs):
    """unbind is `unstack` in NNEF"""
    input_node, axis_node = node.inputs
    add_multi_output_op(
        g,
        node,
        name_to_tensor,
        "unstack",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axis": pick_rank(input_node, axis_node.data)},
        ensure_tuple=False,
    )


def unsqueeze(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs

    dim = axis_node.data
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, dim)]},
        pass_quantization_params=True,
    )


def squeeze(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, dim)]},
        pass_quantization_params=True,
    )


def repeat(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"repeats": axis_node.data},
    )


def size(
    g,
    node,
    name_to_tensor,
    nnef_spec_strict,
    has_dynamic_axes,
    torch_graph,
    **kwargs,
):
    """
    We can not use NNEF shape_of that have been deprecated since 1.0.1 version:

    ```
    The shape_of function is deprecated and is discouraged from use.
    The reason is that it provides syntactic means to access a
    property of tensors that is not defined via the syntax itself.

    Furthermore, its definition is problematic in cases where the shape
    of a tensor is not known in graph compilation time.

    These result in problems with custom operations and operations with results
    of dynamic shape for a consumer of an NNEF document.

    By removing support for the shape_of function from NNEF syntax,
    it becomes possible to de-couple parsing
    from shape propagation in a consumer of an NNEF document.
    ```

    Since it is a core component to express some dynamic network that may use
    tract symbolic dimensions:
    by example using stream size to apply an averaging:
    We map it to `tract_core_shape_of`

    """
    input_node, axis_node = node.inputs
    if nnef_spec_strict or not has_dynamic_axes:
        original_vec_node, axis_node = node.inputs
        original_variable_output = node.outputs[0]
        if original_variable_output.data is None:
            dim = original_vec_node.shape[axis_node.data]
        else:
            dim = original_variable_output.data.numpy().tolist()
        new_node = PythonConstant(
            name=original_variable_output.name,
            data=dim,
        )
        torch_graph.remap_node(original_variable_output, new_node)

        for data_node in torch_graph.data_nodes:
            if (
                isinstance(data_node, FixedTensorList)
                and any(_ is new_node for _ in data_node.data)
                and all(isinstance(_, PythonConstant) for _ in data_node.data)
            ):
                # recompute fixed data based on new infos
                torch_graph.remap_node(
                    data_node,
                    PythonConstant(
                        name=data_node.name,
                        data=[_.data for _ in data_node.data],
                    ),
                )
        torch_graph.op_nodes = [
            _ for _ in torch_graph.op_nodes if _ is not node
        ]

        LOGGER.warning(
            "aten::size replaced by constant traced value (follows NNEF spec)."
            "Keeping dynamism would require dynamic_axes specified."
        )
        return []
    # original_variable_output = node.outputs[0]

    # ensure consistant name to avoid strangeness
    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    shape_tensor_name = f"{input_tensor.name}_shape"
    if shape_tensor_name in name_to_tensor:
        out = name_to_tensor[shape_tensor_name]
    else:
        out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_shape_of",
            inputs=input_tensor,
            force_full_output_tensor_name=shape_tensor_name,
        )

    begin = pick_rank(input_node, axis_node.data)

    index_tensor_name = f"{shape_tensor_name}_{begin}"
    if index_tensor_name not in name_to_tensor:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=out,
            attrs={
                "axes": [0],
                "begin": [begin],
                "end": [begin + 1],
                "stride": [1],
            },
            force_full_output_tensor_name=index_tensor_name,
        )
    outnode = node.outputs[0]
    new_outnode = torch_graph.find_node(index_tensor_name)
    if not new_outnode:
        new_outnode = TensorVariable(
            name=index_tensor_name,
            data=outnode.data,
            shape=outnode.shape,
            dtype=outnode.dtype,
        )
    torch_graph.remap_node(
        from_node=outnode,
        to_node=new_outnode,
    )

    return ["tract_core"]


def reshape(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    (input_node, axis_node) = node.inputs

    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=has_dynamic_axes,
        force_none_as_tensor_ref=True,
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


def matmul(g, node, name_to_tensor, **kwargs):
    (
        input_node,
        other_node,
    ) = node.inputs

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "matmul",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            get_or_add_tensor_variable_in_nnef(g, other_node, name_to_tensor),
        ),
        attrs={
            "transposeA": False,
            "transposeB": False,
        },
    )


def split_with_sizes(g, node, name_to_tensor, **kwargs):
    """We are aware that
    split<?>(
        value: tensor<?>,
        axis: integer,
        ratios: integer[]
    ) -> ( values: tensor<?>[] )

    exists but since tract does not support it, we reexpress it with slice
    """
    (input_node, ratio_node, axis_node) = node.inputs
    assert isinstance(axis_node, PythonConstant)
    assert isinstance(ratio_node, PythonConstant)
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node, n_elements in zip(node.outputs, ratio_node.data):
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if n_elements <= 0:
            raise TorchToNNEFNotImplementedError("unexpected n_elements<=0")
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [pick_rank(input_node, axis_node.data)],
                "begin": [current_dim_elm_idx],
                "end": [current_dim_elm_idx + n_elements],
                "stride": [1],
            },
        )
        if inputs.quant:
            out.quant = inputs.quant
        current_dim_elm_idx += n_elements


def masked_fill(g, node, name_to_tensor, **kwargs):
    input_node, mask_node, value_node = node.inputs

    false_value_node = input_node
    true_value_node = value_node.into_tensor_variable()
    true_value_node.data = true_value_node.data.to(
        false_value_node.dtype
    ).repeat(false_value_node.shape)
    true_value_node.dtype = false_value_node.dtype

    # tract need float where ?
    # mask_node.data = mask_node.data.float()
    # mask_node.dtype = mask_node.data.dtype
    condition_node = mask_node

    inputs = [
        get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
        for _ in [condition_node, true_value_node, false_value_node]
    ]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="select",
        inputs=inputs,
    )


def chunk(g, node, name_to_tensor, **kwargs):
    (input_node, n_chunk_node, axis_node) = node.inputs
    assert n_chunk_node.data == len(node.outputs)
    assert (
        len({tuple(o.shape) for o in node.outputs}) == 1
    ), "all chunk are not equal"
    n_elements = node.outputs[0].shape[axis_node.data]
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node in node.outputs:
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [pick_rank(input_node, axis_node.data)],
                "begin": [current_dim_elm_idx],
                "end": [current_dim_elm_idx + n_elements],
                "stride": [1],
            },
        )
        current_dim_elm_idx += n_elements


def baddbmm(g, node, name_to_tensor, **kwargs):
    input_node, batch1_node, batch2_node, beta_node, alpha_node = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "baddbmm",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            for _ in [input_node, batch1_node, batch2_node]
        ],
        attrs={"beta": beta_node.data, "alpha": alpha_node.data},
    )
    return ["baddbmm"]


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


def roll(g, node, name_to_tensor, has_dynamic_axes, nnef_spec_strict, **kwargs):
    input_node, shifts_node, dims_node = node.inputs
    shifts = shifts_node.data
    dims = dims_node.data
    assert len(shifts) == len(dims), "shifts and dims need to be sample size"
    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    for i, _ in enumerate(shifts):
        tensor_chunks = []
        dim = dims[i]
        shift = shifts[i]
        if not has_dynamic_axes or nnef_spec_strict:
            maxsize = input_node.shape[dim]
        else:
            raise TorchToNNEFNotImplementedError("Should use shape_of")
        shape_out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=input_tensor,
            attrs={
                "axes": [pick_rank(input_node, dim)],
                "begin": [pick_value_in_rank(input_node, dim, -shift)],
                "end": [pick_value_in_rank(input_node, dim, maxsize)],
                "stride": [1],
            },
            output_tensor_name_suffix=f"roll_l{i}_p1",
        )
        tensor_chunks.append(shape_out)
        shape_out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=input_tensor,
            attrs={
                "axes": [pick_rank(input_node, dim)],
                "begin": [0],
                "end": [pick_value_in_rank(input_node, dim, -shift)],
                "stride": [1],
            },
            output_tensor_name_suffix=f"roll_l{i}_p2",
        )
        tensor_chunks.append(shape_out)
        # result = g.op("Concat", *shapes, axis_i=dims[i])
        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "concat",
            inputs=tensor_chunks,
            attrs={"axis": pick_rank(input_node, dim)},
            ensure_tuple=False,
            output_tensor_name_suffix=""
            if i + 1 == len(shifts)
            else f"roll_{i}",
        )
    return []


def embedding(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        weight_node,
        indices_node,
        padding_idx_node,
        scale_grad_by_freq_node,
        sparse_node,
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


def abs(
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
        return
    return unary_output_op_without_params(
        nnef_op_type="abs",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


def log10(g, node, name_to_tensor, **kwargs):
    """mul val may not be good enought"""
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


def copy(
    g, node, name_to_tensor, nnef_spec_strict, torch_graph, null_ref, **kwargs
):
    if nnef_spec_strict:
        # nnef spec include copy fragment
        return unary_output_op_without_params(
            nnef_op_type="copy",
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    torch_graph.remap_node(node.outputs[0], node.inputs[0])


def aten_to_nnef_tensor_and_ops(
    g,
    node,
    name_to_tensor,
    null_ref,
    torch_graph,
    nnef_spec_strict: bool = False,
    has_dynamic_axes: bool = False,
    tract_feature_flags: T.Optional[T.Set[str]] = None,
) -> T.Optional[T.List[str]]:
    """Main primitive dispatcher

    Allow to write in graph any not Quantized Operation from pytorch defined in
    node attribute.

    """
    aten_op_name = node.kind.split("::")[1]

    # remap
    if aten_op_name.endswith("_"):
        aten_op_name = aten_op_name[:-1]
    aten_op_name = REMAP_ATEN_OP_NAMES.get(aten_op_name, aten_op_name)

    if aten_op_name in GENERIC_UNARY_OUTPUT_ATEN_OP_NAMES:
        return unary_output_op_without_params(
            nnef_op_type=aten_op_name,
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    try:
        return globals()[aten_op_name](
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
            torch_graph=torch_graph,
            nnef_spec_strict=nnef_spec_strict,
            has_dynamic_axes=has_dynamic_axes,
            tract_feature_flags=tract_feature_flags,
        )
    except KeyError as exp:
        torch_graph.printall()
        raise exp
