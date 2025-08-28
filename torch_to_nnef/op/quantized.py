"""PyTorch quantized::* operators translation.

Quantized layers and primitives

Maybe usefull when looking at X:
    packed_params._method_names()
"""

import typing as T

import numpy as np
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import InferenceTarget, TractNNEF
from torch_to_nnef.op.helper import (
    OpHelper,
    QuantizedOpRegistry,
    add_nnef_operation,
    add_single_output_op,
)
from torch_to_nnef.utils import torch_version

OP_REGISTRY = QuantizedOpRegistry()


def torch_qtensor_to_ntensor(g, tensor, name):
    np_int_tensor = tensor.int_repr().numpy()
    np_dtype = np_int_tensor.dtype.type
    qscheme = tensor.qscheme()
    if qscheme == torch.per_channel_affine:
        qscale = tensor.q_per_channel_scales().numpy()
        qzerop = tensor.q_per_channel_zero_points().numpy()
    elif qscheme == torch.per_tensor_affine:
        qscale = tensor.q_scale()
        qzerop = tensor.q_zero_point()
    else:
        raise T2NErrorNotImplemented(
            f"not suported quantization scheme {qscheme}"
        )
    n_bits = np_dtype().nbytes * 8
    if torch_version() >= "1.11.0":
        if tensor.dtype == torch.quint2x4:
            n_bits = 2
        elif tensor.dtype == torch.quint4x2:
            n_bits = 4

    return NTensor(
        g,
        name=name,
        shape=tuple(tensor.shape),
        dtype=np_dtype,
        data=np_int_tensor,
        quant={
            "scale": qscale,
            "zero_point": qzerop,
            "bits": n_bits,
            "signed": np.issubdtype(np_dtype, np.signedinteger),
            "symmetric": False,
            "op-name": "zero_point_linear_quantize",
        },
    )


def add_quantized_tensor_to_ngraph(
    g,
    node,
    qtensor: torch.Tensor,
    name_to_tensor: T.Dict[str, NTensor],
    tensor_name: T.Optional[str] = None,
):
    name = (
        f"{node.export_name}_{tensor_name}" if tensor_name else node.export_name
    )
    ntensor = torch_qtensor_to_ntensor(g, qtensor, name)
    name_to_tensor[name] = ntensor
    return ntensor


def register_state_node_as_variable(
    torch_tensor: torch.Tensor,
    slug_name: str,
    node,
    g,
    name_to_tensor,
    inference_target,
):
    # peculiarity of tract implementation
    if (
        len(torch_tensor.shape) == 1
        and isinstance(inference_target, TractNNEF)
        and inference_target.version < "0.18.1"
    ):
        torch_tensor = torch_tensor.unsqueeze(0)
    nnef_tensor_ref = add_quantized_tensor_to_ngraph(
        g, node, torch_tensor, name_to_tensor, slug_name
    )
    var = NOperation(
        graph=g,
        type="variable",
        name=f"{node.export_name}_{slug_name}_var",
        inputs=None,
        outputs=nnef_tensor_ref,
        attribs={
            "label": nnef_tensor_ref.name,
            "shape": list(torch_tensor.shape),
            "dtype": np.float32,  # since need to be marked as <scalar>
            # in graph.nnef
        },
    )

    return var.output


def _weight_bias(g, node, weight, bias, name_to_tensor, inference_target):
    onode = node.outputs[0]
    weight_ref = register_state_node_as_variable(
        weight,
        slug_name="weight",
        node=onode,
        g=g,
        name_to_tensor=name_to_tensor,
        inference_target=inference_target,
    )
    bias_ref = None
    if bias is not None and not (bias == 0).all():
        # we assume whatever qsheme is bias will always be float
        name = onode.export_name + "_bias"
        input_quant_infos = node.inputs[0].quant
        qscheme = weight.qscheme()
        if qscheme == torch.per_channel_affine:
            qscale = weight.q_per_channel_scales()
            qzerop = weight.q_per_channel_zero_points()
        elif qscheme == torch.per_tensor_affine:
            qscale = weight.q_scale()
            qzerop = weight.q_zero_point()
        else:
            raise T2NErrorNotImplemented(
                f"not suported quantization scheme {qscheme}"
            )

        bias_ref = NTensor(
            g,
            name,
            data=(bias.data / (qscale * input_quant_infos["scale"]) + qzerop)
            .round()
            .numpy()
            .astype(np.int32),
            dtype=np.int32,
            shape=tuple(bias.shape),
        )
        add_nnef_operation(
            graph=g,
            type="variable",
            inputs=None,
            outputs=bias_ref,
            attribs={
                "label": bias_ref.name,
                "shape": list(bias_ref.shape),
                "dtype": bias_ref.dtype,
            },
        )

    return weight_ref, bias_ref


def _output_tensor_from_s_and_zp(
    g,
    name_to_tensor,
    onode,
    scale_node,
    zero_point_node,
    suffix_output_tensor: str = "",
):
    out_tensor_name = f"{onode.export_name}{suffix_output_tensor}"
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=np.uint8,
        shape=onode.shape,
        quant={
            "scale": scale_node.data,
            "zero_point": zero_point_node.data,
            "bits": 8,
            "signed": False,
            "symmetric": False,
            "op-name": "zero_point_linear_quantize",
        },
    )
    name_to_tensor[out_tensor_name] = output_tensor
    return output_tensor


def _conv(
    node,
    g,
    name_to_tensor,
    null_ref,
    inference_target,
    suffix_output_tensor="",
    conv_rank: int = 1,
):
    (input_node, packed_params_node, scale_node, zero_point_node) = node.inputs

    packed_params = packed_params_node.data
    conv_weight = packed_params.weight().data
    conv_bias = packed_params.bias()

    # 2nd axis is to remove in conv1d packed_params
    if conv_rank == 1:
        conv_weight = conv_weight.squeeze(2)
    # apply expansion to align inputs with weight {
    for _ in range(input_node.rank - len(conv_weight.shape)):
        conv_weight = conv_weight.unsqueeze(0)

    if (
        conv_bias is not None
        and isinstance(inference_target, TractNNEF)
        and inference_target.version < "0.18.1"
    ):
        for _ in range(input_node.rank - len(conv_bias.shape)):
            conv_bias = conv_bias.unsqueeze(0)

    # }

    stride = packed_params.stride()[-conv_rank:]
    dilation = packed_params.dilation()[-conv_rank:]
    padding = packed_params.padding()[-conv_rank:]
    groups = packed_params.groups()

    weight_ref, bias_ref = _weight_bias(
        g, node, conv_weight, conv_bias, name_to_tensor, inference_target
    )
    output_tensor = _output_tensor_from_s_and_zp(
        g,
        name_to_tensor,
        node.outputs[0],
        scale_node,
        zero_point_node,
        suffix_output_tensor=suffix_output_tensor,
    )
    inputs = [
        name_to_tensor[input_node.export_name],
        weight_ref,
    ]
    if bias_ref is not None:
        inputs.append(bias_ref)

    # original
    NOperation(
        graph=g,
        type="conv",
        inputs=tuple(inputs),
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
    )

    # NOTE: Shall I use qconv instead ?
    # does not seems to work better on unit tests, nor full model export
    # --> fail compaction
    #
    # but add zp and scale for all IO see: /tract/nnef/src/ops/core/qconv.rs
    #
    # NOperation(
    #     graph=g,
    #     type="tract_core_qconv",
    #     inputs=tuple(inputs),
    #     outputs=output_tensor,
    #     attribs={
    #         "dilation": list(dilation),
    #         "padding": [
    #             (pad, pad) if isinstance(pad, int) else pad for pad in padding
    #         ],
    #         "stride": list(stride),
    #         "groups": groups,
    #         "border": "constant",
    #         # specific to qconv
    #         "a0": input_node.quant["zero_point"],
    #         "a_scale": input_node.quant["scale"],
    #         "b0": conv_weight.q_zero_point(),
    #         "b_scale": conv_weight.q_scale(),
    #         "c0": node.outputs[0].quant["zero_point"],
    #         "c_scale": node.outputs[0].quant["scale"],
    #     },
    # )

    return output_tensor


def _linear(
    node, g, name_to_tensor, inference_target, suffix_output_tensor: str = ""
):
    (input_node, packed_params_node, scale_node, zero_point_node) = node.inputs

    packed_params = packed_params_node.data
    weight, bias = packed_params.unpack()
    for _ in range(input_node.rank - len(weight.shape)):
        weight = weight.unsqueeze(0)

    if bias is not None:
        for _ in range(input_node.rank - len(bias.shape)):
            bias = bias.unsqueeze(0)

    weight_ref, bias_ref = _weight_bias(
        g, node, weight, bias, name_to_tensor, inference_target
    )
    output_tensor = _output_tensor_from_s_and_zp(
        g,
        name_to_tensor,
        node.outputs[0],
        scale_node,
        zero_point_node,
        suffix_output_tensor=suffix_output_tensor,
    )

    inputs = [
        name_to_tensor[input_node.export_name],
        weight_ref,
    ]
    if bias_ref is not None:
        inputs.append(bias_ref)

    NOperation(
        graph=g, type="linear", inputs=tuple(inputs), outputs=output_tensor
    )
    return output_tensor


@OP_REGISTRY.register()
def conv1d_relu(g, node, name_to_tensor, inference_target, null_ref, **kwargs):
    """Map PyTorch: 'quantized:conv1d_relu' to NNEF."""
    conv_output_tensor = _conv(
        node,
        g,
        name_to_tensor,
        null_ref,
        inference_target=inference_target,
        suffix_output_tensor="_conv",
    )

    out = add_single_output_op(
        g, node, name_to_tensor, nnef_op_type="relu", inputs=conv_output_tensor
    )
    out.quant = conv_output_tensor.quant


@OP_REGISTRY.register()
def conv1d(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Map PyTorch: 'quantized:conv1d' to NNEF."""
    _conv(
        node,
        g,
        name_to_tensor,
        null_ref,
        inference_target=inference_target,
    )


@OP_REGISTRY.register()
def linear(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'quantized:linear' to NNEF."""
    _linear(node, g, name_to_tensor, inference_target)


@OP_REGISTRY.register()
def linear_relu(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'quantized:linear_relu' to NNEF."""
    linear_output_tensor = _linear(
        node,
        g,
        name_to_tensor,
        inference_target,
        suffix_output_tensor="_linear",
    )
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="relu",
        inputs=linear_output_tensor,
    )
    out.quant = linear_output_tensor.quant


@OP_REGISTRY.register()
def conv2d(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Map PyTorch: 'quantized:conv2d' to NNEF."""
    _conv(node, g, name_to_tensor, null_ref, inference_target, conv_rank=2)


@OP_REGISTRY.register()
def conv2d_relu(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Map PyTorch: 'quantized:conv2d_relu' to NNEF."""
    conv_output_tensor = _conv(
        node,
        g,
        name_to_tensor,
        null_ref,
        inference_target,
        suffix_output_tensor="_conv",
        conv_rank=2,
    )

    out = add_single_output_op(
        g, node, name_to_tensor, nnef_op_type="relu", inputs=conv_output_tensor
    )
    out.quant = conv_output_tensor.quant


@OP_REGISTRY.register()
def add_relu(g, node, name_to_tensor, null_ref, **kwargs):
    """Map PyTorch: 'quantized:add_relu' to NNEF."""
    raise T2NErrorNotImplemented()


def math_op_binary(
    op_type: str, g, node, name_to_tensor, inference_target, **kwargs
):
    (
        x1_node,
        x2_node,
        _,
        _,
    ) = node.inputs  # _, _ are c_scale_node, c_offset_node
    out_node = node.outputs[0]
    out_nnef_tensor_ref = add_quantized_tensor_to_ngraph(
        g, out_node, out_node.tracing_data, name_to_tensor
    )

    x1_tensor = name_to_tensor[x1_node.export_name]
    x2_tensor = name_to_tensor[x2_node.export_name]
    if isinstance(inference_target, TractNNEF) and op_type not in [
        "mul",
        "add",
        "div",
    ]:
        # assume tract target
        # Tract is not assuming any alignment to do when applying
        # mul, add, div ...
        x1_out_nnef_tensor_ref = add_quantized_tensor_to_ngraph(
            g,
            x1_node,
            out_node.tracing_data,
            name_to_tensor,
            tensor_name="casted_aligned",
        )
        NOperation(
            graph=g,
            type="tract_core_cast",
            name=f"{x1_node.export_name}_cast_align",
            inputs=(x1_tensor),
            outputs=tuple([x1_out_nnef_tensor_ref]),
        )
        x1_tensor = x1_out_nnef_tensor_ref
        if x1_node.export_name == x2_node.export_name:
            x2_tensor = x1_out_nnef_tensor_ref
        else:
            x2_out_nnef_tensor_ref = add_quantized_tensor_to_ngraph(
                g,
                x2_node,
                out_node.tracing_data,
                name_to_tensor,
                tensor_name="casted_aligned",
            )
            NOperation(
                graph=g,
                type="tract_core_cast",
                name=f"{x2_node.export_name}_cast_align",
                inputs=(x2_tensor),
                outputs=tuple([x2_out_nnef_tensor_ref]),
            )
            x2_tensor = x2_out_nnef_tensor_ref

    NOperation(
        graph=g,
        type=op_type,
        name=f"{out_node.export_name}_{op_type}",
        inputs=(x1_tensor, x2_tensor),
        outputs=tuple([out_nnef_tensor_ref]),
    )


@OP_REGISTRY.register()
def mul(**kwargs):
    """Map PyTorch: 'quantized:mul' to NNEF."""
    math_op_binary(op_type="mul", **kwargs)
    return []


@OP_REGISTRY.register()
def add(**kwargs):
    """Map PyTorch: 'quantized:add' to NNEF."""
    math_op_binary(op_type="add", **kwargs)
    return []


@OP_REGISTRY.register()
def div(**kwargs):
    """Map PyTorch: 'quantized:div' to NNEF."""
    math_op_binary(op_type="div", **kwargs)
    return []


def quantized_node_to_nnef_tensor_and_ops(
    g,
    node,
    name_to_tensor,
    null_ref,
    torch_graph,
    inference_target: InferenceTarget,
):
    ops_family, op_name = node.kind.split("::")
    assert ops_family == "quantized"
    OP_REGISTRY.get(op_name)(
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
        inference_target=inference_target,
        op_helper=OpHelper(g, node, name_to_tensor, null_ref, inference_target),
    )
