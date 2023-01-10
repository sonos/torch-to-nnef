"""
Quantized layers and primitives

Maybe usefull when looking at X:
    packed_params._method_names()
"""
import typing as T

import numpy as np
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive import _add_single_output_op
from torch_to_nnef.tract import tract_version_lower_than


def _torch_qtensor_to_ntensor(g, tensor, name):
    np_int_tensor = tensor.int_repr().numpy()
    np_dtype = np_int_tensor.dtype.type
    qscheme = tensor.qscheme()
    if qscheme == torch.per_channel_affine:
        qscale = tensor.q_per_channel_scales()
        qzerop = tensor.q_per_channel_zero_points()
    elif qscheme == torch.per_tensor_affine:
        qscale = tensor.q_scale()
        qzerop = tensor.q_zero_point()
    else:
        raise TorchToNNEFNotImplementedError(
            f"not suported quantization scheme {qscheme }"
        )
    return NTensor(
        g,
        name=name,
        shape=tuple(tensor.shape),
        dtype=np_dtype,
        data=np_int_tensor,
        quant={
            "scale": qscale,
            "zero_point": qzerop,
            "bits": np_dtype().nbytes * 8,
            "signed": np.issubdtype(np_dtype, np.signedinteger),
            "symmetric": False,
            "op-name": "zero_point_linear_quantize",
        },
    )


def register_bias_as_int(g, node, name_to_tensor, bias_tensor):
    """Simpler variable register for bias as int"""
    bias_tensor = bias_tensor.int_repr()
    # peculiarity of tract implementation
    if len(bias_tensor.shape) == 1:
        bias_tensor = bias_tensor.unsqueeze(0)
    name = f"{node.export_name}_bias"
    nnef_tensor_ref = NTensor(
        g,
        name=name,
        shape=tuple(bias_tensor.shape),
        dtype=np.int32,
        data=bias_tensor.numpy(),
    )

    name_to_tensor[name] = nnef_tensor_ref

    NOperation(
        graph=g,
        type="variable",
        name=f"{node.export_name}_bias_var",
        inputs=None,
        outputs=nnef_tensor_ref,
        attribs={
            "label": nnef_tensor_ref.name,
            "shape": list(bias_tensor.shape),
            "dtype": np.float32,  # since need to be marked as <scalar> in graph.nnef
        },
    )
    return nnef_tensor_ref


def add_quantized_tensor_to_ngraph(
    g,
    node,
    qtensor: torch.Tensor,
    tensor_name: str,
    name_to_tensor: T.Dict[str, NTensor],
):
    name = f"{node.export_name}_{tensor_name}"
    ntensor = _torch_qtensor_to_ntensor(g, qtensor, name)
    name_to_tensor[name] = ntensor
    return ntensor


def register_state_node_as_variable(
    torch_tensor: torch.Tensor,
    slug_name: str,
    node,
    g,
    name_to_tensor,
):

    # peculiarity of tract implementation
    if len(torch_tensor.shape) == 1:
        torch_tensor = torch_tensor.unsqueeze(0)
    nnef_tensor_ref = add_quantized_tensor_to_ngraph(
        g, node, torch_tensor, slug_name, name_to_tensor
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
            "dtype": np.float32,  # since need to be marked as <scalar> in graph.nnef
        },
    )

    return var.output


def _weight_bias(g, node, weight, bias, name_to_tensor):
    onode = node.outputs[0]
    weight_ref = register_state_node_as_variable(
        weight,
        slug_name="weight",
        node=onode,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    bias_ref = None
    if bias is not None:
        qscheme = weight.qscheme()
        if qscheme == torch.per_channel_affine:
            raise TorchToNNEFNotImplementedError(
                "tract does not support qscheme=per_channel_affine just yet"
            )
        if qscheme == torch.per_tensor_affine:
            input_quant_infos = name_to_tensor[node.inputs[0].export_name].quant
            if not input_quant_infos:
                input_quant_infos = node.inputs[0].quant
            bias_tensor = torch.quantize_per_tensor(
                bias.data,
                scale=weight.q_scale() * input_quant_infos["scale"],
                zero_point=weight.q_zero_point()
                + input_quant_infos["zero_point"],
                dtype=torch.qint32,
            )
        else:
            raise TorchToNNEFNotImplementedError(
                f"not suported quantization scheme {qscheme }"
            )
        if tract_version_lower_than("0.19.0"):
            bias_ref = register_state_node_as_variable(
                bias_tensor,
                slug_name="bias",
                node=onode,
                g=g,
                name_to_tensor=name_to_tensor,
            )
        else:
            bias_ref = register_bias_as_int(
                g, node, name_to_tensor, bias_tensor
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
    if conv_bias is not None and tract_version_lower_than("0.18.1"):
        for _ in range(input_node.rank - len(conv_bias.shape)):
            conv_bias = conv_bias.unsqueeze(0)
    # }

    stride = packed_params.stride()[-conv_rank:]
    dilation = packed_params.dilation()[-conv_rank:]
    padding = packed_params.padding()[-conv_rank:]
    groups = packed_params.groups()

    weight_ref, bias_ref = _weight_bias(
        g, node, conv_weight, conv_bias, name_to_tensor
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
    return output_tensor


def _linear(node, g, name_to_tensor, suffix_output_tensor: str = ""):
    (input_node, packed_params_node, scale_node, zero_point_node) = node.inputs

    packed_params = packed_params_node.data
    weight, bias = packed_params.unpack()
    for _ in range(input_node.rank - len(weight.shape)):
        weight = weight.unsqueeze(0)

    if bias is not None:
        for _ in range(input_node.rank - len(bias.shape)):
            bias = bias.unsqueeze(0)

    weight_ref, bias_ref = _weight_bias(g, node, weight, bias, name_to_tensor)
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


def conv1d_relu(g, node, name_to_tensor, null_ref, **kwargs):
    conv_output_tensor = _conv(
        node, g, name_to_tensor, null_ref, suffix_output_tensor="_conv"
    )

    out = _add_single_output_op(
        g, node, name_to_tensor, nnef_op_type="relu", inputs=conv_output_tensor
    )
    out.quant = conv_output_tensor.quant


def conv1d(g, node, name_to_tensor, null_ref, **kwargs):
    _conv(node, g, name_to_tensor, null_ref)


def linear(g, node, name_to_tensor, **kwargs):
    _linear(node, g, name_to_tensor)


def linear_relu(g, node, name_to_tensor, **kwargs):
    linear_output_tensor = _linear(
        node, g, name_to_tensor, suffix_output_tensor="_linear"
    )
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="relu",
        inputs=linear_output_tensor,
    )
    out.quant = linear_output_tensor.quant


def conv2d(g, node, name_to_tensor, null_ref, **kwargs):
    _conv(node, g, name_to_tensor, null_ref, conv_rank=2)


def conv2d_relu(g, node, name_to_tensor, null_ref, **kwargs):
    conv_output_tensor = _conv(
        node,
        g,
        name_to_tensor,
        null_ref,
        suffix_output_tensor="_conv",
        conv_rank=2,
    )

    out = _add_single_output_op(
        g, node, name_to_tensor, nnef_op_type="relu", inputs=conv_output_tensor
    )
    out.quant = conv_output_tensor.quant


def add_relu(g, node, name_to_tensor, null_ref, **kwargs):
    raise TorchToNNEFNotImplementedError()


def quantized_node_to_nnef_tensor_and_ops(
    g, node, name_to_tensor, null_ref, torch_graph, nnef_spec_strict: bool
):
    ops_family, op_name = node.kind.split("::")
    assert ops_family == "quantized"
    globals()[op_name](
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
        nnef_spec_strict=nnef_spec_strict,
    )
