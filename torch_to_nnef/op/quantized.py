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


def is_signed_q8(np_dtype: np.dtype):
    assert np_dtype in [np.uint8, np.int8]
    return np_dtype == np.int8


def _torch_qtensor_to_ntensor(g, tensor, name):
    np_int_tensor = tensor.int_repr().numpy()
    np_dtype = np_int_tensor.dtype.type
    return NTensor(
        g,
        name=name,
        shape=tuple(tensor.shape),
        dtype=np_dtype,
        data=np_int_tensor,
        quant={
            "scale": tensor.q_scale(),
            "zero_point": tensor.q_zero_point(),
            "bits": 8,
            "signed": is_signed_q8(np_dtype),
            "symmetric": False,
            "op-name": "zero_point_linear_quantize",
        },
    )


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


def _conv(node, g, name_to_tensor, null_ref, suffix_output_tensor=""):
    (input_node, packed_params_node, scale_node, zero_point_node) = node.inputs
    onode = node.outputs[0]

    packed_params = packed_params_node.data
    conv_weight = packed_params.weight().data
    conv_bias = packed_params.bias()

    stride = packed_params.stride()[1:]
    dilation = packed_params.dilation()[1:]
    padding = packed_params.padding()[1:]
    groups = packed_params.groups()

    weight_ref = register_state_node_as_variable(
        # 2nd axis is to remove in conv1d packed_params
        conv_weight.squeeze(2),
        slug_name="weight",
        node=onode,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    bias_ref = None
    if conv_bias is not None:
        bias_ref = register_state_node_as_variable(
            torch.quantize_per_tensor(
                conv_bias.data,
                scale=conv_weight.q_scale(),
                zero_point=conv_weight.q_zero_point(),
                dtype=conv_weight.dtype,
            ),
            slug_name="bias",
            node=onode,
            g=g,
            name_to_tensor=name_to_tensor,
        )

    out_tensor_name = f"{onode.export_name}{suffix_output_tensor}"

    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=np.int8,
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

    inputs = [
        name_to_tensor[input_node.export_name],
        weight_ref,
    ]
    if bias_ref is not None:
        inputs.append(bias_ref)

    NOperation(
        graph=g,
        type="conv",
        name=f"{onode.export_name}_conv",
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


def conv1d_relu(g, node, name_to_tensor, null_ref, torch_graph):

    conv_output_tensor = _conv(
        node, g, name_to_tensor, null_ref, suffix_output_tensor="_conv"
    )

    onode = node.outputs[0]
    out_tensor_name = onode.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=np.uint8,
        quant=conv_output_tensor.quant,
    )
    name_to_tensor[out_tensor_name] = output_tensor
    NOperation(
        graph=g,
        type="relu",
        name=f"{onode.export_name}_relu",
        inputs=conv_output_tensor,
        outputs=tuple([output_tensor]),
        attribs={},
    )


def conv1d(g, node, name_to_tensor, null_ref, torch_graph):
    _conv(node, g, name_to_tensor, null_ref)


def quantized_node_to_nnef_tensor_and_ops(
    g, node, name_to_tensor, null_ref, torch_graph
):
    ops_family, op_name = node.kind.split("::")
    assert ops_family == "quantized"
    globals()[op_name](
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
    )
