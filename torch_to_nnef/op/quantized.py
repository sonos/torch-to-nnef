import typing as T

import numpy as np
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor


def _torch_qtensor_to_ntensor(g, tensor, name):
    np_int_tensor = tensor.int_repr().numpy()
    return NTensor(
        g,
        name=name,
        shape=tuple(tensor.shape),
        dtype=np_int_tensor.dtype.type,
        data=np_int_tensor,
        quant={
            "scale": tensor.q_scale(),
            "zero_point": tensor.q_zero_point(),
            "bits": 8,
            "signed": True,  # Should Be dependant of torch type quint vs qint
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


def _register_state_node_as_variable(
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


def _conv(packed_params, node, g, name_to_tensor):
    conv_weight = packed_params.weight()
    conv_bias = packed_params.bias()

    onode = node.outputs[0]
    stride = packed_params.stride()
    dilation = packed_params.dilation()
    padding = packed_params.padding()
    groups = packed_params.groups()
    weight_ref = _register_state_node_as_variable(
        conv_weight,
        slug_name="weight",
        node=onode,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    bias_ref = _register_state_node_as_variable(
        torch.quantize_per_tensor(
            conv_bias,
            scale=conv_weight.q_scale(),
            zero_point=conv_weight.q_zero_point(),
            dtype=conv_weight.dtype,
        ),
        slug_name="bias",
        node=onode,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    out_tensor_name = f"{onode.export_name}_conv"
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=np.int8,
    )
    name_to_tensor[out_tensor_name] = output_tensor

    NOperation(
        graph=g,
        type="conv",
        name=f"{onode.export_name}_conv",
        inputs=(
            name_to_tensor[node.inputs[0].export_name],
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
    )
    return output_tensor


def conv1d_relu(g, node, name_to_tensor, null_ref, torch_graph):

    packed_params = node.inputs[1].data

    conv_output_tensor = _conv(packed_params, node, g, name_to_tensor)

    onode = node.outputs[0]
    out_tensor_name = onode.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
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
