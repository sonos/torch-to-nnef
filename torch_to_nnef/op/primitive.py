import typing as T
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor
import torch

from .base import _torch_to_nnef_typestr


def add_tensor_to_ngraph(
    g,
    node,
    tensor: torch.Tensor,
    tensor_name: str,
    name_to_tensors: T.Dict[str, NTensor],
):
    name = f"{node.export_name}_{tensor_name}"
    tensor_np = tensor.numpy()
    ntensor = NTensor(
        g,
        name=name,
        shape=tuple(tensor.shape),
        dtype=tensor_np.dtype.type,
        data=tensor_np,
    )
    name_to_tensors[name] = ntensor
    return ntensor


def _unary_op(nnef_op_type: str, g, node, name_to_tensor, null_ref):
    out = NTensor(
        g,
        node.export_name,
        dtype=_torch_to_nnef_typestr(node.subtype or node.dtype),
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    outputs = [out]
    NOperation(
        graph=g,
        type=nnef_op_type,
        name=f"{node.export_name}_op",
        inputs=tuple(
            name_to_tensor[inp] if inp else null_ref
            for inp in node.export_inputs
        ),
        outputs=tuple(outputs),
        attribs={},
    )


def relu(torch_graph, **kwargs):
    return _unary_op("relu", **kwargs)


def _convolution(g, node, name_to_tensor, null_ref, torch_graph):
    # tuple of ints dilation, bool transposed, tuple of ints output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled
    (
        input_name,
        weight_name,
        bias_name,
        stride_name,
        padding_name,
        dilation_name,
        _,  # transposed_name
        _,  # output_padding_name
        groups_name,
        _,  # benchmark_name
        _,  # deterministic_name
        _,  # cuda_enabled
        _,  # allow_tf32
    ) = node.export_inputs

    weight = torch_graph.get_node_by_export_name(weight_name).data
    bias = torch_graph.get_node_by_export_name(bias_name).data

    stride = [
        torch_graph.get_node_by_export_name(in_export_name).value
        for in_export_name in torch_graph.get_node_by_export_name(
            stride_name
        ).export_inputs
    ]
    dilation = [
        torch_graph.get_node_by_export_name(in_export_name).value
        for in_export_name in torch_graph.get_node_by_export_name(
            dilation_name
        ).export_inputs
    ]
    padding = [
        torch_graph.get_node_by_export_name(in_export_name).value
        for in_export_name in torch_graph.get_node_by_export_name(
            padding_name
        ).export_inputs
    ]
    groups = torch_graph.get_node_by_export_name(groups_name).value

    nnef_weight_ref = add_tensor_to_ngraph(
        g, node, weight, "weight", name_to_tensor
    )

    out_tensor_name = node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=weight.numpy().dtype.type,
        shape=tuple(node.tensor_size) if node.tensor_size else None,
    )
    name_to_tensor[out_tensor_name] = output_tensor

    weight_var = NOperation(
        graph=g,
        type="variable",
        name=f"{node.export_name}_weight_var",
        inputs=None,
        outputs=nnef_weight_ref,
        attribs={
            "label": nnef_weight_ref.name,
            "shape": list(nnef_weight_ref.shape),
            "dtype": nnef_weight_ref.dtype,
        },
    )

    nnef_bias_ref = None
    if bias is not None:
        nnef_bias_ref = add_tensor_to_ngraph(
            g, node, bias, "bias", name_to_tensor
        )

    bias_var = NOperation(
        graph=g,
        type="variable",
        name=f"{node.export_name}_bias_var",
        inputs=None,
        outputs=nnef_bias_ref,
        attribs={
            "label": nnef_bias_ref.name,
            "shape": list(nnef_bias_ref.shape),
            "dtype": nnef_bias_ref.dtype,
        },
    )

    NOperation(
        graph=g,
        type="conv",
        name=f"{node.export_name}_op",
        inputs=(
            name_to_tensor[input_name],
            weight_var.output,
            bias_var.output,
        ),
        outputs=output_tensor,
        attribs={
            "dilation": list(dilation),
            "padding": [(pad_left, 0) for pad_left in padding],
            "stride": list(stride),
            "groups": groups,
            "border": "constant",
        },
    )


def aten_to_nnef_tensor_and_ops(g, node, name_to_tensor, null_ref, torch_graph):
    aten_op_name = node.kind.split("::")[1]

    globals()[aten_op_name](
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
    )
