import typing as T
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor
import torch
import numpy as np

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


def _unary_op_with_constants(nnef_op_type, torch_graph, **kwargs):

    g = kwargs["g"]
    node = kwargs["node"]
    name_to_tensor = kwargs["name_to_tensor"]

    for const_node_name in node.export_inputs[1:]:
        const = torch_graph.get_node_by_export_name(const_node_name)
        if const.dtype == "Tensor":
            # is a NodeState

            # for now
            # in such case we inline tensor data values in graph def so it
            # should be kept small.

            # we might want to implement variable instead if too big
            data = const.data.numpy()
            nptype = data.dtype.type
        else:
            nptype = _torch_to_nnef_typestr(const.subtype or const.dtype)
            data = np.array(const.value, dtype=nptype)
        name_to_tensor[const_node_name] = NTensor(
            g,
            const.export_name,
            data=data,
            dtype=nptype,
            shape=const.tensor_size,
        )
    return _unary_op(nnef_op_type, **kwargs)


def relu(torch_graph, **kwargs):
    return _unary_op("relu", **kwargs)


def sigmoid(torch_graph, **kwargs):
    return _unary_op("sigmoid", **kwargs)


def tanh(torch_graph, **kwargs):
    return _unary_op("tanh", **kwargs)


def softmax(**kwargs):
    node = kwargs['node']
    if node.inputs[2]:
        del node.inputs[2]
    return _unary_op_with_constants("softmax", **kwargs)


def softplus(torch_graph, **kwargs):
    """
    Note: numerical stability applied in Pytorch is not done in NNEF vanilla
    implementation, nor case beta != 1.

    Pytorch ref:
        y = (1/beta) * log(exp(beta * x) + 1)  if ((beta * x) < thresh) else x
    NNEF ref:
        y = log(exp(x) + 1.0)

    """
    node = kwargs['node']
    const = torch_graph.get_node_by_export_name(node.export_inputs[1])
    if const.value != 1:
        raise NotImplemented(
            "This version is not implemented and"
            " would need use of a specific fragment"
        )
    node.inputs = node.inputs[:1]
    return _unary_op("softplus", **kwargs)


def elu(**kwargs):
    node = kwargs['node']
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_op_with_constants("elu", **kwargs)


def leaky_relu(**kwargs):
    node = kwargs['node']
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_op_with_constants("leaky_relu", **kwargs)


def prelu(**kwargs):
    node = kwargs['node']
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_op_with_constants("prelu", **kwargs)


def gelu(torch_graph, **kwargs):
    return _unary_op("gelu", **kwargs)


def selu(**kwargs):
    return _unary_op_with_constants("selu", **kwargs)


def silu(**kwargs):
    return _unary_op_with_constants("silu", **kwargs)


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

    def get_array_values_from_inputs(node_export_name: str):
        return [
            torch_graph.get_node_by_export_name(in_export_name).value
            for in_export_name in torch_graph.get_node_by_export_name(
                node_export_name
            ).export_inputs
        ]

    stride = get_array_values_from_inputs(stride_name)
    dilation = get_array_values_from_inputs(dilation_name)
    padding = get_array_values_from_inputs(padding_name)
    groups = torch_graph.get_node_by_export_name(groups_name).value

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        torch_graph,
        g,
        node,
        weight_name,
        bias_name,
        name_to_tensor,
        null_ref,
    )

    NOperation(
        graph=g,
        type="conv",
        name=f"{node.export_name}_op",
        inputs=(
            name_to_tensor[input_name],
            weight_ref,
            bias_ref,
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


def _register_state_node_as_variable(
    node_export_name: str, slug_name: str, torch_graph, node, g, name_to_tensor
):
    torch_tensor = torch_graph.get_node_by_export_name(node_export_name).data

    # peculiarity of Tract implementation
    if len(torch_tensor.shape) == 1:
        torch_tensor = torch_tensor.unsqueeze(0)

    nnef_tensor_ref = add_tensor_to_ngraph(
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
            "shape": list(nnef_tensor_ref.shape),
            "dtype": nnef_tensor_ref.dtype,
        },
    )

    return var.output


def _weight_bias_and_output_tensor(
    torch_graph,
    g,
    node,
    weight_name,
    bias_name,
    name_to_tensor,
    null_ref,
):
    weight_node = torch_graph.get_node_by_export_name(weight_name)
    weight = weight_node.data

    weight_ref = _register_state_node_as_variable(
        node_export_name=weight_name,
        slug_name="weight",
        torch_graph=torch_graph,
        node=node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    bias_ref = null_ref
    bias_node = torch_graph.get_node_by_export_name(bias_name)
    if hasattr(bias_node, 'data'):
        bias_ref = _register_state_node_as_variable(
            node_export_name=bias_name,
            slug_name="bias",
            torch_graph=torch_graph,
            node=node,
            g=g,
            name_to_tensor=name_to_tensor,
        )

    out_tensor_name = node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=weight.numpy().dtype.type,
        shape=tuple(node.tensor_size) if node.tensor_size else None,
    )
    name_to_tensor[out_tensor_name] = output_tensor
    return weight_ref, bias_ref, output_tensor


def linear(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_name,
        weight_name,
        bias_name,
    ) = node.export_inputs

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        torch_graph,
        g,
        node,
        weight_name,
        bias_name,
        name_to_tensor,
        null_ref,
    )

    NOperation(
        graph=g,
        type="linear",
        name=f"{node.export_name}_op",
        inputs=(name_to_tensor[input_name], weight_ref, bias_ref),
        outputs=output_tensor,
        attribs={},
    )


def batch_norm(g, node, name_to_tensor, null_ref, torch_graph):
    """

    nnef inputs:
        input: tensor<scalar>
        mean: tensor<scalar>
        variance: tensor<scalar>
        offset: tensor<scalar>
        scale: tensor<scalar>
        epsilon: scalar

    nnef op:
        output = offset + scale * (input - mean) / sqrt(variance + epsilon);
    """
    (
        input_name,
        weight_name,
        bias_name,
        running_mean_name,
        running_var_name,
        _,  # training
        _,  # momentum
        eps_name,
        _,  # cudnn_enabled
    ) = node.export_inputs

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        torch_graph,
        g,
        node,
        weight_name,
        bias_name,
        name_to_tensor,
        null_ref,
    )
    running_mean_ref = _register_state_node_as_variable(
        running_mean_name,
        slug_name="running_mean",
        torch_graph=torch_graph,
        node=node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    running_var_ref = _register_state_node_as_variable(
        running_var_name,
        slug_name="running_var",
        torch_graph=torch_graph,
        node=node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    eps_val = torch_graph.get_node_by_export_name(eps_name).value

    NOperation(
        graph=g,
        type="batch_normalization",
        name=f"{node.export_name}_op",
        inputs=(
            name_to_tensor[input_name],
            running_mean_ref,
            running_var_ref,
            bias_ref,
            weight_ref,
        ),
        outputs=output_tensor,
        attribs={"epsilon": eps_val},
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
