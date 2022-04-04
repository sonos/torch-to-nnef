# pylint: disable=too-many-lines
import logging
import typing as T

import numpy as np
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_NNEF_STR
from torch_to_nnef.torch_graph import (
    Data,
    ListWithTensor,
    PythonConstant,
    TensorVariable,
)

LOGGER = logging.getLogger(__name__)


def add_nnef_operation(graph, inputs, *args, **kwargs):
    if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
        inputs = maybe_unsqueeze_to_consistant_inputs_ranks(graph, inputs)
    kwargs["graph"] = graph
    kwargs["inputs"] = inputs
    return NOperation(*args, **kwargs)


def add_tensor_variable_node_as_nnef_tensor(
    g: NGraph,
    node: TensorVariable,
    name_to_tensor: T.Dict[str, NTensor],
    name_suffix: str = "",
):
    """Create NNEF tensor and register in graph from torch_graph.Data node

    It automatically adds variable if node is a torch tensor is associated
    (it avoids bloating nnef graph file with matrix values)

    """
    name = node.export_name
    if name_suffix:
        name += f"_{name_suffix}"

    nnef_tensor_ref = NTensor(
        g,
        name,
        dtype=node.np_dtype,
        shape=node.shape,
    )
    if node.data is not None:
        nnef_tensor_ref.data = node.data.detach().numpy()
        nnef_tensor_ref.shape = tuple(node.data.shape)
        add_nnef_operation(
            graph=g,
            type="variable",
            inputs=None,
            outputs=nnef_tensor_ref,
            attribs={
                "label": nnef_tensor_ref.name,
                "shape": list(nnef_tensor_ref.shape),
                "dtype": nnef_tensor_ref.dtype,
            },
        )

    name_to_tensor[name] = nnef_tensor_ref
    return nnef_tensor_ref


def maybe_unsqueeze_to_consistant_inputs_ranks(g, nnef_tensors):
    """May unsqueeze at 0 rank n time to ensure consistant rank between inputs

    This is done at export time and not inference time because:
    inference implementation may use 1 dim expansion from left to right
    like Tract or Tensorflow
    instead of Pytorch expansion which happen in opposite direction.

    """
    tensors_ranks = [len(_.shape) for _ in nnef_tensors]
    if len(set(tensors_ranks)) > 1:
        reference_rank = max(tensors_ranks)
        new_nnef_tensors = []
        for nnef_tensor in nnef_tensors:
            original_rank = len(nnef_tensor.shape)
            missing_dims = reference_rank - original_rank
            if missing_dims > 0 and (
                nnef_tensor.data is None or nnef_tensor.data.size != 1
            ):
                new_shape = list(nnef_tensor.shape)
                new_shape = ([0] * missing_dims) + new_shape
                unsqueeze_axes = [0] * missing_dims

                output_nnef_tensor = NTensor(
                    g,
                    name=f"{nnef_tensor.name}_expanded",
                    dtype=nnef_tensor.dtype,
                    shape=tuple(new_shape),
                )
                NOperation(
                    g,
                    type="unsqueeze",
                    attribs={"axes": unsqueeze_axes},
                    inputs=nnef_tensor,
                    outputs=output_nnef_tensor,
                )
                nnef_tensor = output_nnef_tensor
            new_nnef_tensors.append(nnef_tensor)
        nnef_tensors = tuple(new_nnef_tensors)
    return nnef_tensors


def get_or_add_tensor_variable_in_nnef(
    g, node, name_to_tensor, name_suffix: str = ""
):
    if node.export_name not in name_to_tensor:
        add_tensor_variable_node_as_nnef_tensor(
            g, node, name_to_tensor, name_suffix
        )
    return name_to_tensor[node.export_name]


def external(
    g: NGraph, node: TensorVariable, name_to_tensor: T.Dict[str, NTensor]
):
    """Add External NNEF Operation in graph"""
    nnef_tensor_ref = add_tensor_variable_node_as_nnef_tensor(
        g, node, name_to_tensor
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


def _add_single_output_op(
    g, node, name_to_tensor, nnef_op_type, inputs, attrs=None, ensure_tuple=True
):
    out = add_tensor_variable_node_as_nnef_tensor(
        g, node.outputs[0], name_to_tensor
    )
    if isinstance(inputs, list) and ensure_tuple:
        inputs = tuple(inputs)
    add_nnef_operation(
        graph=g,
        type=nnef_op_type,
        name=f"{node.outputs[0].export_name}_op",
        inputs=inputs,
        outputs=tuple([out]),
        attribs=attrs or {},
    )
    return out


def _unary_output_op_without_params(
    nnef_op_type: str, g, node, name_to_tensor, null_ref
):
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=nnef_op_type,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            if _
            else null_ref
            for _ in node.inputs
        ],
    )


def _unary_input_output_op_with_constant(nnef_op_type, torch_graph, **kwargs):

    g = kwargs["g"]
    node = kwargs["node"]
    name_to_tensor = kwargs["name_to_tensor"]

    for const in node.inputs[1:]:
        if isinstance(const, PythonConstant):
            data = np.array(const.data)
        else:
            data = const.data.numpy()
        nptype = data.dtype.type

        name_to_tensor[const.export_name] = NTensor(
            g,
            const.export_name,
            data=data,
            dtype=nptype,
            shape=data.shape,
        )
    return _unary_output_op_without_params(nnef_op_type, **kwargs)


def _weight_bias_and_output_tensor(
    g,
    node,
    weight_node,
    bias_node,
    name_to_tensor,
    null_ref,
):
    weight_ref = add_tensor_variable_node_as_nnef_tensor(
        node=weight_node,
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix="weight",
    )

    bias_ref = null_ref
    if bias_node.data is not None:
        bias_ref = add_tensor_variable_node_as_nnef_tensor(
            node=bias_node,
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="bias",
        )

    out_node = node.outputs[0]
    out_tensor_name = out_node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=weight_ref.dtype,
        shape=tuple(out_node.shape) if out_node.shape else None,
    )
    name_to_tensor[out_tensor_name] = output_tensor
    return weight_ref, bias_ref, output_tensor


def softmax(**kwargs):
    node = kwargs["node"]
    if node.inputs[2]:
        del node.inputs[2]
    return _unary_input_output_op_with_constant("softmax", **kwargs)


def softplus(torch_graph, **kwargs):
    """
    Note: numerical stability applied in Pytorch is not done in NNEF vanilla
    implementation, nor case beta != 1.

    Pytorch ref:
        y = (1/beta) * log(exp(beta * x) + 1)  if ((beta * x) < thresh) else x
    NNEF ref:
        y = log(exp(x) + 1.0)

    """
    node = kwargs["node"]
    const = node.inputs[1]
    if const.data != 1:
        raise NotImplementedError(
            "This version is not implemented and"
            " would need use of a specific fragment"
        )
    node.inputs = node.inputs[:1]
    return _unary_output_op_without_params("softplus", **kwargs)


def elu(**kwargs):
    node = kwargs["node"]
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("elu", **kwargs)


def leaky_relu(**kwargs):
    node = kwargs["node"]
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("leaky_relu", **kwargs)


def prelu(**kwargs):
    node = kwargs["node"]
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("prelu", **kwargs)


def selu(**kwargs):
    _unary_input_output_op_with_constant("selu", **kwargs)
    return ["selu"]


def silu(**kwargs):
    _unary_input_output_op_with_constant("silu", **kwargs)
    return ["silu"]


def gelu(g, node, name_to_tensor, null_ref, **kwargs):
    _unary_output_op_without_params(
        "gelu",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )
    return ["gelu"]


def hardtanh(**kwargs):
    node = kwargs["node"]
    node.inputs = node.inputs[:2]  # remove inplace param
    _unary_input_output_op_with_constant("hard_tanh", **kwargs)
    return ["hard_tanh"]


def log_softmax(**kwargs):
    node = kwargs["node"]
    if node.inputs[2]:
        del node.inputs[2]
    _unary_input_output_op_with_constant("log_softmax", **kwargs)
    return ["log_softmax"]


def slice_(g, node, name_to_tensor, **kwargs):
    input_node, dim_node, begin_node, end_node, stride_node = node.inputs

    # we assert for now all node except first are all constant

    dim = dim_node.data

    # we use this since by default pytorch generate max int32 value for end
    end = min(end_node.data, input_node.shape[dim])

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "axes": [dim],
            "begin": [begin_node.data],
            "end": [end],
            "stride": [stride_node.data],
        },
    )


def _convolution(g, node, name_to_tensor, null_ref, torch_graph):
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
    if bias_node.data is not None:
        params_nodes.append(bias_node)
    for param_node in params_nodes:
        for _ in range(input_node.rank - param_node.rank):
            param_node.data = param_node.data.unsqueeze(0)
            param_node.shape = list(param_node.data.shape)
    # }

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    add_nnef_operation(
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
    )


def _pooling_op(
    nnef_op_name: str,
    node_inputs: T.List[Data],
    g,
    node,
    name_to_tensor,
):
    """
    NNEF (avg|max)_pool params (not dimension specific):
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )

    """
    (
        input_node,
        kernel_size_node,
        stride_node,
        padding_node,
        dilation_node,
        ceil_mode_node,
    ) = node_inputs

    if ceil_mode_node.data:
        raise NotImplementedError(
            "Use of ceil to compute output shape is not implem"
        )

    padding = padding_node.data or []
    kernel_size = kernel_size_node.data or []
    stride = stride_node.data or []
    if dilation_node:
        dilation = dilation_node.data or []
    else:
        dilation = [0 for _ in stride]

    # peculiarity of tract implementation
    # apparently tract does expect max_pool to be always 2d only (including
    # input.shape)
    onode = node.outputs[0]
    if onode.rank > len(kernel_size):
        missing_n_dims = onode.rank - len(kernel_size)
        kernel_size = ([1] * missing_n_dims) + kernel_size
        stride = ([1] * missing_n_dims) + stride
        dilation = ([1] * missing_n_dims) + dilation
        padding = padding + ([0] * missing_n_dims)

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_name,
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "size": list(kernel_size),
            "padding": [
                (pad, pad) if isinstance(pad, int) else pad for pad in padding
            ],
            "stride": list(stride),
            "dilation": list(dilation),
            "border": "constant",
        },
    )


def linear(g, node, name_to_tensor, null_ref, **kwargs):
    (
        input_node,
        weight_node,
        bias_node,
    ) = node.inputs

    # expand in stored variable export to avoid adding unsqueeze in graph {
    for _ in range(input_node.rank - weight_node.rank):
        weight_node.data = weight_node.data.unsqueeze(0)
        weight_node.shape = list(weight_node.data.shape)

    if bias_node.data is not None:
        for _ in range(input_node.rank - bias_node.rank):
            bias_node.data = bias_node.data.unsqueeze(0)
            bias_node.shape = list(bias_node.data.shape)
    # }

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    add_nnef_operation(
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
        input_node,
        weight_node,
        bias_node,
        running_mean_node,
        running_var_node,
        _,  # training
        _,  # momentum
        eps_node,
        _,  # cudnn_enabled
    ) = node.inputs

    # expand in stored variables export to avoid unsqueeze guessing in graph {
    params_nodes = [weight_node, running_mean_node, running_var_node]
    if bias_node.data is not None:
        params_nodes.append(bias_node)
    for param_node in params_nodes:
        param_node.data = param_node.data.unsqueeze(0)
        param_node.shape = list(param_node.data.shape)
        for _ in range(input_node.rank - param_node.rank):
            param_node.data = param_node.data.unsqueeze(-1)
            param_node.shape = list(param_node.data.shape)
    # }

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )
    running_mean_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="running_mean",
        node=running_mean_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    running_var_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="running_var",
        node=running_var_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    add_nnef_operation(
        graph=g,
        type="batch_normalization",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            running_mean_ref,
            running_var_ref,
            bias_ref,
            weight_ref,
        ),
        outputs=output_tensor,
        attribs={"epsilon": eps_node.data},
    )


def max_pool1d(g, node, name_to_tensor, **kwargs):
    _pooling_op(
        "max_pool",
        node.inputs,
        g,
        node,
        name_to_tensor,
    )


def avg_pool1d(g, node, name_to_tensor, **kwargs):
    count_include_pad = node.inputs[-1].data
    if not count_include_pad:
        raise NotImplementedError("not implemented count_include_pad=False")
    inputs_name_tuple = node.inputs[:-1]  # count_include_pad excluded
    inputs_name_tuple.insert(4, None)  # set missing dilation
    # Dilation is available
    _pooling_op(
        "avg_pool",
        inputs_name_tuple,
        g,
        node,
        name_to_tensor,
    )


def max_pool2d(g, node, name_to_tensor, **kwargs):
    _pooling_op(
        "max_pool",
        node.inputs,
        g,
        node,
        name_to_tensor,
    )


def avg_pool2d(g, node, name_to_tensor, **kwargs):
    _pooling_op(
        "avg_pool",
        node.inputs,
        g,
        node,
        name_to_tensor,
    )


def _adaptive_pool(
    nnef_op_name: str, g, node, name_to_tensor, null_ref, torch_graph
):
    (
        input_node,
        pool_values_node,
    ) = node.inputs

    pool_values = pool_values_node.data
    if not all(
        dim and dim > 0 for dim in input_node.shape[-len(pool_values) :]
    ):
        raise NotImplementedError(
            "dynamic dim used in adaptive pool is not Implemented yet"
        )
    # fixed at export auto adaptation
    stride = [
        int(in_tensor_dim // pool_val)
        for pool_val, in_tensor_dim in zip(
            pool_values, input_node.shape[-len(pool_values) :]
        )
    ]
    onode = node.outputs[0]
    if onode.rank > len(stride):
        missing_n_dims = onode.rank - len(stride)
        stride = ([1] * missing_n_dims) + stride

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_name,
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "size": list(stride),
            "padding": [(0, 0) for _ in stride],
            "stride": list(stride),
            "dilation": [1 for _ in stride],
            "border": "ignore",
        },
    )


def adaptive_avg_pool2d(g, node, name_to_tensor, null_ref, torch_graph):
    # WARNING will liklely only wor with full defined shapes in shape
    _adaptive_pool("avg_pool", g, node, name_to_tensor, null_ref, torch_graph)


def dropout(node, torch_graph, **kwargs):
    (
        input_node,
        _,  # probability
        is_active_node,
    ) = node.inputs
    # should wire directly input_node to output without intermediate
    if is_active_node.data:
        raise NotImplementedError("dropout active at inference")

    # this replace order is important for graph of single nodes or starting with
    torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


def contiguous(node, torch_graph, **kwargs):
    """This does not translate to any operation"""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


def view(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim_node) = node.inputs

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_node.data},
    )


def flatten(g, node, name_to_tensor, null_ref, torch_graph):
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
    _add_single_output_op(
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


def to(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_node,
        _,  # dtype_name
        _,  # non_blocking_name
        _,  # copy_name
        _,  # memory_format_name
    ) = node.inputs

    onode = node.outputs[0]
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        TORCH_DTYPE_TO_NNEF_STR[onode.dtype],
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            # "dtype": onode.np_dtype,
            "shape": list(onode.shape),
        },
    )


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
            raise NotImplementedError("take a look at pow in nnef spec")
    else:
        op_type = "pow"
        inputs += [
            get_or_add_tensor_variable_in_nnef(g, exponent_node, name_to_tensor)
        ]

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        op_type,
        inputs=inputs,
    )


def quantize_per_tensor(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_node,
        scale_node,
        zero_point_node,
        dtype_node,
    ) = node.inputs
    assert dtype_node.data == 13, "is not expected quint8"
    input_node = node.inputs[0]
    tensor = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    tensor.quant = {
        "zero_point": zero_point_node.data,
        "scale": scale_node.data,
        "bits": 8,
        "signed": False,
        "symmetric": False,
        "op-name": "zero_point_linear_quantize",
    }
    torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)


def dequantize(g, node, name_to_tensor, null_ref, torch_graph):
    """
    We will only handle the case of zero_point affine quantization for now.
    which in reverse of quantization is:

       (x - zero_point) / scale
    """
    input_node = node.inputs[0]
    torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)


def transpose(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim0_node, dim1_node) = node.inputs
    dim0 = dim0_node.data
    dim1 = dim1_node.data

    new_dims_ranks = []
    for _ in range(node.outputs[0].rank):
        if _ == dim0:
            new_dims_ranks.append(dim0)
        elif _ == dim1:
            new_dims_ranks.append(dim1)
        else:
            new_dims_ranks.append(_)

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": new_dims_ranks},
    )


def permute(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dims_node) = node.inputs
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": dims_node.data},
    )


def cat(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim_node) = node.inputs
    dim = dim_node.data
    assert isinstance(input_node, ListWithTensor)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise NotImplementedError(f"cat with input_item: {input_item}")
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_item, name_to_tensor
        )
        inputs.append(tensor_ref)
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "concat",
        inputs=inputs,
        attrs={"axis": dim},
        ensure_tuple=False,
    )


def stack(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim_node) = node.inputs
    dim = dim_node.data
    assert isinstance(input_node, ListWithTensor)
    inputs = []
    for input_item in input_node.data:
        if (
            input_item.export_name not in name_to_tensor
            and input_item.data is None
        ):
            torch_graph.printall()
            raise NotImplementedError(f"stack with input_item: {input_item}")
        tensor_ref = get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        )
        inputs.append(tensor_ref)
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "stack",
        inputs=inputs,
        attrs={"axis": dim},
        ensure_tuple=False,
    )


def split(g, node, name_to_tensor, null_ref, torch_graph):
    raise NotImplementedError("split not implemented")


def unsqueeze(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim_node) = node.inputs

    dim = dim_node.data
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [dim]},
    )


def squeeze(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim_node) = node.inputs
    dim = dim_node.data
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [dim]},
    )


def _reducer(
    aten_op_name: str, g, node, name_to_tensor, torch_graph, output_idx: int = 0
):

    (input_node, dim_node, keep_dim_node) = node.inputs

    keep_dim = keep_dim_node.data

    onode = node.outputs[output_idx]
    out = add_tensor_variable_node_as_nnef_tensor(g, onode, name_to_tensor)
    op_reduce_out = None
    if not keep_dim:
        # apply squeeze
        op_reduce_out_name = f"{onode.export_name}_{aten_op_name}"
        op_reduce_out = NTensor(
            g,
            op_reduce_out_name,
            dtype=onode.np_dtype,
            shape=onode.shape,
        )
        name_to_tensor[op_reduce_out_name] = op_reduce_out
    add_nnef_operation(
        graph=g,
        type=aten_op_name,
        name=f"{onode.export_name}_{aten_op_name}",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        outputs=out if keep_dim else op_reduce_out,
        attribs={"axes": dim_node.data},
    )
    if not keep_dim:
        add_nnef_operation(
            graph=g,
            type="squeeze",
            name=f"{onode.export_name}_squeeze",
            inputs=op_reduce_out,
            outputs=out,
            attribs={"axes": dim_node.data},
        )


def mean(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("mean_reduce", g, node, name_to_tensor, torch_graph)


def reduce_sum(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("sum_reduce", g, node, name_to_tensor, torch_graph)


def argmax(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("argmax_reduce", g, node, name_to_tensor, torch_graph)


def argmin(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("argmin_reduce", g, node, name_to_tensor, torch_graph)


def reduce_any(g, node, name_to_tensor, null_ref, torch_graph):
    assert len(node.outputs) == 1
    _reducer("any_reduce", g, node, name_to_tensor, torch_graph)


def reduce_all(g, node, name_to_tensor, null_ref, torch_graph):
    assert len(node.outputs) == 1
    _reducer("all_reduce", g, node, name_to_tensor, torch_graph)


def reduce_max(g, node, name_to_tensor, null_ref, torch_graph):
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise NotImplementedError(
            f"unknown 'max' variant with {n_outputs} outputs used"
        )
    _reducer("max_reduce", g, node, name_to_tensor, torch_graph)
    if n_outputs == 2:
        _reducer(
            "argmax_reduce", g, node, name_to_tensor, torch_graph, output_idx=1
        )


def reduce_min(g, node, name_to_tensor, null_ref, torch_graph):
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise NotImplementedError(
            f"unknown 'min' variant with {n_outputs} outputs used"
        )
    _reducer("min_reduce", g, node, name_to_tensor, torch_graph)
    if n_outputs == 2:
        _reducer(
            "argmin_reduce", g, node, name_to_tensor, torch_graph, output_idx=1
        )


def max_(g, node, name_to_tensor, null_ref, torch_graph):
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_max(g, node, name_to_tensor, null_ref, torch_graph)
    return _unary_output_op_without_params(
        nnef_op_type="max",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


def min_(g, node, name_to_tensor, null_ref, torch_graph):
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_min(g, node, name_to_tensor, null_ref, torch_graph)
    return _unary_output_op_without_params(
        nnef_op_type="min",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


def repeat(g, node, name_to_tensor, **kwargs):
    (input_node, dim_node) = node.inputs
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"repeats": dim_node.data},
    )


def size(g, node, name_to_tensor, null_ref, torch_graph):
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

    """
    original_variable_output = node.outputs[0]
    new_node = PythonConstant(
        name=original_variable_output.name,
        data=original_variable_output.data.numpy().tolist(),
    )
    torch_graph.remap_node(original_variable_output, new_node)

    for data_node in torch_graph.data_nodes:
        if (
            isinstance(data_node, ListWithTensor)
            and any(_ is new_node for _ in data_node.data)
            and all(isinstance(_, PythonConstant) for _ in data_node.data)
        ):
            # recompute fixed data based on new infos
            torch_graph.remap_node(
                data_node,
                PythonConstant(
                    name=data_node.name, data=[_.data for _ in data_node.data]
                ),
            )
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]

    LOGGER.warning(
        "the aten::size need custom NNEF operator from tract internals. "
        " For now we fix values at export time"
    )


def reshape(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, dim_node) = node.inputs
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_node.data},
    )


def reflection_padnd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, pads_node) = node.inputs
    pads = (
        np.array(pads_node.data).reshape(-1, 2).tolist()[::-1]
    )  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"padding": pads, "border": "reflect"},
    )


def replication_padnd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, pads_node) = node.inputs
    pads = pads_node.data
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"padding": pads, "border": "replicate"},
    )


def constant_pad_nd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, pads_node, value_node) = node.inputs
    value = value_node.data
    pads = pads_node.data
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < onode.rank:
        pads = [[0, 0]] * (onode.rank - len(pads)) + pads
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"padding": pads, "value": value},
    )


def where(g, node, name_to_tensor, null_ref, torch_graph):
    (condition_node, true_value_node, false_value_node) = node.inputs

    inputs = []
    for snode in [condition_node, true_value_node, false_value_node]:
        name = snode.export_name
        if name in name_to_tensor:
            inputs.append(name_to_tensor[name])
        else:
            snode_ref = add_tensor_variable_node_as_nnef_tensor(
                name_suffix=name,
                node=snode,
                g=g,
                name_to_tensor=name_to_tensor,
            )
            inputs.append(snode_ref)

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="select",
        inputs=inputs,
    )


def matmul(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_node,
        other_node,
    ) = node.inputs

    _add_single_output_op(
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


def aten_to_nnef_tensor_and_ops(g, node, name_to_tensor, null_ref, torch_graph):
    """Main primitive dispatcher

    Allow to write in graph any not Quantized Operation from pytorch defined in
    node attribute.

    """
    aten_op_name = node.kind.split("::")[1]

    # remap
    if aten_op_name.endswith("_"):
        aten_op_name = aten_op_name[:-1]
    aten_op_name = {
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
        "slice": "slice_"
        # }
    }.get(aten_op_name, aten_op_name)

    if aten_op_name in [
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
        "abs",
        "sign",
        "neg",
        "floor",
        "ceil",
        "round",
        "sqrt",
        "rsqrt",
        "log2",
        "copy",
        "rcp",
        "not",
        "eq",
        "ne",
        "add",
        "sub",
        "mul",
        "div",
        "lt",
        "gt",
        "le",
        "ge",
        "and",
        "or",
    ]:
        return _unary_output_op_without_params(
            nnef_op_type=aten_op_name,
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )

    return globals()[aten_op_name](
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
    )
