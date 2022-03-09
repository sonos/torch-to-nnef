import typing as T

from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor
import torch
import numpy as np

from torch_to_nnef.dtypes import STR_TO_NUMPY_DTYPE
from torch_to_nnef.torch_graph import NodeConstant


def add_tensor_to_ngraph(
    g,
    node,
    tensor: torch.Tensor,
    tensor_name: str,
    name_to_tensor: T.Dict[str, NTensor],
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
    name_to_tensor[name] = ntensor
    return ntensor


def _unary_output_op_without_params(
    nnef_op_type: str, g, node, name_to_tensor, null_ref
):
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
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


def _unary_input_output_op_with_constant(nnef_op_type, torch_graph, **kwargs):

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
            nptype = STR_TO_NUMPY_DTYPE[const.subtype or const.dtype]
            data = np.array(const.value, dtype=nptype)
        name_to_tensor[const_node_name] = NTensor(
            g,
            const.export_name,
            data=data,
            dtype=nptype,
            shape=const.tensor_size,
        )
    return _unary_output_op_without_params(nnef_op_type, **kwargs)


def _register_state_node_as_variable(
    node_export_name: str, slug_name: str, torch_graph, node, g, name_to_tensor
):
    torch_tensor = torch_graph.get_node_by_export_name(node_export_name).data

    # peculiarity of tract implementation
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


def softmax(**kwargs):
    node = kwargs['node']
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
    node = kwargs['node']
    const = torch_graph.get_node_by_export_name(node.export_inputs[1])
    if const.value != 1:
        raise NotImplemented(
            "This version is not implemented and"
            " would need use of a specific fragment"
        )
    node.inputs = node.inputs[:1]
    return _unary_output_op_without_params("softplus", **kwargs)


def elu(**kwargs):
    node = kwargs['node']
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("elu", **kwargs)


def leaky_relu(**kwargs):
    node = kwargs['node']
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("leaky_relu", **kwargs)


def prelu(**kwargs):
    node = kwargs['node']
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


def _convolution(g, node, name_to_tensor, null_ref, torch_graph):
    # tuple of ints dilation, bool transposed, tuple of ints output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled
    (
        input_name,
        weight_name,
        bias_name,
        stride_name,
        padding_name,
        dilation_name,
        transposed_name,
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
    transposed = torch_graph.get_node_by_export_name(transposed_name).value

    wnode = torch_graph.get_node_by_export_name(weight_name)
    if transposed:
        wnode.data = wnode.data.transpose(1, 0)
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
        type="deconv" if transposed else "conv",
        name=f"{node.export_name}_op",
        inputs=(
            name_to_tensor[input_name],
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
    inputs_name_tuple,
    g,
    node,
    name_to_tensor,
    torch_graph,
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
        input_name,
        kernel_size_name,
        stride_name,
        padding_name,
        dilation_name,
        ceil_mode_name,
    ) = inputs_name_tuple

    def get_array_values_from_inputs(node_export_name: str):
        return [
            torch_graph.get_node_by_export_name(in_export_name).value
            for in_export_name in torch_graph.get_node_by_export_name(
                node_export_name
            ).export_inputs
        ]

    ceil_mode = torch_graph.get_node_by_export_name(ceil_mode_name).value
    if ceil_mode:
        raise NotImplementedError(
            "Use of ceil to compute output shape is not implem"
        )

    out_tensor_name = node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        # dtype=?.numpy().dtype.type,
        shape=tuple(node.tensor_size) if node.tensor_size else None,
    )
    name_to_tensor[out_tensor_name] = output_tensor

    padding = get_array_values_from_inputs(padding_name)
    kernel_size = get_array_values_from_inputs(kernel_size_name)
    stride = get_array_values_from_inputs(stride_name)
    if dilation_name:
        dilation = get_array_values_from_inputs(dilation_name)
    else:
        dilation = [0 for _ in stride]

    # peculiarity of tract implementation
    # not sure what to do need discussion with @kali
    # apparently tract does expect max_pool to be always 2d only (including
    # input.shape)

    # To handle this on our side we should
    if len(node.tensor_size) > len(kernel_size):
        missing_n_dims = len(node.tensor_size) - len(kernel_size)
        kernel_size = ([1] * missing_n_dims) + kernel_size
        stride = ([1] * missing_n_dims) + stride
        dilation = ([1] * missing_n_dims) + dilation
        padding = padding + ([0] * missing_n_dims)
    # kernel_size = [1, 1] + kernel_size + [1]
    # but also 'unsqueeze' input by 1 and 'squeeze' it back

    NOperation(
        graph=g,
        type=nnef_op_name,
        name=f"{node.export_name}_op",
        inputs=name_to_tensor[input_name],
        outputs=output_tensor,
        attribs={
            "size": list(kernel_size),
            "padding": [
                (pad, pad) if isinstance(pad, int) else pad for pad in padding
            ],
            "stride": list(stride),
            "dilation": list(dilation),
            "border": "constant",
        },
    )


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


def max_pool1d(g, node, name_to_tensor, null_ref, torch_graph):
    _pooling_op(
        "max_pool",
        node.export_inputs,
        g,
        node,
        name_to_tensor,
        torch_graph,
    )


def avg_pool1d(g, node, name_to_tensor, null_ref, torch_graph):
    count_include_pad = torch_graph.get_node_by_export_name(
        node.export_inputs[-1]
    ).value

    if not count_include_pad:
        raise NotImplementedError("not implemented count_include_pad=False")
    inputs_name_tuple = node.export_inputs[:-1]  # count_include_pad excluded
    inputs_name_tuple.insert(4, None)  # set missing dilation
    # Dilation is available
    _pooling_op(
        "avg_pool",
        inputs_name_tuple,
        g,
        node,
        name_to_tensor,
        torch_graph,
    )


def max_pool2d(g, node, name_to_tensor, null_ref, torch_graph):
    _pooling_op(
        "max_pool",
        node.export_inputs,
        g,
        node,
        name_to_tensor,
        torch_graph,
    )


def avg_pool2d(g, node, name_to_tensor, null_ref, torch_graph):
    _pooling_op(
        "avg_pool",
        node.export_inputs,
        g,
        node,
        name_to_tensor,
        torch_graph,
    )


def _adaptive_pool(
    nnef_op_name: str, g, node, name_to_tensor, null_ref, torch_graph
):
    (
        input_name,
        pool_values_name,
    ) = node.export_inputs
    input_node = torch_graph.get_node_by_export_name(input_name)

    def get_array_values_from_inputs(node_export_name: str):
        return [
            torch_graph.get_node_by_export_name(in_export_name).value
            for in_export_name in torch_graph.get_node_by_export_name(
                node_export_name
            ).export_inputs
        ]

    out_tensor_name = node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        shape=tuple(node.tensor_size) if node.tensor_size else None,
    )
    name_to_tensor[out_tensor_name] = output_tensor

    pool_values = get_array_values_from_inputs(pool_values_name)
    if not all(
        dim and dim > 0 for dim in input_node.tensor_size[-len(pool_values) :]
    ):
        raise NotImplementedError(
            "dynamic dim used in adaptive pool is not Implemented yet"
        )
    # fixed at export auto adaptation
    stride = [
        int(in_tensor_dim // pool_val)
        for pool_val, in_tensor_dim in zip(
            pool_values, input_node.tensor_size[-len(pool_values) :]
        )
    ]
    if len(node.tensor_size) > len(stride):
        missing_n_dims = len(node.tensor_size) - len(stride)
        stride = ([1] * missing_n_dims) + stride
    NOperation(
        graph=g,
        type=nnef_op_name,
        name=f"{node.export_name}_op",
        inputs=name_to_tensor[input_name],
        outputs=output_tensor,
        attribs={
            "size": list(stride),
            "padding": [(0, 0) for _ in stride],
            "stride": list(stride),
            "dilation": [1 for _ in stride],
            "border": "ignore",
        },
    )


def adaptive_avg_pool2d(g, node, name_to_tensor, null_ref, torch_graph):
    """
    RoI pooling generates a fixed size output by pooling regions of variable size.

    fragment avg_roi_pool(
        input: tensor<scalar>,              # the feature maps to pool from
        rois: tensor<scalar>,               # the regions of interest
        batch_index: tensor<integer>,       # batch indices for each RoI
        output_size: integer[] )            # the desired output size
    -> ( output: tensor<scalar> )

    # pool_values = get_array_values_from_inputs(pool_values_name)
    torch_graph.printall()
    input_node = torch_graph.get_node_by_export_name(input_name)

    rois_ref = add_tensor_to_ngraph(
        g,
        node,
        tensor=torch.from_numpy(
            np.array([[0, dim] for dim in input_node.tensor_size]).flatten()
        ),
        tensor_name="rois",
        name_to_tensor=name_to_tensor,
    )
    batch_index_ref = add_tensor_to_ngraph(
        g,
        node,
        tensor=torch.from_numpy(np.arange(input_node.tensor_size[0])),
        tensor_name="batch_index",
        name_to_tensor=name_to_tensor,
    )

    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE(node.subtype or node.dtype),
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out

    outputs = [out]
    NOperation(
        graph=g,
        type="avg_roi_pool",
        name=f"{node.export_name}_op",
        inputs=[name_to_tensor[input_name], rois_ref, batch_index_ref],
        outputs=tuple(outputs),
        attribs={"output_size": node.tensor_size},
    )

    """

    # WARNING will liklely only wor with full defined shapes in tensor_size
    _adaptive_pool("avg_pool", g, node, name_to_tensor, null_ref, torch_graph)


def dropout(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_name,
        _,  # probability
        is_active_name,
    ) = node.export_inputs
    is_active = torch_graph.get_node_by_export_name(is_active_name).value

    # should wire directly input_node to output without intermediate
    if is_active:
        raise NotImplementedError("dropout active at inference")

    # this replace order is important for graph of single nodes or starting with
    # dropout
    torch_graph.rename_node_and_graph_ref(
        original_debug_name=node.debugName, new_debug_name=node.inputs[0]
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
    (input_name, _, _) = node.export_inputs  # start_dim_name  # end_dim_name

    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out

    outputs = [out]
    if input_name not in name_to_tensor:
        import ipdb

        ipdb.set_trace()
        assert False
    NOperation(
        graph=g,
        type="reshape",
        name=f"{node.export_name}_op",
        inputs=name_to_tensor[input_name],
        outputs=tuple(outputs),
        attribs={
            "dtype": out.dtype,
            "shape": list(node.tensor_size),
            "axis_start": 0,
            "axis_count": -1,
        },
    )


def to(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_name,
        _,  # dtype_name
        _,  # non_blocking_name
        _,  # copy_name
        _,  # memory_format_name
    ) = node.export_inputs

    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out

    outputs = [out]
    NOperation(
        graph=g,
        type="cast",
        name=f"{node.export_name}_op",
        inputs=name_to_tensor[input_name],
        outputs=tuple(outputs),
        attribs={
            "dtype": out.dtype,
            "shape": list(node.tensor_size),
        },
    )


def pow(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, exponent_name) = node.export_inputs
    exponent_node = torch_graph.get_node_by_export_name(exponent_name)
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out

    outputs = [out]
    inputs = [name_to_tensor[input_name]]
    if hasattr(exponent_node, "value"):
        exponent = exponent_node.value
        if exponent == 2:
            op_type = "sqr"
        elif exponent == -2:
            op_type = "rsqr"
        else:
            raise NotImplementedError("take a look at pow in nnef spec")
    else:
        op_type = "pow"
        inputs += [name_to_tensor[exponent_name]]

    NOperation(
        graph=g,
        type=op_type,
        name=f"{node.export_name}_op",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        attribs={
            "dtype": out.dtype,
            "shape": list(node.tensor_size),
        },
    )


def quantize_per_tensor(g, node, name_to_tensor, null_ref, torch_graph):
    (
        input_name,
        scale_name,
        zero_point_name,
        _,  # dtype_name
    ) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    scale = torch_graph.get_node_by_export_name(scale_name).value
    zero_point = torch_graph.get_node_by_export_name(zero_point_name).value
    # dtype = torch_graph.get_node_by_export_name(dtype_name).value
    NOperation(
        graph=g,
        type="zero_point_linear_quantize",
        name=f"{node.export_name}_op",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={
            # "dtype": out.dtype,
            # "shape": list(node.tensor_size),
            "zero_point": zero_point,
            "scale": scale,
            "bits": 8,
            "signed": True,  # dtype != torch.quint8,
            "symmetric": False,
        },
    )


def dequantize(g, node, name_to_tensor, null_ref, torch_graph):
    """
    We will only handle the case of zero_point affine quantization for now.
    which in reverse of quantization is:

       (x - zero_point) / scale
    """
    input_name = node.export_inputs[0]
    input_node = torch_graph.get_node_by_export_name(input_name)
    if "linear_quant" not in input_node.attributes:
        raise NotImplementedError("need to propagate linear_quant in attr.")
    scale = np.array(input_node.attributes['linear_quant']['scale'])
    zero_point = np.array(input_node.attributes['linear_quant']['zero_point'])

    cast_name = f"{node.export_name}_cast"
    out_cast = NTensor(
        g,
        cast_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[cast_name] = out_cast

    NOperation(
        graph=g,
        type="cast",
        name=f"{node.export_name}_dequantize",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out_cast]),
        attribs={"dtype": np.float32},
    )

    sub_name = f"{node.export_name}_sub"
    out_sub = NTensor(
        g,
        sub_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[sub_name] = out_sub

    NOperation(
        graph=g,
        type="sub",
        name=f"{node.export_name}_dequantize",
        inputs=tuple(
            [
                name_to_tensor[input_name],
                NTensor(
                    g,
                    sub_name + "_zero_point",
                    dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
                    data=zero_point,
                ),
            ]
        ),
        outputs=tuple([out_sub]),
    )

    div_name = f"{node.export_name}_div"
    out_div = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out_div

    NOperation(
        graph=g,
        type="div",
        name=f"{node.export_name}",
        inputs=tuple(
            [
                out_sub,
                NTensor(
                    g,
                    div_name + "_scale",
                    dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
                    data=scale,
                ),
            ]
        ),
        outputs=tuple([out_div]),
    )


def transpose(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dim0_name, dim1_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    dim0 = torch_graph.get_node_by_export_name(dim0_name).value
    dim1 = torch_graph.get_node_by_export_name(dim1_name).value

    new_dims_ranks = []
    for _ in range(len(node.tensor_size)):
        if _ == dim0:
            new_dims_ranks.append(dim1)
        elif _ == dim1:
            new_dims_ranks.append(dim0)
        else:
            new_dims_ranks.append(_)

    NOperation(
        graph=g,
        type="transpose",
        name=f"{node.export_name}",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"axes": new_dims_ranks},
    )


def permute(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dims_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    dims = torch_graph.get_node_by_export_name(dims_name).attributes['values']
    NOperation(
        graph=g,
        type="transpose",
        name=f"{node.export_name}",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"axes": dims},
    )


def cat(g, node, name_to_tensor, null_ref, torch_graph):
    raise NotImplementedError("cat not implemented")


def split(g, node, name_to_tensor, null_ref, torch_graph):
    raise NotImplementedError("split not implemented")


def unsqueeze(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dim_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    dim = torch_graph.get_node_by_export_name(dim_name).value
    NOperation(
        graph=g,
        type="unsqueeze",
        name=f"{node.export_name}_unsqueeze",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"axes": [dim]},
    )


def squeeze(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dim_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    dim = torch_graph.get_node_by_export_name(dim_name).value
    NOperation(
        graph=g,
        type="squeeze",
        name=f"{node.export_name}_squeeze",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"axes": [dim]},
    )


def _reducer(aten_op_name: str, g, node, name_to_tensor, torch_graph):
    def get_array_values_from_inputs(node_export_name: str):
        ref_node = torch_graph.get_node_by_export_name(node_export_name)
        if ref_node.kind != "prim::ListConstruct":
            return [ref_node.value]
        return [
            torch_graph.get_node_by_export_name(in_export_name).value
            for in_export_name in ref_node.export_inputs
        ]

    (input_name, dim_name, keep_dim_name) = node.export_inputs

    keep_dim = torch_graph.get_node_by_export_name(keep_dim_name).value

    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    op_reduce_out = None
    if not keep_dim:
        # apply squeeze
        op_reduce_out_name = f"{node.export_name}_{aten_op_name}"
        op_reduce_out = NTensor(
            g,
            op_reduce_out_name,
            dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
            shape=node.tensor_size,
        )
        name_to_tensor[op_reduce_out_name] = op_reduce_out
    dims = get_array_values_from_inputs(dim_name)
    NOperation(
        graph=g,
        type=aten_op_name,
        name=f"{node.export_name}_{aten_op_name}",
        inputs=name_to_tensor[input_name],
        outputs=out if keep_dim else op_reduce_out,
        attribs={"axes": dims},
    )
    if not keep_dim:
        NOperation(
            graph=g,
            type="squeeze",
            name=f"{node.export_name}_squeeze",
            inputs=op_reduce_out,
            outputs=out,
            attribs={"axes": dims},
        )


def mean(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("mean_reduce", g, node, name_to_tensor, torch_graph)


def sum(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("sum_reduce", g, node, name_to_tensor, torch_graph)


def argmax(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("argmax_reduce", g, node, name_to_tensor, torch_graph)


def argmin(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("argmin_reduce", g, node, name_to_tensor, torch_graph)


def reduce_any(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("any_reduce", g, node, name_to_tensor, torch_graph)


def reduce_all(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("all_reduce", g, node, name_to_tensor, torch_graph)


def repeat(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dim_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    repeat_dims = torch_graph.get_node_by_export_name(dim_name).attributes[
        'values'
    ]
    NOperation(
        graph=g,
        type="tile",
        name=f"{node.export_name}_tile",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"repeats": repeat_dims},
    )


def reshape(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dim_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    reshape_dims = torch_graph.get_node_by_export_name(dim_name).attributes[
        'values'
    ]
    NOperation(
        graph=g,
        type="reshape",
        name=f"{node.export_name}_reshape",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"shape": reshape_dims},
    )


def reflection_padnd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, pads_name) = node.export_inputs
    pads = torch_graph.get_node_by_export_name(pads_name).attributes['values']
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    if len(pads) < len(node.tensor_size):
        pads = [[0, 0]] * (len(node.tensor_size) - len(pads)) + pads
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    NOperation(
        graph=g,
        type="pad",
        name=f"{node.export_name}_pad",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"padding": pads, "border": "reflect"},
    )


def replication_padnd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, pads_name) = node.export_inputs
    pads = torch_graph.get_node_by_export_name(pads_name).attributes['values']
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    if len(pads) < len(node.tensor_size):
        pads = [[0, 0]] * (len(node.tensor_size) - len(pads)) + pads
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    NOperation(
        graph=g,
        type="pad",
        name=f"{node.export_name}_pad",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"padding": pads, "border": "replicate"},
    )


def constant_pad_nd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, pads_name, value_name) = node.export_inputs
    value = torch_graph.get_node_by_export_name(value_name).value
    pads = torch_graph.get_node_by_export_name(pads_name).attributes['values']
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    if len(pads) < len(node.tensor_size):
        pads = [[0, 0]] * (len(node.tensor_size) - len(pads)) + pads
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.tensor_size,
    )
    name_to_tensor[node.export_name] = out
    NOperation(
        graph=g,
        type="pad",
        name=f"{node.export_name}_pad",
        inputs=name_to_tensor[input_name],
        outputs=tuple([out]),
        attribs={"padding": pads, "value": value},
    )


def aten_to_nnef_tensor_and_ops(g, node, name_to_tensor, null_ref, torch_graph):
    aten_op_name = node.kind.split("::")[1]

    # remap
    aten_op_name = {
        "add_": "add",
        "_relu": "relu",
        "reciprocal": "rcp",
        "clone": "copy",
        "bitwise_not": "not",
        "bitwise_not_cpu": "not",
        "bitwise_cpu": "and",
        "__and__": "and",
        "__or__": "or",
        "less": 'lt',
        "greater": 'gt',
        "less_equal": 'le',
        "greater_equal": 'ge',
        "any": "reduce_any",  # avoid python builtin collision
        "all": "reduce_all",  # avoid python builtin collision
        "reflection_pad1d": "reflection_padnd",
        "replication_pad1d": "replication_padnd",
        "constant_pad1d": "constant_padnd",
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
        "min",
        "max",
        "not",
        "eq",
        "ne",
        "add",
        "sub",
        "mul",
        "div",
        'lt',
        'gt',
        'le',
        'ge',
        'and',
        'or',
        'matmul',
    ]:
        return _unary_output_op_without_params(
            nnef_op_type=aten_op_name,
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    else:
        return globals()[aten_op_name](
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
            torch_graph=torch_graph,
        )
