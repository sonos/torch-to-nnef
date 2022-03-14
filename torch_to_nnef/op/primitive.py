# pylint: disable=too-many-lines
import typing as T

import numpy as np
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import STR_TO_NUMPY_DTYPE
from torch_to_nnef.torch_graph import Data, ListWithTensor, PythonConstant


def _add_output_tensor(
    g,
    node,
    name_to_tensor,
):
    onode = node.outputs[0]
    out = NTensor(
        g,
        onode.export_name,
        dtype=onode.np_dtype,
        shape=onode.shape,
    )
    name_to_tensor[onode.export_name] = out
    return out


def _add_single_output_op(
    g, node, name_to_tensor, nnef_op_type, inputs, attrs=None, ensure_tuple=True
):
    out = _add_output_tensor(g, node, name_to_tensor)
    if isinstance(inputs, list) and ensure_tuple:
        inputs = tuple(inputs)
    NOperation(
        graph=g,
        type=nnef_op_type,
        name=f"{node.outputs[0].export_name}_op",
        inputs=inputs,
        outputs=tuple([out]),
        attribs=attrs or {},
    )


def add_tensor_to_ngraph(
    g,
    node,
    tensor: torch.Tensor,
    tensor_name: str,
    name_to_tensor: T.Dict[str, NTensor],
):
    onode = node.outputs[0]
    name = f"{onode.export_name}_{tensor_name}"
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
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=nnef_op_type,
        inputs=[
            name_to_tensor[inp.export_name] if inp else null_ref
            for inp in node.inputs
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

    nnef_tensor_ref = add_tensor_to_ngraph(
        g, node, torch_tensor, slug_name, name_to_tensor
    )

    var = NOperation(
        graph=g,
        type="variable",
        name=f"{node.outputs[0].export_name}_{slug_name}_var",
        inputs=None,
        outputs=nnef_tensor_ref,
        attribs={
            "label": nnef_tensor_ref.name,
            "shape": list(torch_tensor.shape),
            "dtype": nnef_tensor_ref.dtype,
        },
    )

    return var.output


def _weight_bias_and_output_tensor(
    g,
    node,
    weight_node,
    bias_node,
    name_to_tensor,
    null_ref,
):
    weight = weight_node.data
    weight_ref = _register_state_node_as_variable(
        torch_tensor=weight,
        slug_name="weight",
        node=node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    bias_ref = null_ref
    if bias_node.data is not None:
        bias_ref = _register_state_node_as_variable(
            torch_tensor=bias_node.data,
            slug_name="bias",
            node=node,
            g=g,
            name_to_tensor=name_to_tensor,
        )

    out_node = node.outputs[0]
    out_tensor_name = out_node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=weight.numpy().dtype.type,
        shape=tuple(out_node.shape) if out_node.shape else None,
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
    const = node.inputs[1]
    if const.data != 1:
        raise NotImplementedError(
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

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    NOperation(
        graph=g,
        type="deconv" if transposed else "conv",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            name_to_tensor[input_node.export_name],
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
    # not sure what to do need discussion with @kali
    # apparently tract does expect max_pool to be always 2d only (including
    # input.shape)

    # To handle this on our side we should
    onode = node.outputs[0]
    if len(onode.shape) > len(kernel_size):
        missing_n_dims = len(onode.shape) - len(kernel_size)
        kernel_size = ([1] * missing_n_dims) + kernel_size
        stride = ([1] * missing_n_dims) + stride
        dilation = ([1] * missing_n_dims) + dilation
        padding = padding + ([0] * missing_n_dims)
    # kernel_size = [1, 1] + kernel_size + [1]
    # but also 'unsqueeze' input by 1 and 'squeeze' it back

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_name,
        inputs=name_to_tensor[input_node.export_name],
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

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    NOperation(
        graph=g,
        type="linear",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(name_to_tensor[input_node.export_name], weight_ref, bias_ref),
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

    weight_ref, bias_ref, output_tensor = _weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )
    running_mean_ref = _register_state_node_as_variable(
        running_mean_node.data,
        slug_name="running_mean",
        node=node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    running_var_ref = _register_state_node_as_variable(
        running_var_node.data,
        slug_name="running_var",
        node=node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    NOperation(
        graph=g,
        type="batch_normalization",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            name_to_tensor[input_node.export_name],
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
    if len(onode.shape) > len(stride):
        missing_n_dims = len(onode.shape) - len(stride)
        stride = ([1] * missing_n_dims) + stride

    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_name,
        inputs=name_to_tensor[input_node.export_name],
        attrs={
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
            np.array([[0, dim] for dim in input_node.shape]).flatten()
        ),
        tensor_name="rois",
        name_to_tensor=name_to_tensor,
    )
    batch_index_ref = add_tensor_to_ngraph(
        g,
        node,
        tensor=torch.from_numpy(np.arange(input_node.shape[0])),
        tensor_name="batch_index",
        name_to_tensor=name_to_tensor,
    )

    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE(node.subtype or node.dtype),
        shape=node.shape,
    )
    name_to_tensor[node.export_name] = out

    outputs = [out]
    NOperation(
        graph=g,
        type="avg_roi_pool",
        name=f"{node.export_name}_op",
        inputs=[name_to_tensor[input_name], rois_ref, batch_index_ref],
        outputs=tuple(outputs),
        attribs={"output_size": node.shape},
    )

    """

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
        inputs=name_to_tensor[input_node.export_name],
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
        "cast",
        inputs=name_to_tensor[input_node.export_name],
        attrs={
            "dtype": onode.np_dtype,
            "shape": list(onode.shape),
        },
    )


def pow_(g, node, name_to_tensor, **kwargs):
    (input_node, exponent_node) = node.inputs
    inputs = [name_to_tensor[input_node.export_name]]
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
        inputs += [name_to_tensor[exponent_node.export_name]]

    onode = node.outputs[0]
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        op_type,
        inputs=inputs,
        attrs={
            "dtype": onode.np_dtype,
            "shape": list(onode.shape),
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
        shape=node.shape,
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
            # "shape": list(node.shape),
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
        shape=node.shape,
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
        shape=node.shape,
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
        shape=node.shape,
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
        shape=node.shape,
    )
    name_to_tensor[node.export_name] = out
    dim0 = torch_graph.get_node_by_export_name(dim0_name).value
    dim1 = torch_graph.get_node_by_export_name(dim1_name).value

    new_dims_ranks = []
    for _ in range(len(node.shape)):
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
        shape=node.shape,
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
    (input_node, dim_node) = node.inputs
    dim = dim_node.data
    assert isinstance(input_node, ListWithTensor)
    inputs = []
    for input_item in input_node.data:
        if input_item.export_name in name_to_tensor:
            tensor_ref = name_to_tensor[input_item.export_name]
        else:
            tensor_ref = _register_state_node_as_variable(
                input_item.data,
                slug_name=input_item.export_name,
                node=node,
                g=g,
                name_to_tensor=name_to_tensor,
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
        inputs=name_to_tensor[input_node.export_name],
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
        inputs=name_to_tensor[input_node.export_name],
        attrs={"axes": [dim]},
    )


def _reducer(aten_op_name: str, g, node, name_to_tensor, torch_graph):

    (input_node, dim_node, keep_dim_node) = node.inputs

    keep_dim = keep_dim_node.data

    onode = node.outputs[0]
    out = _add_output_tensor(g, node, name_to_tensor)
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
    NOperation(
        graph=g,
        type=aten_op_name,
        name=f"{onode.export_name}_{aten_op_name}",
        inputs=name_to_tensor[input_node.export_name],
        outputs=out if keep_dim else op_reduce_out,
        attribs={"axes": dim_node.data},
    )
    if not keep_dim:
        NOperation(
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
    _reducer("any_reduce", g, node, name_to_tensor, torch_graph)


def reduce_all(g, node, name_to_tensor, null_ref, torch_graph):
    _reducer("all_reduce", g, node, name_to_tensor, torch_graph)


def repeat(g, node, name_to_tensor, null_ref, torch_graph):
    (input_name, dim_name) = node.export_inputs
    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.shape,
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
    (input_node, dim_node) = node.inputs
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=name_to_tensor[input_node.export_name],
        attrs={"shape": dim_node.data},
    )


def reflection_padnd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, pads_node) = node.inputs
    pads = (
        np.array(pads_node.data).reshape(-1, 2).tolist()[::-1]
    )  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < len(onode.shape):
        pads = [[0, 0]] * (len(onode.shape) - len(pads)) + pads
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=name_to_tensor[input_node.export_name],
        attrs={"padding": pads, "border": "reflect"},
    )


def replication_padnd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, pads_node) = node.inputs
    pads = pads_node.data
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < len(onode.shape):
        pads = [[0, 0]] * (len(onode.shape) - len(pads)) + pads
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=name_to_tensor[input_node.export_name],
        attrs={"padding": pads, "border": "replicate"},
    )


def constant_pad_nd(g, node, name_to_tensor, null_ref, torch_graph):
    (input_node, pads_node, value_node) = node.inputs
    value = value_node.data
    pads = pads_node.data
    pads = np.array(pads).reshape(-1, 2).tolist()[::-1]  # strangeness of torch
    onode = node.outputs[0]
    if len(pads) < len(onode.shape):
        pads = [[0, 0]] * (len(onode.shape) - len(pads)) + pads
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="pad",
        inputs=name_to_tensor[input_node.export_name],
        attrs={"padding": pads, "value": value},
    )


def where(g, node, name_to_tensor, null_ref, torch_graph):
    (condition_name, true_value_name, false_value_name) = node.export_inputs
    condition_node = torch_graph.get_node_by_export_name(condition_name)
    true_value_node = torch_graph.get_node_by_export_name(true_value_name)
    false_value_node = torch_graph.get_node_by_export_name(false_value_name)

    for snode, name in zip(
        [condition_node, true_value_node, false_value_node],
        [condition_name, true_value_name, false_value_name],
    ):
        if hasattr(snode, "value"):
            data = snode.value.numpy()
            nnef_tensor = NTensor(
                g,
                snode.export_name,
                dtype=data.dtype.type,
                shape=snode.shape,
                data=data,
            )
            var = NOperation(
                graph=g,
                type="variable",
                name=f"{node.export_name}_{condition_name}_var",
                inputs=None,
                outputs=nnef_tensor,
                attribs={
                    "label": nnef_tensor.name,
                    "shape": list(snode.shape),
                    "dtype": nnef_tensor.dtype,
                },
            )
            name_to_tensor[name] = var.output

    out = NTensor(
        g,
        node.export_name,
        dtype=STR_TO_NUMPY_DTYPE[node.subtype or node.dtype],
        shape=node.shape,
    )
    name_to_tensor[node.export_name] = out
    NOperation(
        graph=g,
        type="select",
        name=f"{node.export_name}_select",
        inputs=(
            name_to_tensor[condition_name],
            name_to_tensor[true_value_name],
            name_to_tensor[false_value_name],
        ),
        outputs=tuple([out]),
    )


def aten_to_nnef_tensor_and_ops(g, node, name_to_tensor, null_ref, torch_graph):
    """Main primitive dispatcher

    Allow to write in graph any not Quantized Operation from pytorch defined in
    node attribute.

    """
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
        "reflection_pad1d": "reflection_padnd",
        "replication_pad1d": "replication_padnd",
        "constant_pad1d": "constant_padnd",
        # avoid to ovewrite python builtin {
        "any": "reduce_any",
        "all": "reduce_all",
        "sum": "reduce_sum",
        "pow": "pow_",
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

    return globals()[aten_op_name](
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
    )
