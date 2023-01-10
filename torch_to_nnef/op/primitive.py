# pylint: disable=too-many-lines
import logging
import typing as T

import nnef
import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import (
    SCALAR_TYPE_TO_PYTORCH_TYPE,
    TORCH_DTYPE_TO_TRACT_STR,
    numpy_dtype_to_tract_str,
)
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.torch_graph import (
    Data,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.tract import tract_version_lower_than

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
    "abs",
    "sign",
    "neg",
    "floor",
    "ceil",
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
    "lt",
    "gt",
    "le",
    "ge",
    "and",
    "or",
]


def add_nnef_operation(
    graph, inputs, *args, force_consistent_inputs_shapes: bool = True, **kwargs
):
    if (
        isinstance(inputs, (list, tuple))
        and len(inputs) >= 2
        and force_consistent_inputs_shapes
    ):
        inputs = maybe_unsqueeze_to_consistent_inputs_ranks(graph, inputs)
    kwargs["graph"] = graph
    kwargs["inputs"] = inputs
    return NOperation(*args, **kwargs)


def add_tensor_variable_node_as_nnef_tensor(
    g: NGraph,
    node: TensorVariable,
    name_to_tensor: T.Dict[str, NTensor],
    name_suffix: str = "",
    prevent_variable: bool = False,
    force_full_output_tensor_name: T.Optional[str] = None,
):
    """Create NNEF tensor and register in graph from torch_graph.Data node

    It automatically adds variable if node is a torch tensor is associated
    (it avoids bloating nnef graph file with matrix values)

    """
    if force_full_output_tensor_name:
        name = force_full_output_tensor_name
    else:
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
        if not prevent_variable and len(node.data.size()) > 0:
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


def maybe_unsqueeze_to_consistent_inputs_ranks(g, nnef_tensors):
    """May unsqueeze at 0 rank n time to ensure consistent rank between inputs

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
) -> NTensor:
    name = node.export_name
    if name_suffix:
        name += f"_{name_suffix}"

    if name not in name_to_tensor:
        if isinstance(node, PythonConstant):
            node = node.into_tensor_variable()
        add_tensor_variable_node_as_nnef_tensor(
            g, node, name_to_tensor, name_suffix
        )
    return name_to_tensor[name]


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


def pick_rank(input_node, rank: int) -> int:
    """Enforce that axis, axes ect does contains only positive values"""
    if rank >= 0:
        return rank
    if isinstance(input_node, FixedTensorList):
        base_rank = len(input_node.data)
    else:
        base_rank = input_node.rank
    return base_rank + rank


def pick_value_in_rank(input_node, rank: int, index: int) -> int:
    """Enforce that index in axis does contains only positive values"""
    if index >= 0:
        return index
    return input_node.shape[rank] + index


def fill_negone_with_dim_by_rank_order(
    input_node, shapes: T.List[int]
) -> T.List[int]:
    """Cast each -1 encountered in shapes to incremental rank dim in input_node

    This use case was encountered in pytorch .expand operator

    where by example (picked from MHA in pytorch lib):
        # given v1.shape == (10, 1, 20, 30)
        v1.expand([-1, 1, -1, -1])
        # is equivalent to
        v1.expand([10, 1, 20, 30])

    We need to realise those shape at export since NNEF need concret dim value here
    no symbolics are handled

    """
    new_shapes = []
    for rank_id, s in enumerate(shapes):
        if s == -1:
            new_shapes.append(input_node.shape[rank_id])
        elif isinstance(s, nnef.Identifier) or s > 0:
            new_shapes.append(s)
        else:
            raise TorchToNNEFNotImplementedError("unexpected dim value: ", s)
    return new_shapes


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
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "mul",
        inputs=inputs,
    )


def _add_single_output_op(
    g,
    node,
    name_to_tensor,
    nnef_op_type,
    inputs,
    attrs=None,
    ensure_tuple=True,
    output_tensor_name_suffix: str = "",
    pass_quantization_params: bool = False,
    force_full_output_tensor_name: T.Optional[str] = None,
) -> NTensor:
    assert len(node.outputs) == 1
    out = add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
        name_suffix=output_tensor_name_suffix,
        prevent_variable=True,
        force_full_output_tensor_name=force_full_output_tensor_name,
    )
    if isinstance(inputs, list) and ensure_tuple:
        inputs = tuple(inputs)
    add_nnef_operation(
        graph=g,
        type=nnef_op_type,
        inputs=inputs,
        outputs=tuple([out]),
        attribs=attrs or {},
    )
    if pass_quantization_params:
        input_quants = (
            inputs if isinstance(inputs, NTensor) else inputs[0]
        ).quant
        if input_quants:
            out.quant = input_quants
    return out


def _add_multi_output_op(
    g,
    node,
    name_to_tensor,
    nnef_op_type,
    inputs,
    attrs=None,
    ensure_tuple=True,
    output_tensor_name_suffix: str = "",
):
    if len(node.outputs) == 1:
        LOGGER.debug(
            "Obverved multi to be single output "
            "(which may be normal depending on graph)"
        )
    output_tensors = []
    for out_node in node.outputs:
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            name_suffix=output_tensor_name_suffix,
            prevent_variable=True,
        )
        output_tensors.append(out)

    if isinstance(inputs, list) and ensure_tuple:
        inputs = tuple(inputs)
    add_nnef_operation(
        graph=g,
        type=nnef_op_type,
        inputs=inputs,
        outputs=tuple(output_tensors),
        attribs=attrs or {},
    )
    return output_tensors


def _unary_output_op_without_params(
    nnef_op_type: str, g, node, name_to_tensor, null_ref, **kwargs
):
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=nnef_op_type,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            if _ and not (isinstance(_.data, str) and _.data == "none")
            else null_ref
            for _ in node.inputs
        ],
    )


def _unary_input_output_op_with_constant(nnef_op_type, **kwargs):
    # avoid unpacking then repacking {
    g = kwargs["g"]
    node = kwargs["node"]
    name_to_tensor = kwargs["name_to_tensor"]
    # }
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
    weight_ref = get_or_add_tensor_variable_in_nnef(
        node=weight_node,
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix="weight" if weight_node.data is not None else "",
    )

    bias_ref = null_ref
    if bias_node.data is not None:
        bias_ref = get_or_add_tensor_variable_in_nnef(
            node=bias_node,
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="bias" if bias_node.data is not None else "",
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
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    if node.inputs[2]:
        del node.inputs[2]

    # enforce use of positive rank
    node.inputs[1].data = pick_rank(node.inputs[0], node.inputs[1].data)
    return _unary_input_output_op_with_constant("softmax", **kwargs)


def softplus(**kwargs):
    """
    Note: numerical stability applied in Pytorch is not done in NNEF vanilla
    implementation, nor case beta != 1.

    Pytorch ref:
        y = (1/beta) * log(exp(beta * x) + 1)  if ((beta * x) < thresh) else x
    NNEF ref:
        y = log(exp(x) + 1.0)

    """
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    const = node.inputs[1]
    if const.data != 1:
        raise TorchToNNEFNotImplementedError(
            "This version is not implemented and"
            " would need use of a specific fragment"
        )
    node.inputs = node.inputs[:1]
    return _unary_output_op_without_params("softplus", **kwargs)


def elu(**kwargs):
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("elu", **kwargs)


def leaky_relu(**kwargs):
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    node.inputs = node.inputs[:2]  # remove inplace param
    return _unary_input_output_op_with_constant("leaky_relu", **kwargs)


def prelu(**kwargs):
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
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
    return ["erf", "gelu"]


def erf(g, node, name_to_tensor, null_ref, **kwargs):
    """Op should be added to tract-nnef eventualy"""
    _unary_output_op_without_params(
        "erf",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )
    return ["erf"]


def norm(g, node, name_to_tensor, **kwargs):
    """
    NOTE this is only the normed vector
    """
    input_node, p_node, axes_node, keep_dim_node = node.inputs
    if p_node.data not in [1, 2]:
        raise TorchToNNEFNotImplementedError(
            "norm with p only supported for 1 and 2"
        )

    custom_fragment_name = f"norm_p{p_node.data}"
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        custom_fragment_name,
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, dim) for dim in axes_node.data]},
        output_tensor_name_suffix="_norm" if not keep_dim_node.data else "",
    )
    if not keep_dim_node.data:
        _add_single_output_op(
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=out,
            attrs={
                "axes": [pick_rank(input_node, dim) for dim in axes_node.data]
            },
            pass_quantization_params=True,
        )
    return [custom_fragment_name]


def hardtanh(**kwargs):
    node = kwargs["node"]
    node.inputs = node.inputs[:3]  # remove inplace param
    _unary_input_output_op_with_constant("hard_tanh", **kwargs)
    return ["hard_tanh"]


def log_softmax(**kwargs):
    node = kwargs["node"]
    if node.inputs[2]:
        del node.inputs[2]
    input_node, axis_node = node.inputs
    assert isinstance(axis_node.data, int)
    axis_node.data = pick_rank(input_node, axis_node.data)
    _unary_input_output_op_with_constant("log_softmax", **kwargs)
    return ["log_softmax"]


def round_(nnef_spec_strict, **kwargs):
    if nnef_spec_strict:
        LOGGER.warning(
            "round: Spec definition of round in NNEF does not follow IEEE, "
            "so it will not be exactly same behavior"
        )
        _unary_input_output_op_with_constant("round", **kwargs)
        return []
    _unary_input_output_op_with_constant("tract_core_round_even", **kwargs)
    return ["tract_core"]


def slice_(g, node, name_to_tensor, torch_graph, **kwargs):
    input_node, axis_node, begin_node, end_node, stride_node = node.inputs

    # we assert for now all node except first are all constant
    dim = axis_node.data

    # we use this since by default pytorch generate max int32 value for end
    begin = pick_value_in_rank(input_node, dim, begin_node.data)
    end = min(
        pick_value_in_rank(input_node, dim, end_node.data),
        input_node.shape[dim],
    )
    assert begin < end

    if (
        begin_node.data == 0
        and end == input_node.shape[dim]
        and stride_node.data == 1
    ):
        LOGGER.debug("Slice is not needed since it have not effect")
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "axes": [pick_rank(input_node, dim)],
            "begin": [begin],
            "end": [end],
            "stride": [stride_node.data],
        },
    )


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
        force_consistent_inputs_shapes=False,
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

    if ceil_mode_node and ceil_mode_node.data:
        raise TorchToNNEFNotImplementedError(
            "Use of ceil to compute output shape is not implem"
        )

    padding = padding_node.data or []
    kernel_size = kernel_size_node.data or []
    stride = stride_node.data or []
    if dilation_node:
        dilation = dilation_node.data or []
    else:
        dilation = [1 for _ in stride]

    # peculiarity of tract implementation
    # apparently tract does expect max_pool to be always 2d only (including
    # input.shape)
    onode = node.outputs[0]
    if onode.rank > len(kernel_size):
        missing_n_dims = onode.rank - len(kernel_size)
        kernel_size = ([1] * missing_n_dims) + kernel_size
        stride = ([1] * missing_n_dims) + stride
        dilation = ([1] * missing_n_dims) + dilation

        # pre 0.19.0 padding order differ
        if tract_version_lower_than("0.19.0"):
            padding = padding + ([0] * missing_n_dims)
        else:
            padding = ([0] * missing_n_dims) + padding

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

    if weight_node.data is not None:
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


def batch_norm(g, node, name_to_tensor, null_ref, **kwargs):
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
        raise TorchToNNEFNotImplementedError(
            "not implemented count_include_pad=False"
        )
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
    """
    cpp func parameters:
    (const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override

    _pooling_op expect:

    (input_node,
    kernel_size_node,
    stride_node,
    padding_node,
    dilation_node,
    ceil_mode_node)
    """

    count_include_pad = node.inputs[-2].data
    if not count_include_pad:
        raise TorchToNNEFNotImplementedError(
            "not implemented count_include_pad=False"
        )

    divisor_overide = node.inputs[-1].data
    if divisor_overide:
        raise TorchToNNEFNotImplementedError(
            f"not implemented divisor_override={divisor_overide}"
        )
    inputs_tups = node.inputs[:-2]
    inputs_tups.insert(4, None)
    _pooling_op(
        "avg_pool",
        inputs_tups,
        g,
        node,
        name_to_tensor,
    )


def _adaptive_pool(nnef_op_name: str, g, node, name_to_tensor):
    (
        input_node,
        pool_values_node,
    ) = node.inputs

    pool_values = pool_values_node.data
    if not all(
        dim and dim > 0 for dim in input_node.shape[-len(pool_values) :]
    ):
        raise TorchToNNEFNotImplementedError(
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


def adaptive_avg_pool2d(g, node, name_to_tensor, **kwargs):
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("avg_pool", g, node, name_to_tensor)


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


def _get_list_of_int(
    data_node,
    torch_graph,
    name_to_tensor,
    accept_none: int = 0,
    has_dynamic_axes: bool = True,
    force_none_as_tensor_ref: bool = False,
):
    assert accept_none >= 0
    accepted_none = 0

    def cast_element(node, accepted_none):
        nonlocal has_dynamic_axes
        tensor = name_to_tensor.get(node.export_name)
        if tensor is not None and (
            force_none_as_tensor_ref or has_dynamic_axes
        ):
            return nnef.Identifier(tensor.name)
        val = node.data
        if val is None and accept_none > 0 and accepted_none < accept_none:
            accepted_none += 1
            return val
        return int(val)

    if isinstance(data_node, PythonConstant):
        int_list = [int(_) for _ in data_node.data]
    elif isinstance(data_node, FixedTensorList):
        int_list = [cast_element(_, accepted_none) for _ in data_node.data]
        if any(_ is None for _ in int_list):
            for ax_data in data_node.data:
                if ax_data.data is None:
                    producer = torch_graph.find_data_node_producer(ax_data)
                    producer.realise_output_type_and_size()
                    if ax_data.data is not None:
                        ax_data.data = ax_data.data.tolist()
            int_list = [cast_element(_, accepted_none) for _ in data_node.data]
            if len([_ for _ in int_list if _ is None]) > 1:
                raise TorchToNNEFNotImplementedError(
                    f"too much unknown dimensions for view {int_list}"
                )
    else:
        raise TorchToNNEFNotImplementedError(
            "Extracting int list from ", data_node
        )

    assert all(
        isinstance(_, (nnef.Identifier, int)) for _ in int_list
    ), int_list
    return int_list


def view(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    (input_node, axis_node) = node.inputs
    dim_data = _get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=has_dynamic_axes,
    )
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


def _cast_to_if_not_dtype_and_variable(
    g,
    name_to_tensor,
    node,
    nnef_tensor: NTensor,
    cast_to: np.dtype,
    suffix: str = "",
):
    """Force casting not expressed in IR graph in case of div by example.

    This is neccessary since tract and maybe other inference engine may not cast
    implicitly to float during div operation by example leading to rounding
    issues.

    """
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=nnef_tensor,
        attrs={
            "to": numpy_dtype_to_tract_str(cast_to),
        },
        output_tensor_name_suffix=suffix,
    )
    return out, ["tract_core"]


def div(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    input_node = node.inputs[0]
    divisor_node = node.inputs[1]
    suffix_div_op_output = ""
    rounding_mode = None

    used_custom_fragment = []

    for c_node in [input_node, divisor_node]:
        c_node.cast_float_inplace()

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    divisor_tensor = get_or_add_tensor_variable_in_nnef(
        g, divisor_node, name_to_tensor
    )
    io_casting_with_dtype = None

    int_types = (torch.int8, torch.int16, torch.int32, torch.int64)
    if hasattr(input_node, "dtype") and input_node.dtype in int_types:
        io_casting_with_dtype = input_node.np_dtype
        if nnef_spec_strict:
            raise TorchToNNEFNotImplementedError(
                "What NNEF compliance mean in such case ?"
            )
        input_tensor, custom_fragments = _cast_to_if_not_dtype_and_variable(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            nnef_tensor=input_tensor,
            cast_to=np.float32,
            suffix="casted",
        )
        used_custom_fragment += custom_fragments

    if len(node.inputs) == 3:
        rounding_mode = node.inputs[2].data

    if len(node.inputs) == 3 or io_casting_with_dtype is not None:
        suffix_div_op_output = "div"

    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix=suffix_div_op_output,
    )

    if rounding_mode:
        out = _add_single_output_op(
            g,
            node,
            name_to_tensor,
            rounding_mode,
            inputs=out,
            output_tensor_name_suffix=""
            if io_casting_with_dtype is None
            else rounding_mode,
        )
        if rounding_mode == "trunc":
            used_custom_fragment.append(rounding_mode)

    if io_casting_with_dtype is not None:
        if nnef_spec_strict:
            raise TorchToNNEFNotImplementedError(
                "What NNEF compliance mean in such case ?"
            )
        _, custom_fragments = _cast_to_if_not_dtype_and_variable(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            nnef_tensor=out,
            cast_to=io_casting_with_dtype,
        )
        used_custom_fragment += custom_fragments
    return list(set(used_custom_fragment))


def floor_divide(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    input_node, divisor_node = node.inputs
    for c_node in [input_node, divisor_node]:
        c_node.cast_float_inplace()

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    divisor_tensor = get_or_add_tensor_variable_in_nnef(
        g, divisor_node, name_to_tensor
    )
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "div",
        inputs=(
            input_tensor,
            divisor_tensor,
        ),
        output_tensor_name_suffix="div",
    )
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "trunc",
        inputs=out,
    )
    return ["trunc"]


def trunc(g, node, name_to_tensor, **kwargs):
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "trunc",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, node.inputs[0], name_to_tensor
        ),
    )
    return ["trunc"]


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
    _add_single_output_op(
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
            # pow(x,y) := exp(x*log(y))
            # so let's precompute log of y and apply exp(x*resy)
            out_mul = _add_single_output_op(
                g,
                node,
                name_to_tensor,
                "log",
                inputs=[
                    get_or_add_tensor_variable_in_nnef(
                        g, input_node, name_to_tensor
                    ),
                ],
                output_tensor_name_suffix="log_part_pow",
            )
            out_log = _add_single_output_op(
                g,
                node,
                name_to_tensor,
                "mul",
                inputs=[
                    out_mul,
                    get_or_add_tensor_variable_in_nnef(
                        g,
                        PythonConstant(
                            name=exponent_node.export_name, data=abs(exponent)
                        ),
                        name_to_tensor,
                    ),
                ],
                output_tensor_name_suffix="mul_part_pow",
            )

            out = _add_single_output_op(
                g,
                node,
                name_to_tensor,
                "exp",
                inputs=[out_log],
                output_tensor_name_suffix="exp_part_pow"
                if exponent < 0
                else "",
            )
            if exponent < 0:  # in case of neg exponent: x**-y = 1/(x**y)
                out = _add_single_output_op(
                    g,
                    node,
                    name_to_tensor,
                    "div",
                    inputs=[
                        get_or_add_tensor_variable_in_nnef(
                            g,
                            PythonConstant(
                                name=f"{out.name}_div_since_neg_part_pow",
                                data=1.0,
                            ),
                            name_to_tensor,
                        ),
                        out,
                    ],
                )

            return
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


def quantize_per_tensor(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        input_node,
        scale_node,
        zero_point_node,
        dtype_node,
    ) = node.inputs
    assert dtype_node.data == 13, "is not expected quint8"
    input_node = node.inputs[0]
    tensor = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    quant_infos = {
        "zero_point": zero_point_node.data,
        "scale": scale_node.data,
        "bits": 8,
        "signed": False,
        "symmetric": False,
        "op-name": "zero_point_linear_quantize",
    }
    if nnef_spec_strict:
        LOGGER.debug(
            "quantize with nnef_spec_strict: set quant info on direct output"
        )
        tensor.quant = quant_infos
        return []
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=tensor,
    )
    out.quant = quant_infos
    return ["tract_core"]


def dequantize(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    """
    We will only handle the case of zero_point affine quantization for now.
    which in reverse of quantization is:

       (x - zero_point) * scale
    """
    input_node = node.inputs[0]
    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError(
            "What NNEF compliance mean in such case"
        )
    _, fragment_names = _cast_to_if_not_dtype_and_variable(
        g,
        name_to_tensor,
        node,
        nnef_tensor,
        cast_to=np.float32,
    )
    return fragment_names


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

    _add_single_output_op(
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
    _add_single_output_op(
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
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "concat",
        inputs=inputs,
        attrs={"axis": pick_rank(input_node, dim)},
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
    _add_single_output_op(
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
    _add_multi_output_op(
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
    _add_single_output_op(
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
    _add_single_output_op(
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


def _reducer(aten_op_name: str, g, node, name_to_tensor, output_idx: int = 0):

    (input_node, axis_node, keep_dim_node) = node.inputs

    keep_dim = keep_dim_node.data

    onode = node.outputs[output_idx]
    out = add_tensor_variable_node_as_nnef_tensor(
        g,
        onode,
        name_to_tensor,
        prevent_variable=True,
    )
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

    # can be either 1 or n axes {
    if isinstance(axis_node.data, int):
        axes = [pick_rank(input_node, axis_node.data)]
    else:
        axes = [pick_rank(input_node, _) for _ in axis_node.data]
    #  }
    attribs = {"axes": axes}
    add_nnef_operation(
        graph=g,
        type=aten_op_name,
        name=f"{onode.export_name}_{aten_op_name}",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        outputs=out if keep_dim else op_reduce_out,
        attribs=attribs,
    )
    if not keep_dim:
        add_nnef_operation(
            graph=g,
            type="squeeze",
            name=f"{onode.export_name}_squeeze",
            inputs=op_reduce_out,
            outputs=out,
            attribs=attribs,
        )


def mean(g, node, name_to_tensor, **kwargs):
    _reducer("mean_reduce", g, node, name_to_tensor)


def reduce_sum(g, node, name_to_tensor, **kwargs):
    _reducer("sum_reduce", g, node, name_to_tensor)


def argmax(g, node, name_to_tensor, **kwargs):
    _reducer("argmax_reduce", g, node, name_to_tensor)


def argmin(g, node, name_to_tensor, **kwargs):
    _reducer("argmin_reduce", g, node, name_to_tensor)


def reduce_any(g, node, name_to_tensor, **kwargs):
    assert len(node.outputs) == 1
    _reducer("any_reduce", g, node, name_to_tensor)


def reduce_all(g, node, name_to_tensor, **kwargs):
    assert len(node.outputs) == 1
    _reducer("all_reduce", g, node, name_to_tensor)


def reduce_max(g, node, name_to_tensor, **kwargs):
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise TorchToNNEFNotImplementedError(
            f"unknown 'max' variant with {n_outputs} outputs used"
        )
    _reducer("max_reduce", g, node, name_to_tensor)
    if n_outputs == 2:
        _reducer("argmax_reduce", g, node, name_to_tensor, output_idx=1)


def reduce_min(g, node, name_to_tensor, **kwargs):
    n_outputs = len(node.outputs)
    if n_outputs > 2:
        raise TorchToNNEFNotImplementedError(
            f"unknown 'min' variant with {n_outputs} outputs used"
        )
    _reducer("min_reduce", g, node, name_to_tensor)
    if n_outputs == 2:
        _reducer("argmin_reduce", g, node, name_to_tensor, output_idx=1)


def max_(g, node, name_to_tensor, null_ref, **kwargs):
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_max(g, node, name_to_tensor)
    return _unary_output_op_without_params(
        nnef_op_type="max",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


def min_(g, node, name_to_tensor, null_ref, **kwargs):
    if isinstance(node.inputs[1], PythonConstant):
        return reduce_min(g, node, name_to_tensor)
    return _unary_output_op_without_params(
        nnef_op_type="min",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )


def repeat(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs
    _add_single_output_op(
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
        out = _add_single_output_op(
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
        _add_single_output_op(
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

    dim_data = _get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=has_dynamic_axes,
        force_none_as_tensor_ref=True,
    )
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


def pad(node, **kwargs):
    kind = node.inputs.pop(2)
    if kind.data == "constant":
        return constant_pad_nd(node=node, **kwargs)
    if kind.data in ["reflection", "reflect"]:  # pre 1.12.0  # post 1.12.0
        node.inputs = node.inputs[:2]
        return reflection_padnd(node=node, **kwargs)
    if kind.data == "replicate":
        node.inputs = node.inputs[:2]
        return replication_padnd(node=node, **kwargs)
    raise TorchToNNEFNotImplementedError(
        f"pad kind={kind.data} not implemented"
    )


def reflection_padnd(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    (input_node, pads_node) = node.inputs
    pads = _get_list_of_int(
        pads_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    assert isinstance(pads, list)
    assert all(isinstance(_, int) for _ in pads)
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
        attrs={"padding": pads, "border": "reflect"},
    )


def replication_padnd(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    (input_node, pads_node) = node.inputs
    pads = _get_list_of_int(
        pads_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    assert isinstance(pads, list)
    assert all(isinstance(_, int) for _ in pads)
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


def constant_pad_nd(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    (input_node, pads_node, value_node) = node.inputs
    pads = _get_list_of_int(
        pads_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    assert isinstance(pads, list)
    assert all(isinstance(_, int) for _ in pads)
    value = value_node.data
    if value is None:
        value = 0  # add default value if not set
    # ensure cast to same dtype as output
    value = torch.tensor(value, dtype=node.outputs[0].dtype).tolist()
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


def where(g, node, name_to_tensor, **kwargs):
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


def matmul(g, node, name_to_tensor, **kwargs):
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
    for (out_node, n_elements) in zip(node.outputs, ratio_node.data):
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
        add_nnef_operation(
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


def arange(g, node, name_to_tensor, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (start_node, end_node, step_node) = node.inputs
    LOGGER.warning(
        "aten::arange replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    node.outputs[0].data = torch.arange(
        start_node.data, end_node.data, step=step_node.data
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


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
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="select",
        inputs=inputs,
    )


def ones(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    LOGGER.warning(
        "the aten::ones replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = torch.float32
    if len(_) > 0:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[_[0].data]
    dim_data = _get_list_of_int(
        input_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    node.outputs[0].data = torch.ones(dim_data, dtype=dtype)
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


def zeros_like(g, node, name_to_tensor, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    LOGGER.warning(
        "the aten::zeros_like replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = torch.float32
    if len(_) > 0:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[_[0].data]

    node.outputs[0].data = torch.zeros(
        input_node.shape,
        dtype=dtype,
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


def new_zeros(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        _,  # input_node,
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs
    LOGGER.warning(
        "the aten::new_zeros replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]

    assert shape_node.data

    node.outputs[0].data = torch.zeros(
        shape_node.data,
        dtype=dtype,
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


def zeros(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs
    LOGGER.warning(
        "the aten::zeros replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]

    assert shape_node.data

    node.outputs[0].data = torch.zeros(
        shape_node.data,
        dtype=dtype,
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
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
        add_nnef_operation(
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


def layer_norm(g, node, name_to_tensor, null_ref, **kwargs):
    (
        input_tensor_node,
        normalized_shape_node,
        weight_node,
        bias_node,
        eps_node,
        elementwise_affine_node,
    ) = node.inputs

    mean_axes = sorted(
        input_tensor_node.rank - r - 1
        for r, _ in enumerate(normalized_shape_node.data)
    )
    has_affine = elementwise_affine_node.data and not (
        # check affine as any use
        (bias_node.data == 0).all().tolist()
        and (weight_node.data == 1).all().tolist()
    )
    inputs = [input_tensor_node]
    op_name = "layer_norm"
    if has_affine:
        op_name = "layer_norm_with_affine"
        inputs += [weight_node, bias_node]
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=op_name,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            if _
            else null_ref
            for _ in inputs
        ],
        attrs={"mean_axes": mean_axes, "eps": eps_node.data},
    )

    return [op_name]


def _expand_build_repeats(input_node, shape_node, shapes):
    repeats = []
    for input_dim, shape_dim in zip(
        input_node.shape, shapes[-len(input_node.shape) :]
    ):
        if shape_dim in [-1, input_dim]:
            repeats.append(1)
        else:
            if input_dim > 1:
                if isinstance(shape_dim, nnef.Identifier):
                    raise TorchToNNEFNotImplementedError(
                        "Need for addition of div Op. Not yet implemented"
                    )
                repeats.append(int(shape_dim / input_dim))
            else:
                # div per 1 hence shape_dim
                repeats.append(shape_dim)

    if len(shape_node.data) - input_node.rank > 0:
        base_mul = 1
        mul_to_ids = []
        for val in shape_node.data[: -input_node.rank]:
            if isinstance(val, TensorVariable):
                mul_to_ids.append(val)
            else:
                base_mul *= val
        if mul_to_ids:
            if base_mul == 1 and len(mul_to_ids) == 1:
                base_mul = nnef.Identifier(mul_to_ids[0].export_name)
            else:
                raise TorchToNNEFNotImplementedError(
                    "In such case would need to apply mul chain ops "
                    "and replace base_mul with related assigned symbol"
                )
        repeats.insert(0, base_mul)
    return repeats


def expand(
    g, node, name_to_tensor, nnef_spec_strict, has_dynamic_axes, **kwargs
):
    """
    Illustration of expand:
        torch.arange(9).reshape(3, 3).expand(2, 3, 3)

        Out[4]:
        tensor([[[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]],

                [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]]])

    which can be re-expressed as:
        torch.arange(9).reshape(3, 3).repeat(2).reshape(2, 3, 3)

    this allows us to express it as a NNEF tile followed by a reshape.

    """
    (input_node, shape_node) = node.inputs

    shapes = []
    for dim in shape_node.data:
        if isinstance(dim, PythonConstant):
            dim = dim.data
        elif isinstance(dim, TensorVariable):
            if nnef_spec_strict or not has_dynamic_axes:
                dim = int(dim.data)
            else:
                dim = nnef.Identifier(dim.export_name)
        shapes.append(dim)

    repeats = _expand_build_repeats(input_node, shape_node, shapes)

    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"repeats": repeats},
        output_tensor_name_suffix="repeat",
    )
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=out,
        attrs={"shape": fill_negone_with_dim_by_rank_order(input_node, shapes)},
    )


def glu(g, node, name_to_tensor, **kwargs):
    input_node, axis_node = node.inputs
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="glu",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
        ],
        attrs={
            "axis": pick_rank(input_node, axis_node.data),
            "half_dim_size": int(input_node.shape[axis_node.data] / 2),
            "dim_size": input_node.shape[axis_node.data],
        },
    )
    return ["glu"]


def clamp_min(g, node, name_to_tensor, **kwargs):
    input_node = node.inputs[0]
    clamp_value_node = node.inputs[1]

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="max",
        inputs=[
            input_tensor,
            get_or_add_tensor_variable_in_nnef(
                g, clamp_value_node, name_to_tensor
            ),
        ],
    )


def clamp_max(g, node, name_to_tensor, **kwargs):
    input_node = node.inputs[0]
    clamp_value_node = node.inputs[1]

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="min",
        inputs=[
            input_tensor,
            get_or_add_tensor_variable_in_nnef(
                g, clamp_value_node, name_to_tensor
            ),
        ],
    )


def clamp(g, node, name_to_tensor, **kwargs):
    input_node, min_clamp, max_clamp = node.inputs

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if min_clamp.data:
        output = _add_single_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="max",
            inputs=[
                input_tensor,
                get_or_add_tensor_variable_in_nnef(
                    g, min_clamp, name_to_tensor
                ),
            ],
            output_tensor_name_suffix="clamp_min" if max_clamp.data else "",
        )
        input_tensor = output

    if max_clamp.data:
        _add_single_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="min",
            inputs=[
                input_tensor,
                get_or_add_tensor_variable_in_nnef(
                    g, max_clamp, name_to_tensor
                ),
            ],
        )


def group_norm(g, node, name_to_tensor, **kwargs):
    """
    It is a special case of NNEF batch_normalization
    with variance and mean being tensor
    """
    (
        input_node,
        n_groups_node,
        scale_node,
        offset_node,
        eps_node,
        _,  # is_affine_node
    ) = node.inputs
    for nd in [offset_node, scale_node]:
        for _ in range(input_node.rank - nd.rank - 1):
            nd.data = nd.data.unsqueeze(-1)
        nd.shape = list(nd.data.shape)

    offset_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="offset",
        node=offset_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    scale_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="scale",
        node=scale_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    # x.reshape(3, 1* 2* 2).mean_or_std(dim=1).repeat(2, 1).t().reshape(6)
    _add_single_output_op(
        g=g,
        name_to_tensor=name_to_tensor,
        node=node,
        nnef_op_type="group_norm",
        # name=f"{node.outputs[0].export_name}_op",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            offset_ref,
            scale_ref,
        ),
        attrs={
            "epsilon": eps_node.data,
            "num_groups": n_groups_node.data,
            "batch_size": input_node.shape[0],
            "num_channels": input_node.shape[1],
        },
    )
    return ["group_norm"]


def select(g, node, name_to_tensor, **kwargs):
    input_node, axis_node, index_node = node.inputs
    out = _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "axes": [pick_rank(input_node, axis_node.data)],
            "begin": [
                pick_value_in_rank(input_node, axis_node.data, index_node.data)
            ],
            "end": [
                pick_value_in_rank(
                    input_node, axis_node.data, index_node.data + 1
                )
            ],
            "stride": [1],
        },
        output_tensor_name_suffix="_select",
        pass_quantization_params=True,
    )
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=out,
        attrs={"axes": [pick_rank(input_node, axis_node.data)]},
        pass_quantization_params=True,
    )


def baddbmm(g, node, name_to_tensor, **kwargs):
    input_node, batch1_node, batch2_node, beta_node, alpha_node = node.inputs
    _add_single_output_op(
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


def index_(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    """
    fragment gather<?>(
        input: tensor<?>,                 # the tensor to gather from
        indices: tensor<integer>,         # the indices to gather at
        axis: integer = 0 )               # the axis to gather at
    -> ( output: tensor<?> )
    """
    # gather
    input_node, indexes_node = node.inputs
    # input_node = TensorVariable([?], shape=(169,4))
    # indexes_node = FixedTensorList (data=[TensorVariable([?], shape=(2401,))])
    if len(indexes_node.data) > 1:
        raise TorchToNNEFNotImplementedError("index dim>1 not implemented")

    custom_fragments = []
    if nnef_spec_strict:
        op_name = "gather"
    else:
        op_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
    _add_single_output_op(
        g,
        node,
        name_to_tensor,
        op_name,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            get_or_add_tensor_variable_in_nnef(
                g, indexes_node.data[0], name_to_tensor
            ),
        ],
        attrs={
            "axis": 0,
        },
    )
    return custom_fragments


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
    _add_single_output_op(
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
    _add_single_output_op(
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
        shape_out = _add_single_output_op(
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
        shape_out = _add_single_output_op(
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
        input_tensor = _add_single_output_op(
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


def aten_to_nnef_tensor_and_ops(
    g,
    node,
    name_to_tensor,
    null_ref,
    torch_graph,
    nnef_spec_strict: bool = False,
    has_dynamic_axes: bool = False,
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
        nnef_spec_strict=nnef_spec_strict,
        has_dynamic_axes=has_dynamic_axes,
    )
