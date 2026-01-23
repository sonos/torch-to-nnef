import typing as T

import nnef
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten.reducer import reducer_helper
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    get_tract_dyn_axis_size_soc,
)
from torch_to_nnef.torch_graph import Data
from torch_to_nnef.torch_graph.ir_data import PythonConstant, TensorVariable

OP_REGISTRY = AtenOpRegistry()

TRACT_SUPPORT_DYNAMIC_POOLING = False  # to update if that's the case 1 day


def _pooling_op(
    nnef_op_name: str,
    node_inputs: T.List[Data],
    node,
    op_helper,
):
    """Generic pool operator translation from aten to NNEF.

    NNEF (avg|max)_pool params (not dimension specific):.
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
        raise T2NErrorNotImplemented(
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
        if (
            isinstance(op_helper.inference_target, TractNNEF)
            and op_helper.inference_target.version < "0.19.0"
        ):
            padding = padding + ([0] * missing_n_dims)
        else:
            padding = ([0] * missing_n_dims) + padding

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_name,
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
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


@OP_REGISTRY.register()
def max_pool1d(g, node, op_helper, **kwargs):
    """Map PyTorch: 'aten:max_pool1d' to NNEF."""
    _pooling_op("max_pool", node.inputs, node, op_helper)


@OP_REGISTRY.register()
def avg_pool1d(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:avg_pool1d' to NNEF."""
    count_include_pad = node.inputs[-1].data
    if not count_include_pad:
        raise T2NErrorNotImplemented("not implemented count_include_pad=False")
    inputs_name_tuple = node.inputs[:-1]  # count_include_pad excluded
    inputs_name_tuple.insert(4, None)  # set missing dilation

    # Dilation is available
    _pooling_op("avg_pool", inputs_name_tuple, node, op_helper)


@OP_REGISTRY.register(["max_pool2d", "max_pool3d"])
def max_pool_nd(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:max_pool2d', 'aten:max_pool3d' to NNEF."""
    _pooling_op("max_pool", node.inputs, node, op_helper)


@OP_REGISTRY.register(["avg_pool2d", "avg_pool3d"])
def avg_pool_nd(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:avg_pool(2|3)d', 'aten:max_pool3d' to NNEF.

    Cpp func parameters:.
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
        raise T2NErrorNotImplemented("not implemented count_include_pad=False")

    divisor_overide = node.inputs[-1].data
    if divisor_overide:
        raise T2NErrorNotImplemented(
            f"not implemented divisor_override={divisor_overide}"
        )
    inputs_tups = node.inputs[:-2]
    inputs_tups.insert(4, None)
    _pooling_op("avg_pool", inputs_tups, node, op_helper)


def _adaptive_pool(nnef_op_name: str, op_helper, node):
    (
        input_node,
        pool_values_node,
    ) = node.inputs

    pool_values = pool_values_node.data
    if not all(
        dim and dim > 0 for dim in input_node.shape[-len(pool_values) :]
    ):
        raise T2NErrorNotImplemented(
            "dynamic dim used in adaptive pool is not Implemented yet"
        )
    # fixed at export auto adaptation
    onode = node.outputs[0]
    is_reducer = all(pv == 1 for pv in pool_values)
    if (
        TRACT_SUPPORT_DYNAMIC_POOLING
        and isinstance(op_helper.inference_target, TractNNEF)
        and op_helper.inference_target.has_dynamic_axes
        and not is_reducer
    ):
        stride = []
        start_ix = input_node.rank - len(pool_values) - 1
        for axis_offset, pool_val in zip(
            range(start_ix, input_node.rank),
            pool_values,
        ):
            axis = start_ix + axis_offset
            soc = get_tract_dyn_axis_size_soc(op_helper, input_node, axis=axis)
            numerator_nnef = op_helper.name_to_tensor[soc.output_name]
            if pool_val == 1:
                out = numerator_nnef
            else:
                pool_val_nnef = op_helper.get_or_add_tensor_variable_in_nnef(
                    PythonConstant(
                        name=f"{onode.export_name}_pool_val{axis}",
                        data=pool_val,
                    )
                )
                out = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "div",
                    inputs=(
                        numerator_nnef,
                        pool_val_nnef,
                    ),
                    output_tensor_name_suffix=f"stride_{axis}",
                    maybe_cast_align_tract=False,  # here you want to stay TDim
                )
            stride.append(nnef.Identifier(out.name))
    else:
        stride = [
            int(in_tensor_dim // pool_val)
            for pool_val, in_tensor_dim in zip(
                pool_values, input_node.shape[-len(pool_values) :]
            )
        ]

    if onode.rank > len(stride):
        missing_n_dims = onode.rank - len(stride)
        stride = ([1] * missing_n_dims) + stride

    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if is_reducer:
        reduce_node = node
        axes_node = reduce_node.inputs[1]
        axes_node.name += "_reducer"
        axes_node.set_data(
            [input_node.rank - _ - 1 for _ in range(len(axes_node.data))][::-1]
        )
        node.inputs.append(
            PythonConstant(
                name=f"{reduce_node.outputs[0].export_name}_keep_dim", data=True
            )
        )

        return reducer_helper(
            {
                "max_pool": "max_reduce",
                "avg_pool": "mean_reduce",
            }[nnef_op_name],
            reduce_node,
            op_helper,
        )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_name,
        inputs=inp,
        attrs={
            "size": list(stride),
            "padding": [(0, 0) for _ in stride],
            "stride": list(stride),
            "dilation": [1 for _ in stride],
            "border": "ignore",
        },
    )
    return []


# warning! no support for return_indice=True
@OP_REGISTRY.register(
    ["adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d"]
)
def adaptive_avg_poolnd(g, node, op_helper, **kwargs):
    """Map PyTorch: 'aten:adaptive_avg_pool{1,2,3}d' to NNEF."""
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("avg_pool", op_helper, node)


# warning! no support for return_indice=True
@OP_REGISTRY.register(
    ["adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"]
)
def adaptive_max_poolnd(node, op_helper, **kwargs):
    """Map PyTorch: adaptive_max_pool{1,2,3}d to NNEF."""
    node.outputs = node.outputs[:1]
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("max_pool", op_helper, node)


@OP_REGISTRY.register(["upsample_nearest2d"])
def upsample_nearest2d(node, op_helper, **kwargs):
    """Operator mapping PyTorch: 'aten:upsample_nearest2d' to NNEF."""
    (input_node, size_node, scale_factor_node) = node.inputs
    if size_node.data:
        raise T2NErrorNotImplemented("size in upsampling not defined in NNEF")
    if scale_factor_node.data is None or not all(
        isinstance(_, float) for _ in scale_factor_node.data
    ):
        raise T2NErrorNotImplemented(
            f"unable to export scale_factor {scale_factor_node.data}"
        )
    # NOTE: this implmentation is very suboptimal compared to
    # onnx:resize operator:
    # it should be reified in tract as a proper 'debox' operator.
    # Also current implementation anoyingly need to pass
    # the channel dim c (by default it's the 2nd dim)
    # with classical notation: N,Cin,Hin, Win -> N,Cout,Hout,Wout

    scales = [int(sf) for sf in scale_factor_node.data]
    kernel_data = torch.ones([1, 1, 1, 1] + scales)
    kernel = TensorVariable(
        name=f"{node.outputs[0].export_name}_kernel",
        data=kernel_data,
        shape=kernel_data.shape,
        dtype=input_node.dtype,
    )
    bias = PythonConstant(
        name=f"{node.outputs[0].export_name}_bias",
        data=0,
    )
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if (
        isinstance(op_helper.inference_target, TractNNEF)
        and op_helper.inference_target.version > "0.22.0"
        and op_helper.inference_target.upsample_with_debox
    ):
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "debox",
            inputs=inp,
            attrs={
                "size": [1, 1] + scales,
                "stride": [1, 1] + scales,
                "padding": [(0, 0), (0, 0), (0, 0), (0, 0)],
            },
        )
        return []
    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "deconv",
        inputs=(
            inp,
            op_helper.get_or_add_tensor_variable_in_nnef(kernel),
            op_helper.get_or_add_tensor_variable_in_nnef(bias),
        ),
        attrs={
            "stride": [1, 1] + scales,
            "padding": [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        },
        output_tensor_name_suffix="_deconv",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "squeeze",
        inputs=out,
        attrs={"axes": [0, 1]},
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    return []
