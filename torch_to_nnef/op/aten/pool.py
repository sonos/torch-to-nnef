import typing as T

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.torch_graph import Data

OP_REGISTRY = AtenOpRegistry()


def _pooling_op(
    nnef_op_name: str,
    node_inputs: T.List[Data],
    g,
    node,
    name_to_tensor,
    inference_target,
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
        if (
            isinstance(inference_target, TractNNEF)
            and inference_target.version < "0.19.0"
        ):
            padding = padding + ([0] * missing_n_dims)
        else:
            padding = ([0] * missing_n_dims) + padding

    add_single_output_op(
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


@OP_REGISTRY.register()
def max_pool1d(g, node, name_to_tensor, inference_target, **kwargs):
    _pooling_op(
        "max_pool", node.inputs, g, node, name_to_tensor, inference_target
    )


@OP_REGISTRY.register()
def avg_pool1d(g, node, name_to_tensor, inference_target, **kwargs):
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
        inference_target,
    )


@OP_REGISTRY.register(["max_pool2d", "max_pool3d"])
def max_pool_nd(g, node, name_to_tensor, inference_target, **kwargs):
    _pooling_op(
        "max_pool", node.inputs, g, node, name_to_tensor, inference_target
    )


@OP_REGISTRY.register(["avg_pool2d", "avg_pool3d"])
def avg_pool_nd(g, node, name_to_tensor, inference_target, **kwargs):
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
        "avg_pool", inputs_tups, g, node, name_to_tensor, inference_target
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

    add_single_output_op(
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


# warning! no support for return_indice=True
@OP_REGISTRY.register(
    ["adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d"]
)
def adaptive_avg_poolnd(g, node, name_to_tensor, **kwargs):
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("avg_pool", g, node, name_to_tensor)


# warning! no support for return_indice=True
@OP_REGISTRY.register(
    ["adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"]
)
def adaptive_max_poolnd(g, node, name_to_tensor, **kwargs):
    node.outputs = node.outputs[:1]
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("max_pool", g, node, name_to_tensor)
