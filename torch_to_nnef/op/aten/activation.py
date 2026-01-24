from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
    unary_input_output_op_with_constant,
    unary_output_op_without_attr,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register(["softmax", "_softmax"])
def softmax(**kwargs):
    """Map PyTorch: 'aten:softmax', 'aten:_softmax' to NNEF."""
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    if node.inputs[2]:
        del node.inputs[2]

    # enforce use of positive rank
    node.inputs[1].set_data(pick_axis(node.inputs[0], node.inputs[1].data))
    return unary_input_output_op_with_constant("softmax", **kwargs)


@OP_REGISTRY.register()
def softplus(**kwargs):
    """Map PyTorch: 'aten:softplus' to NNEF.

    Note: numerical stability applied in PyTorch is not done in NNEF vanilla
    implementation, nor case beta != 1.

    PyTorch ref:
        y = (1/beta) * log(exp(beta * x) + 1)  if ((beta * x) < thresh) else x

    NNEF ref:
        y = log(exp(x) + 1.0)

    """
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    const = node.inputs[1]
    if const.data != 1:
        raise T2NErrorNotImplemented(
            "This version is not implemented and "
            "would need use of a specific fragment"
        )
    node.inputs = node.inputs[:1]
    return unary_output_op_without_attr("softplus", **kwargs)


@OP_REGISTRY.register()
def elu(**kwargs):
    """Map PyTorch: 'aten:elu' to NNEF."""
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    node.inputs = node.inputs[:2]  # remove inplace param
    return unary_input_output_op_with_constant("elu", **kwargs)


@OP_REGISTRY.register()
def leaky_relu(**kwargs):
    """Map PyTorch: 'aten:leaky_relu' to NNEF."""
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    node.inputs = node.inputs[:2]  # remove inplace param
    return unary_input_output_op_with_constant("leaky_relu", **kwargs)


@OP_REGISTRY.register()
def prelu(**kwargs):
    """Map PyTorch: 'aten:prelu' to NNEF."""
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    node.inputs = node.inputs[:2]  # remove inplace param
    return unary_input_output_op_with_constant("prelu", **kwargs)


@OP_REGISTRY.register()
def selu(**kwargs):
    """Map PyTorch: 'aten:selu' to NNEF."""
    unary_input_output_op_with_constant("selu", **kwargs)
    return ["selu"]


@OP_REGISTRY.register()
def silu(**kwargs):
    """Map PyTorch: 'aten:silu' to NNEF."""
    unary_input_output_op_with_constant("silu", **kwargs)
    return ["silu"]


@OP_REGISTRY.register()
def relu6(**kwargs):
    """Map PyTorch: 'aten:relu6' to NNEF."""
    unary_input_output_op_with_constant("relu6", **kwargs)
    return ["relu6"]


@OP_REGISTRY.register()
def hardswish(inference_target, **kwargs):
    """Map PyTorch: 'aten:hardswish' to NNEF."""
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= "0.19.9"
    ):
        unary_input_output_op_with_constant("tract_core_hard_swish", **kwargs)
        return ["tract_core"]
    unary_input_output_op_with_constant("hardswish", **kwargs)
    return ["relu6", "hardswish"]


@OP_REGISTRY.register()
def gelu(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Map PyTorch: 'aten:gelu' to NNEF."""
    if len(node.inputs) == 2 and node.inputs[1].data == "tanh":
        node.inputs = node.inputs[:1]
        unary_output_op_without_attr(
            "gelu_fast_approx",
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
        return ["gelu_fast_approx"]
    if isinstance(inference_target, TractNNEF):
        unary_output_op_without_attr(
            "tract_gelu",
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
        return ["tract_gelu"]
    unary_output_op_without_attr(
        "gelu",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )
    return ["gelu"]


@OP_REGISTRY.register()
def erf(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Op should be added to tract-nnef eventualy."""
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= "0.19.9"
    ):
        unary_input_output_op_with_constant(
            "tract_core_erf",
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
        return ["tract_core"]
    unary_output_op_without_attr(
        "erf",
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )
    return ["erf"]


@OP_REGISTRY.register()
def hardtanh(**kwargs):
    """Map PyTorch: 'aten:hardtanh' to NNEF."""
    node = kwargs["node"]
    node.inputs = node.inputs[:3]  # remove inplace param
    for inode in node.inputs[1:]:
        if isinstance(inode, PythonConstant):
            inode.set_data(float(inode.data))
    unary_input_output_op_with_constant("hard_tanh", **kwargs)
    return ["hard_tanh"]


@OP_REGISTRY.register()
def log_softmax(inference_target, **kwargs):
    """Map PyTorch: 'aten:log_softmax' to NNEF."""
    node = kwargs["node"]
    if node.inputs[2]:
        del node.inputs[2]
    input_node, axis_node = node.inputs
    axis_node.set_data(pick_axis(input_node, axis_node.data))
    assert isinstance(axis_node.data, int)
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= "0.21.14"
    ):
        unary_input_output_op_with_constant("tract_core_log_softmax", **kwargs)
        return ["tract_core"]
    unary_input_output_op_with_constant("log_softmax", **kwargs)
    return ["log_softmax"]


@OP_REGISTRY.register()
def clamp_min(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:clamp_min' to NNEF."""
    input_node = node.inputs[0]
    clamp_value_node = node.inputs[1]

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    add_single_output_op(
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


@OP_REGISTRY.register()
def clamp_max(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:clamp_max' to NNEF."""
    input_node = node.inputs[0]
    clamp_value_node = node.inputs[1]

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    add_single_output_op(
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


@OP_REGISTRY.register()
def clamp(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:clamp' to NNEF."""
    input_node, min_clamp, max_clamp = node.inputs

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if min_clamp.data:
        output = add_single_output_op(
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
        add_single_output_op(
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


@OP_REGISTRY.register()
def glu(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:glu' to NNEF."""
    input_node, axis_node = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="glu",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
        ],
        attrs={
            "axis": pick_axis(input_node, axis_node.data),
            "half_dim_size": int(input_node.shape[axis_node.data] / 2),
            "dim_size": input_node.shape[axis_node.data],
        },
    )
    return ["glu"]
