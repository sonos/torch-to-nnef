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
from torch_to_nnef.tensor.quant import QTensorTract
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
def elu(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:elu' to NNEF.

    PyTorch's `aten::elu(self, alpha=1, scale=1, input_scale=1)` takes
    three scalar parameters. NNEF's standard `elu` fragment is
    hard-coded to `alpha=1`, so we emit a custom `elu` fragment (see
    `torch_to_nnef/op/fragment/elu.nnef`) that exposes `alpha` as an
    attribute. `scale` and `input_scale` are not part of the NNEF op
    surface; the emitter raises on non-default values for those (rare in
    practice -- the common form is `F.elu(x, alpha=k)`).
    """
    input_node = node.inputs[0]
    alpha_node = node.inputs[1] if len(node.inputs) >= 2 else None
    extra_scalar_nodes = node.inputs[2:4]
    for extra in extra_scalar_nodes:
        if (
            isinstance(extra, PythonConstant)
            and extra.data is not None
            and float(extra.data) != 1.0
        ):
            raise T2NErrorNotImplemented(
                f"elu with non-default scale/input_scale (got {extra.data!r})"
            )
    alpha = 1.0
    if isinstance(alpha_node, PythonConstant) and alpha_node.data is not None:
        alpha = float(alpha_node.data)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type="elu",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
        ],
        attrs={"alpha": alpha},
    )
    return ["elu"]


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
    """Map PyTorch: 'aten:prelu' to NNEF.

    PyTorch's `PReLU(num_parameters=C)` stores the slope as a 1-D tensor
    of shape `(C,)` and applies it along the channel axis (dim=1) of an
    input shaped `(B, C, *spatial)`. NNEF broadcasts left-aligned
    (prepends 1s), so a raw `(C,)` weight would broadcast to the
    *trailing* axis instead of the channel axis -- i.e. wrong.

    Pre-unsqueeze the weight to `(C, 1, 1, ...)` so left-alignment
    yields `(1, C, 1, 1, ...)` and broadcast lands on the channel axis.
    Same pattern as group_norm/batch_norm scale/offset. The single-slope
    case (`num_parameters=1` -> shape `(1,)`) is left untouched
    since broadcasting is then trivially correct.
    """
    # avoid unpack/pack {
    node = kwargs["node"]
    # }
    node.inputs = node.inputs[:2]  # remove inplace param
    input_node, weight_node = node.inputs
    weight_data = weight_node.data
    if (
        weight_data is not None
        and getattr(weight_data, "ndim", 0) == 1
        and weight_data.shape[0] > 1
        and input_node.rank is not None
        and input_node.rank > 2
    ):
        if isinstance(weight_data, QTensorTract):
            raise T2NErrorNotImplemented(
                "prelu with quantized multi-parameter weight"
            )
        for _ in range(input_node.rank - weight_node.rank - 1):
            weight_node.set_data(
                weight_node.data.unsqueeze(-1), force_shape=True
            )
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
def threshold(**kwargs):
    """Map PyTorch: 'aten:threshold' to NNEF.

    PyTorch ref: `y = x if x > threshold else value`.
    """
    node = kwargs["node"]
    node.inputs = node.inputs[:3]  # (input, threshold, value)
    for inode in node.inputs[1:]:
        if isinstance(inode, PythonConstant):
            inode.set_data(float(inode.data))
    unary_input_output_op_with_constant("threshold", **kwargs)
    return ["threshold"]


@OP_REGISTRY.register()
def mish(**kwargs):
    """Map PyTorch: 'aten:mish' to NNEF.

    PyTorch ref: `y = x * tanh(softplus(x))`. Tract has no native
    op so we emit a fragment built from `softplus`/`tanh`/`mul`.
    """
    node = kwargs["node"]
    node.inputs = node.inputs[:1]  # drop the inplace flag if present
    unary_output_op_without_attr("mish", **kwargs)
    return ["mish"]


@OP_REGISTRY.register()
def hardsigmoid(**kwargs):
    """Map PyTorch: 'aten:hardsigmoid' to NNEF.

    PyTorch ref: `y = clamp((x + 3) / 6, 0, 1)`. Tract has no native
    op for this, so we emit a custom fragment built from `min`/`max`
    and arithmetic primitives.
    """
    node = kwargs["node"]
    node.inputs = node.inputs[:1]  # drop the inplace flag if present
    unary_output_op_without_attr("hardsigmoid", **kwargs)
    return ["hardsigmoid"]


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
        node.inputs = node.inputs[:1]
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
    """Map PyTorch: 'aten:clamp' to NNEF.

    PyTorch's `clamp(input, min=None, max=None)` skips a bound when it
    is `None` (the unset sentinel) -- NOT when it is 0.0. The earlier
    `if X.data:` truthy check evaluated to False for the literal 0.0,
    silently dropping `min=0` / `max=0` clamps and producing wrong
    output for any input crossing the unset bound. Same root pattern as
    the `flatten` `or 0/-1` bug. Use explicit `is None` checks.
    """
    input_node, min_clamp, max_clamp = node.inputs

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    has_min = min_clamp.data is not None
    has_max = max_clamp.data is not None
    if has_min:
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
            output_tensor_name_suffix="clamp_min" if has_max else "",
        )
        input_tensor = output

    if has_max:
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
