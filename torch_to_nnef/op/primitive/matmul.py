from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    weight_bias_and_output_tensor,
)
from torch_to_nnef.tract import tract_version_lower_than

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
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

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
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


@OP_REGISTRY.register()
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

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
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


@OP_REGISTRY.register()
def einsum(g, node, name_to_tensor, **kwargs):
    raise TorchToNNEFNotImplementedError(
        "einsum operator is not supported by `NNEF` or `tract-nnef` and"
        "breaking it down to primite ops may be tricky"
    )


@OP_REGISTRY.register(
    torch_op_ids=["matmul", "bmm"]
)  # since NNEF matmul does not care about rank
def matmul(g, node, name_to_tensor, **kwargs):
    (
        input_node,
        other_node,
    ) = node.inputs

    add_single_output_op(
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


@OP_REGISTRY.register()
def baddbmm(g, node, name_to_tensor, **kwargs):
    input_node, batch1_node, batch2_node, beta_node, alpha_node = node.inputs
    add_single_output_op(
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
