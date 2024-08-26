import typing as T

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    weight_bias_and_output_tensor,
)
from torch_to_nnef.qtensor.base import QTensor
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


def _get_padding_same_symetric(
    L_in: int, stride: int, kernel_size: int, dilation: int
) -> T.Tuple[int, int]:
    """This function computes the number of elements to add for zero-padding."""
    if stride > 1:
        raise TorchToNNEFNotImplementedError("stride > 1 not implemented")
    offset = -dilation * (kernel_size - 1) - 1 + 1
    L_out = L_in + offset
    qte_pad = L_in - L_out
    side_pad = qte_pad // 2
    padding = (side_pad, qte_pad - side_pad)
    return padding


@OP_REGISTRY.register()
def _convolution_mode(
    g, node, name_to_tensor, null_ref, inference_target, **kwargs
):
    (
        input_node,
        weight_node,
        bias_node,
        stride_node,
        padding_node,
        dilation_node,
        groups_node,
    ) = node.inputs

    stride = stride_node.data
    dilation = dilation_node.data
    padding = padding_node.data
    groups = groups_node.data

    assert isinstance(padding, str), padding
    if padding == "valid":
        padding = [0] * len(stride)
    elif padding == "same":
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # """
        # NOTES: pads the input so the output has the shape as the input.
        # However, this mode doesn’t support any stride values other than 1.
        # """
        # also:
        # """
        # tries to pad evenly left and right, but if the amount of columns to
        # be added is odd, it will add the extra column to the right.
        # (the same logic applies vertically: there may be an extra row of zeros at the bottom).
        # """
        # NOTE: This implementation have little test coverage
        padding = []
        for idx, _ in enumerate(stride):
            padding.append(
                _get_padding_same_symetric(
                    L_in=input_node.shape[-(idx + 1)],
                    stride=1,
                    kernel_size=weight_node.shape[2:][idx],
                    dilation=dilation[idx],
                )
            )
    else:
        raise TorchToNNEFNotImplementedError(padding)

    # expand in stored variables export to avoid unsqueeze guessing in graph {
    params_nodes = [weight_node]
    if (
        bias_node.data is not None
        and isinstance(inference_target, TractNNEF)
        and inference_target.version < "0.18.1"
    ):
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
        type="conv",
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
def _convolution(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
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

    # TODO: problem with conv on qtensor for weight or bias
    # since these params can now be dynamic
    # >> all following code need to happen in the graph
    # >> TODAY THIS IS THE CASE for all OPS of THIS KIND
    if transposed:
        if groups is not None:
            # torch weight shape:
            # (in_channels, out_channels/ groups, kernel_size[0],kernel_size[1])
            # expected formulation for NNEF: O, I/G, H, W
            i = weight_node.data.shape[0]
            o = weight_node.data.shape[1]
            remaining_shape = list(weight_node.data.shape)[2:]
            expose_group_shape = [groups, int(i / groups), o] + remaining_shape
            final_expected_shape = [
                int(i / groups),
                int(o * groups),
            ] + remaining_shape
            weight_node.data = (
                weight_node.data.reshape(expose_group_shape)
                .transpose(0, 1)
                .reshape(final_expected_shape)
            )
        weight_node.data = weight_node.data.transpose(1, 0)

    # expand in stored variables export to avoid unsqueeze guessing in graph {
    params_nodes = [weight_node]
    if (
        bias_node.data is not None
        and isinstance(inference_target, TractNNEF)
        and inference_target.version < "0.18.1"
    ):
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

    suffix_weight = ""
    suffix_bias = ""
    if weight_node.data is not None:
        if isinstance(weight_node.data, QTensor):
            suffix_weight = "weight_raw2d"
        else:
            for _ in range(input_node.rank - weight_node.rank):
                weight_node.data = weight_node.data.unsqueeze(0)
                weight_node.shape = list(weight_node.data.shape)

    if bias_node.data is not None:
        if isinstance(weight_node.data, QTensor):
            suffix_bias = "bias_raw2d"
        else:
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
        suffix_weight_name=suffix_weight,
        suffix_bias_name=suffix_bias,
    )
    if suffix_weight:
        weight_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "unsqueeze",
            inputs=weight_ref,
            attrs={"axes": [0] * (input_node.rank - weight_node.rank)},
            output_tensor_name_suffix=f"{suffix_weight}_unsqueeze",
        )
    if suffix_bias:
        bias_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "unsqueeze",
            inputs=bias_ref,
            attrs={"axes": [0] * (input_node.rank - bias_node.rank)},
            output_tensor_name_suffix=f"{suffix_bias}_unsqueeze",
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
def einsum(g, node, name_to_tensor, inference_target, **kwargs):
    if not isinstance(inference_target, TractNNEF):
        raise TorchToNNEFNotImplementedError(
            "einsum operator is not supported by `NNEF` and "
            "breaking it down to primitive ops would be a siginficant work"
        )

    expr_node, args_node, _ = node.inputs
    inps_dtypes = {_.dtype for _ in args_node.data}
    assert inps_dtypes, inps_dtypes
    dtype_str = TORCH_DTYPE_TO_TRACT_STR[inps_dtypes.pop()]

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_einsum",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, dnode, name_to_tensor)
            for dnode in args_node.data
        ],
        ensure_tuple=False,
        force_consistent_inputs_shapes=False,
        attrs={"expr": expr_node.data, "acc": dtype_str, "output": ""},
    )
    return ["tract_core"]


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
    for ab_node in [alpha_node, beta_node]:
        if isinstance(alpha_node, PythonConstant):
            ab_node.data = float(ab_node.data)
        else:
            raise TorchToNNEFNotImplementedError()
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
