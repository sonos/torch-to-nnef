import typing as T

import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import NUMPY_TO_TORCH_DTYPE, TORCH_DTYPE_TO_TRACT_STR
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    weight_bias_and_output_tensor,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant
from torch_to_nnef.utils import LOGGER

OP_REGISTRY = AtenOpRegistry()


def _get_padding_same_symetric(
    l_in: int, stride: int, kernel_size: int, dilation: int
) -> T.Tuple[int, int]:
    """This function computes the number of elements to add for zero-padding."""
    if stride > 1:
        raise T2NErrorNotImplemented("stride > 1 not implemented")
    offset = -dilation * (kernel_size - 1) - 1 + 1
    l_out = l_in + offset
    qte_pad = l_in - l_out
    side_pad = qte_pad // 2
    padding = (side_pad, qte_pad - side_pad)
    return padding


@OP_REGISTRY.register(
    # Most of these aten symbols never reach the trace -- pytorch
    # decomposes them through `aten::_convolution_mode` or
    # `aten::_convolution` upstream. `convolution_overrideable` is an
    # autograd hook for backend-specific dispatch; same story. The
    # registrations are kept so the docs/contributing/supported_operators
    # page reflects reality (every torch.nn `Conv*d` and
    # `F.conv*d` form does work in practice via the canonical lowering).
    [
        "_convolution_mode",
        "convolution",
        "convolution_overrideable",
        "conv1d",
        "conv2d",
        "conv3d",
    ]
)
def _convolution_mode(
    g, node, name_to_tensor, null_ref, inference_target, **kwargs
):
    """Map PyTorch: 'aten:_convolution_mode', 'aten:convolution',... to NNEF."""
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

    if isinstance(padding, (list, tuple)):
        # `aten::conv1d/conv2d/conv3d` already carry an int-list padding,
        # so no normalization is needed.
        pass
    elif padding == "valid":
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
        # (the same logic applies vertically: there may be an extra row of
        # zeros at the bottom).
        # """
        # NOTE: This implementation have little test coverage
        padding = []
        for idx, _ in enumerate(stride):
            padding.append(
                _get_padding_same_symetric(
                    l_in=input_node.shape[-(idx + 1)],
                    stride=1,
                    kernel_size=weight_node.shape[2:][idx],
                    dilation=dilation[idx],
                )
            )
    else:
        raise T2NErrorNotImplemented(padding)

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


def _emit_conv(
    g,
    node,
    name_to_tensor,
    null_ref,
    inference_target,
    *,
    input_node,
    weight_node,
    bias_node,
    stride,
    padding,
    dilation,
    groups,
    transposed,
    output_padding=None,
):
    """Emit NNEF `conv` / `deconv` for the convolution family.

    Shared body of `aten::_convolution` and `aten::conv_transpose{1,2,3}d`.
    Emits `deconv` for `transposed=True`, else `conv`. Transposed convs
    need a weight repack: torch stores
    `(in_channels, out_channels // groups, *spatial)`; NNEF's `deconv`
    expects `(in_channels // groups, out_channels, *spatial)`. The
    grouped-then-transposed reshape is in-place on `weight_node.data`.

    PyTorch's transposed conv has an `output_padding` parameter that
    extends the output by up to `stride-1` on the "after" side, used to
    disambiguate the inverse of a strided conv (multiple input sizes
    yield the same forward-conv output size). NNEF's `deconv` does not
    expose `output_padding` directly, but the difference is exactly the
    same as removing `output_padding` from the cropping on the "after"
    side. So `pytorch (pad, output_padding)` maps to NNEF
    `padding=(pad, pad - output_padding)` (asymmetric). Without this
    asymmetric adjustment, every deconv with `output_padding > 0`
    underestimates the output by `output_padding`, and the shape
    mismatch propagates into downstream shape inference (typically
    surfaced by a `reshape` whose static target no longer matches).
    """
    # TODO: problem with conv on qtensor for weight or bias
    # since these params can now be dynamic
    # >> all following code need to happen in the graph
    # >> TODAY THIS IS THE CASE for all OPS of THIS KIND
    if transposed and isinstance(inference_target, TractNNEF):
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
            weight_node.set_data(
                weight_node.data.reshape(expose_group_shape)
                .transpose(0, 1)
                .reshape(final_expected_shape),
                force_shape=True,
            )
        weight_node.set_data(weight_node.data.transpose(1, 0), force_shape=True)

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )

    nnef_padding = []
    for axis, pad in enumerate(padding):
        if isinstance(pad, int):
            pad_before, pad_after = pad, pad
        else:
            pad_before, pad_after = pad
        if transposed and output_padding is not None:
            op = output_padding[axis] if axis < len(output_padding) else 0
            if op:
                # Subtract output_padding from the "after"-side crop. We
                # require that the result stays non-negative; otherwise
                # we'd need a post-deconv pad op to reach the right
                # output size. PyTorch's constraint `output_padding <
                # max(stride, dilation)` keeps this in the safe range
                # for the common cases (stride>=2 with pad>=1) -- if a
                # model triggers the corner where `pad_after < op`
                # we raise so the gap is explicit rather than silently
                # producing a wrong-shape NNEF.
                if pad_after - op < 0:
                    raise T2NErrorNotImplemented(
                        "conv_transpose `output_padding > padding` "
                        "would require a negative NNEF deconv padding "
                        f"(axis={axis}, pad={pad_after}, "
                        f"output_padding={op}); emit a post-deconv pad "
                        "op instead (not yet implemented)."
                    )
                pad_after -= op
        nnef_padding.append((pad_before, pad_after))

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
            "padding": nnef_padding,
            "stride": list(stride),
            "groups": groups,
            "border": "constant",
        },
        force_consistent_inputs_shapes=False,
    )


@OP_REGISTRY.register()
def _convolution(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Map PyTorch: 'aten:_convolution' to NNEF."""
    (
        input_node,
        weight_node,
        bias_node,
        stride_node,
        padding_node,
        dilation_node,
        transposed_node,
        output_padding_node,
        groups_node,
        _,  # benchmark_name
        _,  # deterministic_name
        _,  # cuda_enabled
        _,  # allow_tf32
    ) = node.inputs
    _emit_conv(
        g,
        node,
        name_to_tensor,
        null_ref,
        inference_target,
        input_node=input_node,
        weight_node=weight_node,
        bias_node=bias_node,
        stride=stride_node.data,
        padding=padding_node.data,
        dilation=dilation_node.data,
        groups=groups_node.data,
        transposed=transposed_node.data,
        output_padding=output_padding_node.data,
    )


@OP_REGISTRY.register(
    ["conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]
)
def conv_transpose_nd(
    g, node, name_to_tensor, null_ref, inference_target, **kwargs
):
    """Map PyTorch: 'aten:conv_transpose{1,2,3}d' to NNEF.

    Marked `CompositeImplicitAutograd` upstream, so PyTorch usually
    decomposes these to `aten::_convolution(transposed=True)` before
    the trace reaches t2n. Registering them anyway keeps the support
    page accurate and gives a working path if PyTorch ever stops
    decomposing for some platform.

    Signature: `(input, weight, bias?, stride, padding, output_padding,
    groups, dilation)` -- 8 positional args.
    """
    (
        input_node,
        weight_node,
        bias_node,
        stride_node,
        padding_node,
        output_padding_node,
        groups_node,
        dilation_node,
    ) = node.inputs
    _emit_conv(
        g,
        node,
        name_to_tensor,
        null_ref,
        inference_target,
        input_node=input_node,
        weight_node=weight_node,
        bias_node=bias_node,
        stride=stride_node.data,
        padding=padding_node.data,
        dilation=dilation_node.data,
        groups=groups_node.data,
        transposed=True,
        output_padding=output_padding_node.data,
    )


@OP_REGISTRY.register()
def conv_tbc(
    g, node, name_to_tensor, null_ref, op_helper, inference_target, **kwargs
):
    """Map PyTorch: 'aten:conv_tbc' to NNEF.

    `conv_tbc(input, weight, bias, pad)` is a 1-D convolution over a
    `(T, B, C)` input -- time-batch-channel layout -- with weight
    `(kernel, C_in, C_out)`. Equivalent semantically to
    `conv1d(input.permute(1, 2, 0), weight.permute(2, 1, 0), bias,
    padding=pad).permute(2, 0, 1)`, which is exactly the
    `permute -> conv -> permute` chain we emit.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    input_node, weight_node, bias_node, pad_node = node.inputs
    pad = int(pad_node.data)
    # Repack weight from (kernel, C_in, C_out) to (C_out, C_in, kernel).
    if weight_node.data is None:
        raise T2NErrorNotImplemented(
            "conv_tbc needs static weight (got graph-input weight)"
        )
    weight_node.set_data(
        weight_node.data.permute(2, 1, 0).contiguous(), force_shape=True
    )
    # Permute input from (T, B, C) to (B, C, T).
    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    inp_bct = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=inp_ref,
        attrs={"axes": [1, 2, 0]},
        output_tensor_name_suffix="ctbc_pre",
    )
    weight_ref = op_helper.get_or_add_tensor_variable_in_nnef(weight_node)
    bias_ref = (
        op_helper.get_or_add_tensor_variable_in_nnef(bias_node)
        if bias_node.data is not None
        else null_ref
    )
    # 1-D conv: output (B, C_out, T_out).
    onode = node.outputs[0]
    t_out, b_out, c_out = onode.shape
    conv_out = NTensor(
        g,
        f"{onode.export_name}_ctbc_conv",
        dtype=onode.np_dtype,
        shape=(b_out, c_out, t_out),
    )
    name_to_tensor[conv_out.name] = conv_out
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="conv",
        name=f"{conv_out.name}_op",
        inputs=(inp_bct, weight_ref, bias_ref),
        outputs=(conv_out,),
        attribs={
            "dilation": [1],
            "padding": [(pad, pad)],
            "stride": [1],
            "groups": 1,
            "border": "constant",
        },
        force_consistent_inputs_shapes=False,
    )
    # Permute back to (T, B, C_out).
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=conv_out,
        attrs={"axes": [2, 0, 1]},
    )


@OP_REGISTRY.register()
def linear(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Map PyTorch: 'aten:linear' to NNEF."""
    (
        input_node,
        weight_node,
        bias_node,
    ) = node.inputs

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
        suffix_weight_name=(
            "weight_raw2d" if weight_node.data is not None else ""
        ),
        suffix_bias_name="bias_raw2d" if bias_node.data is not None else "",
    )

    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.force_linear_accumulation_in_f32
        and weight_node.dtype != torch.float32
    ):
        if inference_target.version < "0.21.11":
            LOGGER.warning(
                "linear can not yet have "
                "accumulation in f32 (waiting tract>=0.21.11)"
                " fallback to f16"
            )
        else:
            if input_node.rank == 3:
                expr = "bij,kj->bik"
                if weight_node.rank != 2:
                    raise T2NErrorNotImplemented(weight_node.rank)
            elif input_node.rank == 4:
                expr = "bcij,ckj->bcik"
                if weight_node.rank != 3:
                    raise T2NErrorNotImplemented(weight_node.rank)
            else:
                raise T2NErrorNotImplemented(node.inputs[0].rank)

            intermediate_output = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_einsum",
                inputs=[
                    get_or_add_tensor_variable_in_nnef(
                        g, input_node, name_to_tensor
                    ),
                    weight_ref,
                ],
                ensure_tuple=False,
                force_consistent_inputs_shapes=False,
                attrs={"expr": expr, "acc": "f32", "output": ""},
                output_tensor_name_suffix="_linear",
            )

            if bias_ref is not None:
                bias_ref = add_single_output_op(
                    g,
                    node,
                    name_to_tensor,
                    "tract_core_cast",
                    inputs=bias_ref,
                    attrs={"to": "f32"},
                    output_tensor_name_suffix="_biasf32",
                )
                bias_ref = add_single_output_op(
                    g,
                    node,
                    name_to_tensor,
                    "unsqueeze",
                    inputs=bias_ref,
                    attrs={"axes": list(range(node.outputs[0].rank - 1))},
                    output_tensor_name_suffix="_biasf32_unsqueezed",
                )
                intermediate_output = add_single_output_op(
                    g,
                    node,
                    name_to_tensor,
                    "add",
                    inputs=[intermediate_output, bias_ref],
                    force_consistent_inputs_shapes=False,
                    output_tensor_name_suffix="_biased",
                )

            cast_and_add_nnef_operation(
                name_to_tensor=name_to_tensor,
                graph=g,
                type="tract_core_cast",
                name=f"{node.outputs[0].export_name}_op",
                inputs=intermediate_output,
                outputs=output_tensor,
                attribs={"to": TORCH_DTYPE_TO_TRACT_STR[input_node.dtype]},
            )
            return ["tract_core"]

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
    return []


@OP_REGISTRY.register()
def einsum(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:einsum' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "einsum operator is not supported by `NNEF` and "
            "breaking it down to primitive ops would be a siginficant work"
        )

    expr_node, args_node = node.inputs[:2]
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


def _emit_unsqueeze_intermediate(g, src: NTensor, axes, suffix: str) -> NTensor:
    """Emit an `unsqueeze` whose output carries the right intermediate shape.

    `add_single_output_op` builds its output NNEF tensor from
    `node.outputs[0]`, which for chained-helper emits means every
    intermediate would inherit the *final* op's shape. For multi-step
    decomposition we need to author the intermediate NTensor by hand.
    """
    new_shape = list(src.shape)
    for ax in axes:
        new_shape.insert(ax, 1)
    out = NTensor(
        g,
        name=f"{src.name}_{suffix}",
        dtype=src.dtype,
        shape=tuple(new_shape),
    )
    NOperation(
        g,
        type="unsqueeze",
        attribs={"axes": list(axes)},
        inputs=src,
        outputs=out,
    )
    return out


@OP_REGISTRY.register(
    torch_op_ids=["matmul", "bmm", "mm"]
)  # since NNEF matmul does not care about rank
def matmul(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:matmul', 'aten:bmm', 'aten:mm' to NNEF.

    NNEF `matmul` requires *equal* rank on both operands; PyTorch's
    `aten::matmul` accepts rank-1 forms with these documented semantics:

      - `(K,) @ (..., K, N)`   -> `(..., N)` : promote A to `(..., 1, K)`,
        matmul gives `(..., 1, N)`, squeeze the row-1 axis.
      - `(..., M, K) @ (K,)`   -> `(..., M)` : promote B to `(..., K, 1)`,
        matmul gives `(..., M, 1)`, squeeze the col-1 axis.
      - `(K,) @ (K,)`          -> `()` (scalar): promote both, matmul
        gives `(1, 1)`, squeeze both axes.

    Both inputs are promoted to a common `target_rank = max(a_rank,
    b_rank, 2)` -- batch dims get prepended `1`s; for rank-1 inputs the
    vector lands as the row (A) or column (B) dim with a singleton on
    the opposite side. The post-matmul squeeze drops those singletons
    to match `_infer_trace_result_matmul`'s rank prediction so
    downstream `unsqueeze(-1)` / shape ops resolve their axes against
    the same rank the IR is tracking.
    """
    (input_node, other_node) = node.inputs
    a_rank = input_node.rank
    b_rank = other_node.rank

    a_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    b_ref = get_or_add_tensor_variable_in_nnef(g, other_node, name_to_tensor)

    a_is_v = a_rank < 2
    b_is_v = b_rank < 2

    if not (a_is_v or b_is_v):
        # Standard case: both rank >= 2. Let the generic emitter handle
        # it -- `force_consistent_inputs_shapes` prepends 1s to the
        # smaller-rank input if needed (e.g. rank-3 @ rank-2), and the
        # IR shape inferrer already accounts for that broadcasting.
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "matmul",
            inputs=(a_ref, b_ref),
            attrs={"transposeA": False, "transposeB": False},
        )
        return

    target_rank = max(a_rank, b_rank, 2)

    if a_is_v:
        # (K,) -> (1, ..., 1, K): leading singletons for batch dims +
        # the row axis. After this A has shape `(1,) * (target_rank-1) +
        # (K,)`; matmul will broadcast batch dims with B.
        a_axes = list(range(target_rank - 1))
        a_ref = _emit_unsqueeze_intermediate(g, a_ref, a_axes, "matmul_a_rank2")
    elif a_rank < target_rank:
        a_axes = list(range(target_rank - a_rank))
        a_ref = _emit_unsqueeze_intermediate(g, a_ref, a_axes, "matmul_a_bcast")

    if b_is_v:
        # (K,) -> (1, ..., 1, K, 1): leading singletons for batch dims,
        # K stays, then a trailing singleton on the col axis.
        b_axes = list(range(target_rank - 2)) + [target_rank - 1]
        b_ref = _emit_unsqueeze_intermediate(g, b_ref, b_axes, "matmul_b_rank2")
    elif b_rank < target_rank:
        b_axes = list(range(target_rank - b_rank))
        b_ref = _emit_unsqueeze_intermediate(g, b_ref, b_axes, "matmul_b_bcast")

    matmul_out_shape = list(a_ref.shape)
    matmul_out_shape[-1] = b_ref.shape[-1]
    matmul_out = NTensor(
        g,
        name=f"{node.outputs[0].export_name}_matmul",
        dtype=a_ref.dtype,
        shape=tuple(matmul_out_shape),
    )
    NOperation(
        g,
        type="matmul",
        attribs={"transposeA": False, "transposeB": False},
        inputs=(a_ref, b_ref),
        outputs=matmul_out,
    )

    squeeze_axes = []
    if a_is_v:
        squeeze_axes.append(target_rank - 2)
    if b_is_v:
        squeeze_axes.append(target_rank - 1)

    out_final = add_tensor_variable_node_as_nnef_tensor(
        g, node.outputs[0], name_to_tensor, prevent_variable=True
    )
    NOperation(
        g,
        type="squeeze",
        attribs={"axes": sorted(squeeze_axes)},
        inputs=matmul_out,
        outputs=out_final,
    )


@OP_REGISTRY.register()
def cartesian_prod(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::cartesian_prod(tensors)` to NNEF.

    Inputs are 1-D tensors of sizes `n_0..n_{K-1}`; the result is a
    `(prod_k n_k, K)` matrix whose rows enumerate every tuple in
    lexicographic order over the first axis.

    Each input column `k` is built by `unsqueeze` + `tile` over all
    other dims, then `reshape` to `(prod, 1)`. The K columns are
    concatenated along axis 1. Static sizes only.
    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.torch_graph import FixedTensorList

    (tensors_node,) = node.inputs
    if not isinstance(tensors_node, FixedTensorList):
        raise T2NErrorNotImplemented(
            f"cartesian_prod expects a FixedTensorList; got {tensors_node!r}"
        )
    items = list(tensors_node.data)
    if not items:
        raise T2NErrorNotImplemented(
            "cartesian_prod with an empty list is not supported"
        )
    for t in items:
        if t.rank != 1:
            raise T2NErrorNotImplemented(
                f"cartesian_prod: only 1-D inputs supported, got rank {t.rank}"
            )
        if not isinstance(t.shape[0], int):
            raise T2NErrorNotImplemented(
                f"cartesian_prod: dynamic size not supported (got {t.shape})"
            )
    sizes = [int(t.shape[0]) for t in items]
    total = 1
    for s in sizes:
        total *= s
    k = len(items)
    columns = []
    for idx, t in enumerate(items):
        ref = get_or_add_tensor_variable_in_nnef(g, t, name_to_tensor)
        # Insert (K - 1) trailing/leading singleton axes so the value
        # axis lands at position `idx` in a rank-K view, then `tile`
        # over the other dims.
        axes_to_unsqueeze = [j for j in range(k) if j != idx]
        unsq = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=ref,
            attrs={"axes": axes_to_unsqueeze},
            output_tensor_name_suffix=f"_cprod_unsq_{idx}",
        )
        repeats = list(sizes)
        repeats[idx] = 1
        tiled = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tile",
            inputs=unsq,
            attrs={"repeats": repeats},
            output_tensor_name_suffix=f"_cprod_tile_{idx}",
        )
        flat = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "reshape",
            inputs=tiled,
            attrs={
                "dtype": node.outputs[0].np_dtype,
                "shape": [total, 1],
            },
            output_tensor_name_suffix=f"_cprod_flat_{idx}",
        )
        columns.append(flat)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "concat",
        inputs=columns,
        attrs={"axis": 1},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def block_diag(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::block_diag(tensors)` to NNEF (rank-2 blocks only).

    Builds an `(M, N)` matrix where `M = sum m_i`, `N = sum n_i` and
    each block `i` (shape `(m_i, n_i)`) sits at row offset
    `sum_{j<i} m_j` and col offset `sum_{j<i} n_j`; off-diagonal
    blocks are zero.

    Each block is `pad`-extended to full width `(m_i, N)` then all
    are `concat`-stacked along axis 0. Static shapes only.
    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.torch_graph import FixedTensorList

    (mats_node,) = node.inputs
    if not isinstance(mats_node, FixedTensorList):
        raise T2NErrorNotImplemented(
            f"block_diag expects a FixedTensorList; got {mats_node!r}"
        )
    blocks = list(mats_node.data)
    if not blocks:
        raise T2NErrorNotImplemented(
            "block_diag with an empty list is not supported"
        )
    for blk in blocks:
        if blk.rank != 2:
            raise T2NErrorNotImplemented(
                f"block_diag: only rank-2 blocks supported; got rank "
                f"{blk.rank}. (PyTorch promotes 0-D / 1-D to (1, 1) / "
                "(1, n); add a pre-unsqueeze pass if needed.)"
            )
        if not all(isinstance(d, int) for d in blk.shape):
            raise T2NErrorNotImplemented(
                "block_diag: dynamic block shapes not supported "
                f"(got {blk.shape})."
            )
    row_sizes = [int(b.shape[0]) for b in blocks]
    col_sizes = [int(b.shape[1]) for b in blocks]
    total_cols = sum(col_sizes)
    padded_refs = []
    col_offsets = [0]
    for n_i in col_sizes[:-1]:
        col_offsets.append(col_offsets[-1] + n_i)
    right_pads = [
        total_cols - off - n_i
        for off, n_i in zip(col_offsets, col_sizes, strict=True)
    ]
    for idx, (blk, left_pad, right_pad) in enumerate(
        zip(blocks, col_offsets, right_pads, strict=True)
    ):
        blk_ref = get_or_add_tensor_variable_in_nnef(g, blk, name_to_tensor)
        if left_pad == 0 and right_pad == 0:
            padded_refs.append(blk_ref)
            continue
        padded_refs.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "pad",
                inputs=blk_ref,
                attrs={
                    "padding": [(0, 0), (int(left_pad), int(right_pad))],
                    "border": "constant",
                    "value": 0.0,
                },
                output_tensor_name_suffix=f"_blkdiag_pad_{idx}",
            )
        )
    del row_sizes
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "concat",
        inputs=padded_refs,
        attrs={"axis": 0},
        ensure_tuple=False,
    )


@OP_REGISTRY.register()
def kron(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::kron(a, b)` to NNEF (rank-2 inputs only).

    Kronecker product:
    `kron(a, b)[i*p+k, j*q+l] = a[i, j] * b[k, l]` for `a` `(m, n)`
    and `b` `(p, q)`, producing `(m*p, n*q)`.

    Lowered via interleaved unsqueeze + broadcast-mul + reshape:
        `a_e = a.unsqueeze(1).unsqueeze(3)`  -> `(m, 1, n, 1)`
        `b_e = b.unsqueeze(0).unsqueeze(2)`  -> `(1, p, 1, q)`
        `prod = a_e * b_e`                   -> `(m, p, n, q)`
        `out = prod.reshape(m*p, n*q)`

    Higher-rank inputs would need the same interleaved pattern at
    every dim, plus shape resolution per axis; raise for now.
    """
    a_node, b_node = node.inputs[:2]
    if a_node.rank != 2 or b_node.rank != 2:
        raise T2NErrorNotImplemented(
            "kron: only rank-2 inputs supported. Higher-rank kron "
            "requires interleaved unsqueeze on every axis. "
            f"Got a.rank={a_node.rank}, b.rank={b_node.rank}."
        )
    m_dim, n_dim = a_node.shape
    p_dim, q_dim = b_node.shape
    if not all(isinstance(d, int) for d in (m_dim, n_dim, p_dim, q_dim)):
        raise T2NErrorNotImplemented(
            "kron: dynamic input shapes not supported "
            f"(got a.shape={a_node.shape}, b.shape={b_node.shape})."
        )
    a_ref = get_or_add_tensor_variable_in_nnef(g, a_node, name_to_tensor)
    b_ref = get_or_add_tensor_variable_in_nnef(g, b_node, name_to_tensor)
    a_unsq = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=a_ref,
        attrs={"axes": [1, 3]},
        output_tensor_name_suffix="_kron_a_unsq",
    )
    b_unsq = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=b_ref,
        attrs={"axes": [0, 2]},
        output_tensor_name_suffix="_kron_b_unsq",
    )
    prod = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=[a_unsq, b_unsq],
        output_tensor_name_suffix="_kron_prod",
        force_consistent_inputs_shapes=False,
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=prod,
        attrs={
            "dtype": node.outputs[0].np_dtype,
            "shape": [m_dim * p_dim, n_dim * q_dim],
        },
    )


@OP_REGISTRY.register()
def inner(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::inner(input, other)` to NNEF.

    `inner(a, b)[..., i, ..., j, ...] = sum(a[..., i, :] * b[..., j, :])`,
    i.e. a matmul of `a` against `b.transpose(-1, -2)`. Works for any
    rank: the trailing axis is reduced and the leading dims of `a` /
    `b` stack as independent index dims.

    For 1D inputs the trace materializes torch's "0-D scalar" output;
    NNEF doesn't have a 0-D tensor type, so for the 1D case we let the
    standard matmul emit a `(1, 1)` result and rely on torch's trace
    having squeezed it upstream (it does -- `aten::inner` with 1D
    inputs is the dot-product overload that returns a 0-D real).
    """
    a_node, b_node = node.inputs[:2]
    if a_node.rank != 2 or b_node.rank != 2:
        raise T2NErrorNotImplemented(
            "inner: only rank-2 inputs supported. Higher-rank "
            "`torch.inner` does a Cartesian product over the leading "
            "dims (shape `(*a_dims, *b_dims)`) rather than a batched "
            "matmul; that would need leading-dim flatten + reshape. "
            f"Got a.rank={a_node.rank}, b.rank={b_node.rank}."
        )
    a_ref = get_or_add_tensor_variable_in_nnef(g, a_node, name_to_tensor)
    b_ref = get_or_add_tensor_variable_in_nnef(g, b_node, name_to_tensor)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "matmul",
        inputs=[a_ref, b_ref],
        attrs={"transposeA": False, "transposeB": True},
        force_consistent_inputs_shapes=False,
    )


@OP_REGISTRY.register()
def vdot(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::vdot(a, b)` to NNEF.

    1-D dot product `sum(a * b)`. Complex inputs would need a conjugate
    on `a`; raise for now (real-only).
    """
    a_node, b_node = node.inputs[:2]
    if a_node.dtype in (torch.complex64, torch.complex128) or b_node.dtype in (
        torch.complex64,
        torch.complex128,
    ):
        raise T2NErrorNotImplemented(
            "vdot: complex inputs not supported (needs conjugate of `a`)"
        )
    if a_node.rank != 1 or b_node.rank != 1:
        raise T2NErrorNotImplemented("vdot: only 1-D inputs supported")
    a_ref = get_or_add_tensor_variable_in_nnef(g, a_node, name_to_tensor)
    b_ref = get_or_add_tensor_variable_in_nnef(g, b_node, name_to_tensor)
    prod = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=[a_ref, b_ref],
        output_tensor_name_suffix="_vdot_mul",
    )
    reduced = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "sum_reduce",
        inputs=prod,
        attrs={"axes": [0]},
        output_tensor_name_suffix="_vdot_reduce",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "squeeze", inputs=reduced, attrs={"axes": [0]}
    )


@OP_REGISTRY.register()
def bilinear(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map `aten::bilinear(input1, input2, weight, bias)` to NNEF.

    `bilinear(x1, x2, W, b)[..., k] = sum_{i, j} x1[..., i] * W[k, i, j]
                                                    * x2[..., j] + b[k]`.

    Lowered via `tract_core_einsum` with the three-operand expression
    `bi, kij, bj -> bk`. tract's `einsum` doesn't accept ellipsis, so
    only rank-2 `x1` / `x2` are supported here.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "bilinear requires `tract_core_einsum` (TractNNEF target)"
        )
    input1_node, input2_node, weight_node, bias_node = node.inputs
    if input1_node.rank != 2 or input2_node.rank != 2:
        raise T2NErrorNotImplemented(
            "bilinear: only rank-2 inputs supported "
            f"(got {input1_node.rank} / {input2_node.rank}); leading "
            "batch dims need a different einsum expr."
        )
    if weight_node.rank != 3:
        raise T2NErrorNotImplemented(
            f"bilinear: weight must be rank 3, got {weight_node.rank}"
        )
    has_bias = (
        bias_node is not None and getattr(bias_node, "data", None) is not None
    )
    inp1 = get_or_add_tensor_variable_in_nnef(g, input1_node, name_to_tensor)
    inp2 = get_or_add_tensor_variable_in_nnef(g, input2_node, name_to_tensor)
    weight = get_or_add_tensor_variable_in_nnef(g, weight_node, name_to_tensor)
    suffix = "_bilinear_einsum" if has_bias else ""
    intermed = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_einsum",
        inputs=[inp1, weight, inp2],
        attrs={"expr": "bi,kij,bj->bk", "acc": "f32", "output": ""},
        ensure_tuple=False,
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix=suffix,
    )
    if has_bias:
        bias = get_or_add_tensor_variable_in_nnef(g, bias_node, name_to_tensor)
        # Bias is (out,); add to (B, out) intermed. Unsqueeze to (1, out)
        # so tract's broadcasting matches torch's (which broadcasts the
        # trailing axis) rather than the leading-axis alignment that
        # tript-IR's auto-rank pass falls back on.
        bias_2d = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "unsqueeze",
            inputs=bias,
            attrs={"axes": [0]},
            output_tensor_name_suffix="_bilinear_bias_2d",
        )
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "add",
            inputs=[intermed, bias_2d],
            force_consistent_inputs_shapes=False,
        )
    return ["tract_core"]


@OP_REGISTRY.register()
def chain_matmul(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::chain_matmul(matrices)` to a chain of `matmul` ops.

    `matrices` is a `FixedTensorList` of `>=2` 2-D tensors. The chain
    is reduced left-to-right; this matches torch's deprecation note
    that recommends `linalg.multi_dot` (which picks a parenthesization
    by cost). For inference graphs the per-matrix shapes are fixed and
    constant-folding handles the planning, so the naive left-fold is
    enough.
    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.torch_graph import FixedTensorList

    (matrices_node,) = node.inputs
    if not isinstance(matrices_node, FixedTensorList):
        raise T2NErrorNotImplemented(
            "aten::chain_matmul expects a FixedTensorList; got "
            f"{matrices_node!r}"
        )
    mats = list(matrices_node.data)
    if len(mats) < 2:
        raise T2NErrorNotImplemented(
            f"aten::chain_matmul expects >= 2 matrices, got {len(mats)}"
        )
    refs = [
        get_or_add_tensor_variable_in_nnef(g, m, name_to_tensor) for m in mats
    ]
    acc = refs[0]
    for idx, rhs in enumerate(refs[1:], start=1):
        is_final = idx == len(refs) - 1
        acc = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "matmul",
            inputs=[acc, rhs],
            attrs={"transposeA": False, "transposeB": False},
            output_tensor_name_suffix=""
            if is_final
            else f"_chain_matmul_{idx}",
        )


@OP_REGISTRY.register(["baddbmm", "addmm", "bias_addmm"])
def baddbmm(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:baddbmm', 'aten:addmm', 'aten:bias_addmm'.

    `bias_addmm` is a dispatcher-fused addmm variant (the `self`
    operand is a broadcasted bias); semantics are identical so we
    route it through the same `addmm` fragment.
    """
    input_node, batch1_node, batch2_node, beta_node, alpha_node = node.inputs
    for ab_node in [alpha_node, beta_node]:
        if isinstance(alpha_node, PythonConstant):
            ab_node.set_data(float(ab_node.data))
        else:
            raise T2NErrorNotImplemented()
    inputs = [
        get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
        for _ in [input_node, batch1_node, batch2_node]
    ]

    new_inputs = []
    target_dtype = node.outputs[0].dtype
    for inp in inputs:
        if NUMPY_TO_TORCH_DTYPE[inp.dtype] != target_dtype and isinstance(
            inference_target, TractNNEF
        ):
            target_dtype_tract = TORCH_DTYPE_TO_TRACT_STR[target_dtype]
            inp = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_cast",
                inputs=inp,
                attrs={"to": target_dtype_tract},
                force_full_output_tensor_name=f"{inp.name}_cast_{target_dtype_tract}",
            )
        new_inputs.append(inp)
    inputs = new_inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "addmm",
        inputs=inputs,
        attrs={"beta": beta_node.data, "alpha": alpha_node.data},
    )
    return ["addmm"]


def _emit_fused_matmul(g, node, name_to_tensor, inference_target, fragment):
    """Shared body for `addbmm` / `addmv` / `addr`.

    All three follow the same `(self, A, B, *, beta, alpha)` aten
    signature as `baddbmm` -- only the fragment that does the actual
    math differs.
    """
    input_node, a_node, b_node, beta_node, alpha_node = node.inputs
    for ab_node in [alpha_node, beta_node]:
        if isinstance(ab_node, PythonConstant):
            ab_node.set_data(float(ab_node.data))
        else:
            raise T2NErrorNotImplemented()
    inputs = [
        get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
        for _ in [input_node, a_node, b_node]
    ]
    new_inputs = []
    target_dtype = node.outputs[0].dtype
    for inp in inputs:
        if NUMPY_TO_TORCH_DTYPE[inp.dtype] != target_dtype and isinstance(
            inference_target, TractNNEF
        ):
            target_dtype_tract = TORCH_DTYPE_TO_TRACT_STR[target_dtype]
            inp = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_cast",
                inputs=inp,
                attrs={"to": target_dtype_tract},
                force_full_output_tensor_name=(
                    f"{inp.name}_cast_{target_dtype_tract}"
                ),
            )
        new_inputs.append(inp)
    # `force_consistent_inputs_shapes=False`: the inputs of these
    # fused-matmul fragments have intentionally different ranks
    # (e.g. addbmm: `input` is `(n, p)`, `batch1` is `(b, n, m)`),
    # so the default rank-alignment pass that unsqueezes the smaller
    # input would corrupt the math.
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        fragment,
        inputs=new_inputs,
        attrs={"beta": beta_node.data, "alpha": alpha_node.data},
        force_consistent_inputs_shapes=False,
    )
    return [fragment]


@OP_REGISTRY.register()
def addbmm(g, node, name_to_tensor, inference_target, **kwargs):
    """aten::addbmm -> `beta*self + alpha*sum_b(bmm(b1, b2))`."""
    return _emit_fused_matmul(
        g, node, name_to_tensor, inference_target, "addbmm"
    )


@OP_REGISTRY.register()
def addmv(g, node, name_to_tensor, inference_target, **kwargs):
    """aten::addmv -> `beta*self + alpha*(mat @ vec)`."""
    return _emit_fused_matmul(
        g, node, name_to_tensor, inference_target, "addmv"
    )


@OP_REGISTRY.register()
def addr(g, node, name_to_tensor, inference_target, **kwargs):
    """aten::addr -> `beta*self + alpha*(vec1 outer vec2)`."""
    return _emit_fused_matmul(g, node, name_to_tensor, inference_target, "addr")


def _make_ntensor(g, name_to_tensor, name: str, shape, np_dtype):
    """Create an `NTensor` with an explicit shape and register it.

    The shared `add_single_output_op` helper inherits the final node's
    shape for every intermediate it emits, which is wrong for
    rank-changing chains (matmul wrapped in unsqueeze + squeeze).
    """
    tensor = NTensor(g, name, dtype=np_dtype, shape=tuple(shape))
    name_to_tensor[name] = tensor
    return tensor


@OP_REGISTRY.register()
def dot(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:dot' to NNEF.

    `torch.dot(a, b)` is the 1-D x 1-D inner product, returning a
    scalar. NNEF's `matmul` requires rank >= 2, so we unsqueeze the
    inputs to (1, K) and (K, 1), matmul, then squeeze the (1, 1) back
    to a scalar.
    """
    a_node, b_node = node.inputs
    onode = node.outputs[0]
    np_dtype = onode.np_dtype
    base = onode.export_name
    k = a_node.shape[0]

    a_ref = get_or_add_tensor_variable_in_nnef(g, a_node, name_to_tensor)
    b_ref = get_or_add_tensor_variable_in_nnef(g, b_node, name_to_tensor)

    a_unsq = _make_ntensor(g, name_to_tensor, f"{base}_dot_a", (1, k), np_dtype)
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="unsqueeze",
        name=f"{a_unsq.name}_op",
        inputs=(a_ref,),
        outputs=(a_unsq,),
        attribs={"axes": [0]},
    )
    b_unsq = _make_ntensor(g, name_to_tensor, f"{base}_dot_b", (k, 1), np_dtype)
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="unsqueeze",
        name=f"{b_unsq.name}_op",
        inputs=(b_ref,),
        outputs=(b_unsq,),
        attribs={"axes": [1]},
    )
    mm_out = _make_ntensor(
        g, name_to_tensor, f"{base}_dot_mm", (1, 1), np_dtype
    )
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="matmul",
        name=f"{mm_out.name}_op",
        inputs=(a_unsq, b_unsq),
        outputs=(mm_out,),
        attribs={"transposeA": False, "transposeB": False},
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=mm_out,
        attrs={"axes": [0, 1]},
    )


@OP_REGISTRY.register()
def mv(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:mv' to NNEF.

    `torch.mv(M, v)` is matrix-vector with `M` rank-2 and `v` rank-1,
    returning a rank-1 result. NNEF `matmul` needs rank-2 on both
    sides, so unsqueeze `v` to (K, 1), matmul to (M, 1), squeeze back.
    """
    m_node, v_node = node.inputs
    onode = node.outputs[0]
    np_dtype = onode.np_dtype
    base = onode.export_name
    m_dim, k_dim = m_node.shape

    m_ref = get_or_add_tensor_variable_in_nnef(g, m_node, name_to_tensor)
    v_ref = get_or_add_tensor_variable_in_nnef(g, v_node, name_to_tensor)

    v_unsq = _make_ntensor(
        g, name_to_tensor, f"{base}_mv_v", (k_dim, 1), np_dtype
    )
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="unsqueeze",
        name=f"{v_unsq.name}_op",
        inputs=(v_ref,),
        outputs=(v_unsq,),
        attribs={"axes": [1]},
    )
    mm_out = _make_ntensor(
        g, name_to_tensor, f"{base}_mv_mm", (m_dim, 1), np_dtype
    )
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="matmul",
        name=f"{mm_out.name}_op",
        inputs=(m_ref, v_unsq),
        outputs=(mm_out,),
        attribs={"transposeA": False, "transposeB": False},
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=mm_out,
        attrs={"axes": [1]},
    )
