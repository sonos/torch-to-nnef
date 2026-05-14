import typing as T

import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import NUMPY_TO_TORCH_DTYPE, TORCH_DTYPE_TO_TRACT_STR
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
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
):
    """Emit NNEF `conv` / `deconv` for the convolution family.

    Shared body of `aten::_convolution` and `aten::conv_transpose{1,2,3}d`.
    Emits `deconv` for `transposed=True`, else `conv`. Transposed convs
    need a weight repack: torch stores
    `(in_channels, out_channels // groups, *spatial)`; NNEF's `deconv`
    expects `(in_channels // groups, out_channels, *spatial)`. The
    grouped-then-transposed reshape is in-place on `weight_node.data`.
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
        _,  # output_padding_name
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
        _,  # output_padding -- not propagated to NNEF deconv
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


@OP_REGISTRY.register(
    torch_op_ids=["matmul", "bmm", "mm"]
)  # since NNEF matmul does not care about rank
def matmul(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:matmul', 'aten:bmm', 'aten:mm' to NNEF."""
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
