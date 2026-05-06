import math

import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
    weight_bias_and_output_tensor,
)
from torch_to_nnef.tensor.quant import QTensorTract
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def batch_norm(g, node, name_to_tensor, null_ref, inference_target, **kwargs):
    """Translate operator `aten::batch_norm` to NNEF.

    Nnef inputs:.
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
    if isinstance(inference_target, TractNNEF):
        params_nodes = [weight_node, running_mean_node, running_var_node]
        if bias_node.data is not None:
            params_nodes.append(bias_node)
        for param_node in params_nodes:
            if isinstance(param_node.data, QTensorTract):
                raise T2NErrorNotImplemented(
                    "should write unsqueeze within NNEF graph"
                )
            param_node.set_data(param_node.data.unsqueeze(0), force_shape=True)
            for _ in range(input_node.rank - param_node.rank):
                param_node.set_data(
                    param_node.data.unsqueeze(-1), force_shape=True
                )
    # }

    upcast_f32 = (
        isinstance(inference_target, TractNNEF)
        and input_node.dtype == torch.float16
        and inference_target.force_norm_in_f32
    )
    if upcast_f32:
        running_mean_node.set_data(
            running_mean_node.data.float(), force_dtype=True
        )
        running_var_node.set_data(
            running_var_node.data.float(), force_dtype=True
        )
        # output_tensor.dtype is based on weight_node
    weight_ref, bias_ref, output_ref = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
        suffix_out_name="_f32" if upcast_f32 else "",
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

    base_inp_ref = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    custom_fragments = []
    if upcast_f32:
        inp_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=base_inp_ref,
            attrs={"to": "f32"},
            output_tensor_name_suffix="_ucast_f32",
        )
        custom_fragments.append("tract_core")
    else:
        inp_ref = base_inp_ref
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="batch_normalization",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            inp_ref,
            running_mean_ref,
            running_var_ref,
            bias_ref,
            weight_ref,
        ),
        outputs=output_ref,
        attribs={"epsilon": eps_node.data},
    )
    if upcast_f32:
        inp_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=output_ref,
            attrs={"to": TORCH_DTYPE_TO_TRACT_STR[node.outputs[0].dtype]},
        )
    return custom_fragments


@OP_REGISTRY.register(
    ["norm", "linalg_vector_norm", "linalg_norm", "frobenius_norm"]
)
def norm(g, node, name_to_tensor, inference_target, **kwargs):
    """NOTE this is only the normed vector."""
    if node.kind in ["aten::linalg_vector_norm", "aten::linalg_norm"]:
        # new in PyTorch 2.0
        input_node, p_node, axes_node, keep_dim_node, _ = node.inputs
    elif node.kind == "aten::frobenius_norm":
        input_node, axes_node, keep_dim_node = node.inputs
        p_node = PythonConstant(name=f"{node.outputs[0].name}_p_node", data=2)
    else:
        input_node, p_node, axes_node, keep_dim_node = node.inputs
    if p_node.data is None:
        p_node.set_data(2)

    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    upcast_f32 = (
        p_node.data != 1
        and isinstance(inference_target, TractNNEF)
        and inference_target.force_norm_in_f32
        and input_node.dtype == torch.float16
    )
    custom_fragments = []
    if upcast_f32:
        input_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=input_tensor,
            attrs={"to": "f32"},
            output_tensor_name_suffix="_ucast_f32",
        )
        custom_fragments.append("tract_core")

    use_norm_spe_norm = p_node.data in [1, 2]
    order = float(p_node.data)
    custom_fragment_name = (
        f"norm_p{p_node.data}" if use_norm_spe_norm else "norm_pn"
    )
    attrs = {"axes": [pick_axis(input_node, dim) for dim in axes_node.data]}
    if not use_norm_spe_norm:
        assert isinstance(p_node.data, (float, int))
        attrs["ord"] = order
    if order == math.inf:
        custom_fragment_name = "norm_pinf"
        del attrs["ord"]
    elif order == -math.inf:
        custom_fragment_name = "norm_neg_inf"
        del attrs["ord"]
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        custom_fragment_name,
        inputs=input_tensor,
        attrs=attrs,
        output_tensor_name_suffix="_norm"
        if (not keep_dim_node.data or upcast_f32)
        else "",
    )
    if upcast_f32:
        out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=out,
            attrs={"to": "f16"},
            output_tensor_name_suffix=""
            if keep_dim_node.data
            else "_downcast_f16",
        )
    if not keep_dim_node.data:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=out,
            attrs={
                "axes": [pick_axis(input_node, dim) for dim in axes_node.data]
            },
            pass_quantization_params=True,
        )
    return [custom_fragment_name]


@OP_REGISTRY.register(["layer_norm", "native_layer_norm"])
def layer_norm(g, node, name_to_tensor, null_ref, **kwargs):
    """Map PyTorch: 'aten:layer_norm', 'aten:native_layer_norm' to NNEF."""
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
    # has_affine is only true when both weight and bias are real tensors and
    # at least one is non-identity (not all-1 / all-0). `LayerNorm(...,
    # elementwise_affine=False)` leaves both at None: the affine path would
    # then crash trying to materialize None as an NNEF tensor.
    weight_defined = weight_node.data is not None
    bias_defined = bias_node.data is not None
    has_affine = (
        bool(elementwise_affine_node.data)
        and weight_defined
        and bias_defined
        and not (
            (bias_node.data == 0).all().tolist()
            and (weight_node.data == 1).all().tolist()
        )
    )
    inputs = [input_tensor_node]
    op_name = "layer_norm"
    if has_affine:
        op_name = "layer_norm_with_affine"
        inputs += [weight_node, bias_node]
    add_single_output_op(
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


def prefer_native_tract_rms_norm(inference_target, mean_axes) -> bool:
    """Return True when we should emit tract's native rms_norm primitive.

    Native `tract_transformers_rms_norm` is registered through tract's
    transformers extension, which t2n only auto-enables (via the
    `--nnef-tract-transformers` CLI flag) for tract >= 0.22.0. The native
    op also takes a single integer `axis`, so multi-axis
    `normalized_shape` keeps the fragment fallback.
    """
    return (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= "0.22.0"
        and len(mean_axes) == 1
    )


@OP_REGISTRY.register(["rms_norm"])
def rms_norm(g, node, name_to_tensor, inference_target, null_ref, **kwargs):
    """Map PyTorch: 'aten:rms_norm' to NNEF.

    Signature from `torch.nn.functional.rms_norm`:
        `rms_norm(input, normalized_shape, weight, eps)`

    On tract >= 0.22.0 with a single normalized dim, emit the native
    `tract_transformers_rms_norm` op (gives tract access to its optimized
    GPU kernels and rewrite rules) and chain a `mul` for elementwise affine.
    Multi-axis `normalized_shape` and non-tract targets fall back to the
    custom `rms_norm{,_with_affine}` fragments.
    """
    (
        input_tensor_node,
        normalized_shape_node,
        weight_node,
        eps_node,
    ) = node.inputs
    mean_axes = sorted(
        input_tensor_node.rank - r - 1
        for r, _ in enumerate(normalized_shape_node.data)
    )
    eps = 1e-6 if eps_node.data is None else float(eps_node.data)
    weight_defined = weight_node.data is not None
    has_affine = weight_defined and not (weight_node.data == 1).all().tolist()

    if prefer_native_tract_rms_norm(inference_target, mean_axes):
        input_ref = get_or_add_tensor_variable_in_nnef(
            g, input_tensor_node, name_to_tensor
        )
        rms_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="tract_transformers_rms_norm",
            inputs=[input_ref],
            attrs={"axis": mean_axes[0], "eps": eps},
            output_tensor_name_suffix="_rms" if has_affine else "",
        )
        if has_affine:
            weight_ref = get_or_add_tensor_variable_in_nnef(
                g, weight_node, name_to_tensor
            )
            add_single_output_op(
                g,
                node,
                name_to_tensor,
                nnef_op_type="mul",
                inputs=[rms_ref, weight_ref],
            )
        return ["tract_transformers"]

    # Fragment fallback: tract < 0.22.0, multi-axis normalized_shape, or
    # non-tract targets (Khronos etc.).
    inputs = [input_tensor_node]
    op_name = "rms_norm"
    if has_affine:
        op_name = "rms_norm_with_affine"
        inputs += [weight_node]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=op_name,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            for _ in inputs
        ],
        attrs={"mean_axes": mean_axes, "eps": eps},
    )
    return [op_name]


@OP_REGISTRY.register(["group_norm", "native_group_norm"])
def group_norm(g, node, name_to_tensor, inference_target, **kwargs):
    """Translate operators `aten::group_norm` to NNEF.

    Decomposed flow:

    1. Reshape `input` from `(B, C, *spatial)` to `(B, C, S)` where
       `S = prod(spatial)`: the t2n emitter knows the spatial shape
       statically and does the flatten here.
    2. Call the `group_norm` fragment, which works entirely in 3D
       `(B, num_groups, C/num_groups * S)` then projects back to
       `(B, C, S)`. The fragment does NOT apply scale/offset.
    3. Reshape the 3D result back to `(B, C, *spatial)`.
    4. Multiply by `scale` and add `offset`: both pre-unsqueezed to
       trailing-1 shape so NNEF's left-aligned broadcast extends them
       cleanly to the full input rank (this is the same pattern other
       norms use).
    """
    (
        input_node,
        n_groups_node,
        scale_node,
        offset_node,
        eps_node,
        _,  # is_affine_node
    ) = node.inputs
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "use tract_core_cast in 'group_norm' fragment"
        )

    # Pre-unsqueeze scale/offset so NNEF broadcast extends them across
    # the full input rank. After the loop they have shape
    # `(num_channels, 1, 1, ...)` matching `input_node.rank - 1`
    # trailing 1s; NNEF's left-aligned `maybe_align_inputs_ranks` then
    # prepends a single 1 to match the full input rank.
    for nd in [offset_node, scale_node]:
        for _ in range(input_node.rank - nd.rank - 1):
            if isinstance(nd.data, QTensorTract):
                raise T2NErrorNotImplemented(
                    "should write unsqueeze within NNEF graph"
                )
            nd.set_data(nd.data.unsqueeze(-1), force_shape=True)

    upcast_f32 = (
        isinstance(inference_target, TractNNEF)
        and input_node.dtype == torch.float16
        and inference_target.force_norm_in_f32
    )
    if upcast_f32:
        offset_node.set_data(offset_node.data.float(), force_dtype=True)
        scale_node.set_data(scale_node.data.float(), force_dtype=True)

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

    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    custom_fragments = []
    if upcast_f32:
        inp_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=inp_ref,
            attrs={"to": "f32"},
            output_tensor_name_suffix="_ucast_f32",
        )
        custom_fragments.append("tract_core")

    batch_size = input_node.shape[0]
    num_channels = input_node.shape[1]
    flat_spatial = 1
    for d in input_node.shape[2:]:
        flat_spatial *= int(d)
    flat_3d_shape = (batch_size, num_channels, flat_spatial)
    final_shape = tuple(int(d) for d in input_node.shape)
    np_dtype = offset_node.np_dtype if upcast_f32 else input_node.np_dtype

    base = node.outputs[0].export_name

    # Reshape input to 3D `(B, C, S)`.
    input_3d_name = f"{base}_gn_input_3d"
    input_3d_tensor = NTensor(
        g, input_3d_name, dtype=np_dtype, shape=flat_3d_shape
    )
    name_to_tensor[input_3d_name] = input_3d_tensor
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="reshape",
        name=f"{input_3d_name}_op",
        inputs=(inp_ref,),
        outputs=(input_3d_tensor,),
        attribs={"shape": list(flat_3d_shape)},
    )

    # Call the simplified group_norm fragment in 3D.
    custom_fragments.append("group_norm")
    gn_3d_name = f"{base}_gn_norm_3d"
    gn_3d_tensor = NTensor(g, gn_3d_name, dtype=np_dtype, shape=flat_3d_shape)
    name_to_tensor[gn_3d_name] = gn_3d_tensor
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="group_norm",
        name=f"{gn_3d_name}_op",
        inputs=(input_3d_tensor,),
        outputs=(gn_3d_tensor,),
        attribs={
            "epsilon": eps_node.data,
            "num_groups": n_groups_node.data,
            "batch_size": batch_size,
            "num_channels": num_channels,
        },
    )

    # Reshape back to original `(B, C, *spatial)` shape.
    reshaped_name = f"{base}_gn_reshape_back"
    reshaped_tensor = NTensor(
        g, reshaped_name, dtype=np_dtype, shape=final_shape
    )
    name_to_tensor[reshaped_name] = reshaped_tensor
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="reshape",
        name=f"{reshaped_name}_op",
        inputs=(gn_3d_tensor,),
        outputs=(reshaped_tensor,),
        attribs={"shape": list(final_shape)},
    )

    # Apply scale.
    scaled_name = f"{base}_gn_scaled"
    scaled_tensor = NTensor(g, scaled_name, dtype=np_dtype, shape=final_shape)
    name_to_tensor[scaled_name] = scaled_tensor
    cast_and_add_nnef_operation(
        graph=g,
        name_to_tensor=name_to_tensor,
        type="mul",
        name=f"{scaled_name}_op",
        inputs=(reshaped_tensor, scale_ref),
        outputs=(scaled_tensor,),
        attribs={},
    )

    # Apply offset; `add_single_output_op` wires the last op to
    # `node.outputs[0]` (the model's output).
    out_ref = add_single_output_op(
        g=g,
        name_to_tensor=name_to_tensor,
        node=node,
        nnef_op_type="add",
        inputs=(scaled_tensor, offset_ref),
        output_tensor_name_suffix="_f32" if upcast_f32 else "",
    )
    if upcast_f32:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=out_ref,
            attrs={"to": "f16"},
        )
    return custom_fragments


@OP_REGISTRY.register()
def _weight_norm(g, node, name_to_tensor, inference_target, **kwargs):
    """https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/WeightNorm.cpp#L82.

    Formulation:
        v * (g / norm(v, 2, dim=dim_node))

    Note:
        this is a form of unit norm with scale g
    """
    (
        vin_node,
        gin_node,
        dim_node,
    ) = node.inputs

    assert isinstance(dim_node.data, int)

    upcast_f32 = (
        isinstance(inference_target, TractNNEF)
        and vin_node.dtype == torch.float16
        and inference_target.force_norm_in_f32
    )
    custom_fragments = ["weight_norm"]
    inp_ref = get_or_add_tensor_variable_in_nnef(g, vin_node, name_to_tensor)
    if upcast_f32:
        gin_node.set_data(gin_node.data.float(), force_dtype=True)
        inp_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=inp_ref,
            attrs={"to": "f32"},
            output_tensor_name_suffix="upcast_f32",
        )
        custom_fragments.append("tract_core")

    out_ref = add_single_output_op(
        g=g,
        name_to_tensor=name_to_tensor,
        node=node,
        nnef_op_type="weight_norm",
        inputs=(
            inp_ref,
            get_or_add_tensor_variable_in_nnef(g, gin_node, name_to_tensor),
        ),
        attrs={
            "axes": [
                i for i, _ in enumerate(vin_node.shape) if i != dim_node.data
            ],
        },
        output_tensor_name_suffix="_f32" if upcast_f32 else "",
    )
    if upcast_f32:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=out_ref,
            attrs={"to": "f16"},
        )
    return custom_fragments
