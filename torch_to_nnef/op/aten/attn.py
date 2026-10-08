"""Attention mechanisms."""

import numpy as np
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.op.fragment import TMPL_FRAGMENTS
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


def reify_with_tract_transformers_sdpa(i: InferenceTarget) -> bool:
    return (
        isinstance(i, TractNNEF)
        and i.version >= "0.22.0"
        and i.reify_sdpa_operator
    )


def _broadcast_mask_query_axis(
    op_helper, attn_mask_node, attn_mask_tensor, query_node
):
    """Expand an additive attention mask along the query axis.

    PyTorch broadcasts the attention mask against the `(.., L, S)` score
    matrix, so masks derived from a key-padding mask carry a size-1 query
    axis (e.g. `[B, H, 1, S]`). tract's `tract_transformers_sdpa`
    (FlashSDPA) does not broadcast that axis at eval time and raises an
    incompatible-shape error, so tile the mask up to the query length when
    both dims are statically known.
    """
    mask_shape = attn_mask_node.shape
    if mask_shape is None or len(mask_shape) < 2:
        return attn_mask_tensor
    mask_q = mask_shape[-2]
    query_len = query_node.shape[-2]
    if not (isinstance(mask_q, int) and isinstance(query_len, int)):
        return attn_mask_tensor
    if mask_q != 1 or query_len <= 1:
        return attn_mask_tensor
    repeats = [1] * len(mask_shape)
    repeats[-2] = query_len
    new_shape = list(mask_shape)
    new_shape[-2] = query_len
    return op_helper.add_intermediate_op(
        src=attn_mask_tensor,
        op_type="tile",
        attrs={"repeats": repeats},
        new_shape=new_shape,
        suffix="sdpa_mask_qbroadcast",
    )


def _emit_tract_cast(g, src: NTensor, name: str, to: str, dtype) -> NTensor:
    """Emit a tract_core_cast intermediate with an explicit dtype."""
    out = NTensor(g, name=name, dtype=dtype, shape=tuple(src.shape))
    NOperation(
        g,
        type="tract_core_cast",
        attribs={"to": to},
        inputs=src,
        outputs=out,
    )
    return out


def _cast_sdpa_input_to_f32(g, src: NTensor, suffix: str) -> NTensor:
    if src.dtype == np.float32:
        return src
    return _emit_tract_cast(
        g,
        src,
        f"{src.name}_{suffix}",
        "f32",
        np.float32,
    )


@OP_REGISTRY.register()
# pylint: disable-next=too-many-branches
def scaled_dot_product_attention(
    g, node, name_to_tensor, inference_target, **kwargs
):
    """Translate operator: `aten::scaled_dot_product_attention` to NNEF.

    reference:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    (
        query_node,
        key_node,
        value_node,
        attn_mask_node,
        dropout_p_node,
        is_causal_node,
        *_,
    ) = node.inputs

    if dropout_p_node.data != 0.0:
        raise T2NErrorNotImplemented(
            "scaled_dot_product_attention with > 0 dropout_p not implemented"
        )

    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "Only support tract since: "
            " type casting is need, "
            " and getting shape of tensor is important too "
        )

    query_tensor = get_or_add_tensor_variable_in_nnef(
        g, query_node, name_to_tensor
    )
    key_tensor = get_or_add_tensor_variable_in_nnef(g, key_node, name_to_tensor)
    value_tensor = get_or_add_tensor_variable_in_nnef(
        g, value_node, name_to_tensor
    )

    inputs = [query_tensor, key_tensor, value_tensor]

    scale = None
    reify_tract_spda = reify_with_tract_transformers_sdpa(inference_target)
    if len(node.inputs) >= 7:  # added param between torch 1.13 and 2.2
        scale_node = node.inputs[6]
        if scale_node.data is not None:
            scale = scale_node.data

            # If we export with tract >= 0.22.0 with reify_sdpa_operator,
            # scale is expressed as an attribute
            # so we don't need to add it to the list of input.
            if not reify_tract_spda:
                scale_tensor = get_or_add_tensor_variable_in_nnef(
                    g, scale_node, name_to_tensor
                )
                inputs.append(scale_tensor)

    is_causal = is_causal_node.data

    has_masked_attn = not isinstance(attn_mask_node, PythonConstant)

    if has_masked_attn:
        attn_mask_tensor = get_or_add_tensor_variable_in_nnef(
            g, attn_mask_node, name_to_tensor
        )
        if reify_tract_spda:
            attn_mask_tensor = _broadcast_mask_query_axis(
                kwargs["op_helper"],
                attn_mask_node,
                attn_mask_tensor,
                query_node,
            )
        inputs.append(attn_mask_tensor)
    else:
        assert attn_mask_node.data is None

    dtype_str = "f32"
    if query_node.dtype == torch.float16:
        dtype_str = "f16"
    inner_dtype = (
        "f32" if inference_target.force_attention_inner_in_f32 else dtype_str
    )

    if reify_tract_spda:
        sdpa_inputs = list(inputs)
        sdpa_dtype_str = dtype_str
        cast_sdpa_output_back = False
        if (
            inference_target.force_attention_inner_in_f32
            and inference_target.upcast_reified_sdpa_inputs_to_f32
            and query_node.dtype == torch.float16
        ):
            sdpa_inputs[0] = _cast_sdpa_input_to_f32(
                g, sdpa_inputs[0], "sdpa_q_f32"
            )
            sdpa_inputs[1] = _cast_sdpa_input_to_f32(
                g, sdpa_inputs[1], "sdpa_k_f32"
            )
            sdpa_inputs[2] = _cast_sdpa_input_to_f32(
                g, sdpa_inputs[2], "sdpa_v_f32"
            )
            if has_masked_attn and len(sdpa_inputs) > 3:
                sdpa_inputs[3] = _cast_sdpa_input_to_f32(
                    g, sdpa_inputs[3], "sdpa_mask_f32"
                )
            sdpa_dtype_str = "f32"
            cast_sdpa_output_back = True

        # Define SDPA attributes
        attrs = {
            "datum_type": sdpa_dtype_str,
            "acc_datum_type": inner_dtype,
            "is_causal": is_causal,
        }
        if scale is not None:
            attrs["scale"] = scale

        if cast_sdpa_output_back:
            output_node = node.outputs[0]
            sdpa_out = NTensor(
                g,
                name=f"{output_node.export_name}_sdpa_f32",
                dtype=np.float32,
                shape=tuple(output_node.shape),
            )
            NOperation(
                g,
                type="tract_transformers_sdpa",
                attribs=attrs,
                inputs=tuple(sdpa_inputs),
                outputs=sdpa_out,
            )
            add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_cast",
                inputs=sdpa_out,
                attrs={"to": dtype_str},
            )
        else:
            add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_transformers_sdpa",
                inputs=tuple(sdpa_inputs),
                attrs=attrs,
            )
        return ["tract_transformers"]

    tmpl_fragment_name = "scaled_dot_product_attention"
    if inference_target.version < "0.21.11":
        tmpl_fragment_name = f"legacy_{tmpl_fragment_name}"
    tmpl = TMPL_FRAGMENTS[tmpl_fragment_name]
    fragment = tmpl.into_concrete_fragment(
        scale=scale,
        causal=is_causal,
        rank=key_node.rank,
        dtype=dtype_str,
        inner_dtype=inner_dtype,
        attn_mask=has_masked_attn,
    )

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        fragment.name,
        inputs=tuple(inputs),
    )

    return [fragment]
