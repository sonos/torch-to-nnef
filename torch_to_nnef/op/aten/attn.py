"""Attention mechanisms."""

import torch

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


@OP_REGISTRY.register()
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
        # Define SDPA attributes
        attrs = {
            "datum_type": dtype_str,
            "acc_datum_type": inner_dtype,
            "is_causal": is_causal,
        }
        if scale is not None:
            attrs["scale"] = scale

        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_transformers_sdpa",
            inputs=tuple(inputs),
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
