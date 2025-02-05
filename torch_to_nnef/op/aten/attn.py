"""Attention mechanisms"""

import torch

from torch_to_nnef import inference_target
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.op.fragment import TMPL_FRAGMENTS
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def scaled_dot_product_attention(
    g, node, name_to_tensor, inference_target, **kwargs
):
    """
    reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
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
        raise TorchToNNEFNotImplementedError(
            "scaled_dot_product_attention with > 0 dropout_p not implemented"
        )

    if not isinstance(inference_target, TractNNEF):
        raise TorchToNNEFNotImplementedError(
            "Only support tract since float casting"
            " is important for overflow"
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
    if len(node.inputs) == 7:  # added param between torch 1.13 and 2.2
        scale_node = node.inputs[-1]
        if scale_node.data is not None:
            scale = scale_node.data
            scale_tensor = get_or_add_tensor_variable_in_nnef(
                g, scale_node, name_to_tensor
            )
            inputs.append(scale_tensor)
    is_causal = is_causal_node.data

    has_masked_attn = not isinstance(attn_mask_node, PythonConstant)

    if has_masked_attn:
        if attn_mask_node.dtype not in [torch.float32, torch.float16]:
            raise TorchToNNEFNotImplementedError(
                "scaled_dot_product_attention with attn_mask_node non float not implemented"
            )
        attn_mask_tensor = get_or_add_tensor_variable_in_nnef(
            g, attn_mask_node, name_to_tensor
        )
        inputs.append(attn_mask_tensor)
    else:
        assert attn_mask_node.data is None

    dtype_str = "f32"
    if query_node.dtype == torch.float16:
        dtype_str = "f16"

    tmpl = TMPL_FRAGMENTS["scaled_dot_product_attention"]
    fragment = tmpl.into_concrete_fragment(
        scale=scale,
        causal=is_causal,
        rank=key_node.rank,
        dtype=dtype_str,
        inner_dtype=(
            "f32"
            if inference_target.force_attention_softmax_in_f32
            else dtype_str
        ),
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
