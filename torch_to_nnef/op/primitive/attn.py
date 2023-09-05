""" Attention mechanisms """

import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def scaled_dot_product_attention(g, node, name_to_tensor, **kwargs):
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
    ) = node.inputs
    if is_causal_node.data is True:
        raise TorchToNNEFNotImplementedError(
            "scaled_dot_product_attention with causal not implemented"
        )

    if dropout_p_node.data != 0.0:
        raise TorchToNNEFNotImplementedError(
            "scaled_dot_product_attention with > 0 dropout_p not implemented"
        )
    if attn_mask_node.dtype != torch.float32:
        raise TorchToNNEFNotImplementedError(
            "scaled_dot_product_attention with attn_mask_node non float not implemented"
        )

    query_tensor = get_or_add_tensor_variable_in_nnef(
        g, query_node, name_to_tensor
    )
    key_tensor = get_or_add_tensor_variable_in_nnef(g, key_node, name_to_tensor)
    value_tensor = get_or_add_tensor_variable_in_nnef(
        g, value_node, name_to_tensor
    )
    fragment_name = ""
    if key_node.rank == 3:
        fragment_name = "scaled_dot_product_attention_3d"
    elif key_node.rank == 4:
        fragment_name = "scaled_dot_product_attention_4d"
    else:
        raise TorchToNNEFNotImplementedError(
            "shape unexpected for scaled_dot_product_attention"
        )

    attn_mask_tensor = get_or_add_tensor_variable_in_nnef(
        g, attn_mask_node, name_to_tensor
    )

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        fragment_name,
        inputs=(query_tensor, key_tensor, value_tensor, attn_mask_tensor),
        # attrs={"axes": [pick_rank(input_node, dim) for dim in axes_node.data]},
    )

    return [fragment_name]
