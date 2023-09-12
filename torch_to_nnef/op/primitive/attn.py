""" Attention mechanisms """

import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

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

    query_tensor = get_or_add_tensor_variable_in_nnef(
        g, query_node, name_to_tensor
    )
    key_tensor = get_or_add_tensor_variable_in_nnef(g, key_node, name_to_tensor)
    value_tensor = get_or_add_tensor_variable_in_nnef(
        g, value_node, name_to_tensor
    )

    fragment_suffix_id = ""
    has_masked_attn = not isinstance(attn_mask_node, PythonConstant)

    inputs = [query_tensor, key_tensor, value_tensor]
    if has_masked_attn:
        if attn_mask_node.dtype != torch.float32:
            raise TorchToNNEFNotImplementedError(
                "scaled_dot_product_attention with attn_mask_node non float not implemented"
            )
        attn_mask_tensor = get_or_add_tensor_variable_in_nnef(
            g, attn_mask_node, name_to_tensor
        )
        inputs.append(attn_mask_tensor)
    else:
        assert attn_mask_node.data is None
        fragment_suffix_id = "_nomask"

    fragment_name = ""
    if key_node.rank == 3:
        fragment_name = f"scaled_dot_product_attention_3d{fragment_suffix_id}"
    elif key_node.rank == 4:
        fragment_name = f"scaled_dot_product_attention_4d{fragment_suffix_id}"
    else:
        raise TorchToNNEFNotImplementedError(
            "shape unexpected for scaled_dot_product_attention"
        )

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        fragment_name,
        inputs=tuple(inputs),
        # attrs={"axes": [pick_rank(input_node, dim) for dim in axes_node.data]},
    )

    return [fragment_name]
