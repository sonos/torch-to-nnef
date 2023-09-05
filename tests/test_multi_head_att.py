"""Tests simple primitives."""

import math
import os
from functools import partial

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from .test_primitive import TernaryPrimitive
from .utils import check_model_io_test, set_seed  # noqa: E402

INPUT_AND_MODELS = []
set_seed(int(os.environ.get("SEED", 0)))

# NOTE: More than 2 head seems to fail
# hidden_dim = 256  # 4  # KO: 768 or
# n_heads = 16  # 2  # KO: 12 or
# keys = torch.randint(2, (1, 2, hidden_dim)).float()
# values = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float()
# queries = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float() * 2
# INPUT_AND_MODELS += [
#     ((keys, values, queries), op)
#     for op in [
#         torch.nn.MultiheadAttention(
#             hidden_dim, num_heads=n_heads, dropout=0.0, batch_first=True
#         )
#     ]
# ]


class SelfAttn(torch.nn.Module):
    def __init__(self, size=64, batch_size=100, need_weights=False):
        super().__init__()
        self.need_weights = need_weights
        self.selfattn = torch.nn.MultiheadAttention(
            size, num_heads=2, dropout=0.0, batch_first=False
        )
        self.key_padding_mask = torch.ones((1, batch_size)).float()

    def forward(self, x):
        if self.need_weights:
            y = self.selfattn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self.key_padding_mask,
                need_weights=self.need_weights,
            )
        else:
            y, _ = self.selfattn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self.key_padding_mask,
                need_weights=self.need_weights,
            )
        return y


X = torch.rand((100, 1, 64)).float()
Xmini = torch.randint(high=8, size=(1, 2, 4)).float()

INPUT_AND_MODELS = [
    (
        torch.rand((100, 1, 64)).float(),
        SelfAttn(need_weights=True),
    ),
    (
        (Xmini, Xmini, Xmini),
        TernaryPrimitive(
            partial(
                F.scaled_dot_product_attention,
                attn_mask=torch.randint(high=1, size=(2, 1)).float(),
            )
        ),
    ),
    (
        torch.randint(high=5, size=(3, 1, 4)).float(),
        SelfAttn(size=4, batch_size=3, need_weights=False),
    ),
]


def _simulate_scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p=0.0):
    attn_weight = torch.softmax(
        (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1
    )
    return attn_weight @ V


def test_equivalent_implementation():
    val = torch.arange(8.0).reshape(1, 2, 4).float()
    query = val
    key = val
    value = val
    attn_mask = torch.ones((1, 2, 1))
    reference_result = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )

    torch_sim_result = _simulate_scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0
    )
    np.testing.assert_almost_equal(
        reference_result.numpy(), torch_sim_result.numpy()
    )


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_primitive_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
