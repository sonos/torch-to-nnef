"""Tests simple primitives."""

import math
import os
from functools import partial

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from torch_to_nnef.inference_target import TractNNEF

from .test_primitive import TernaryPrimitive
from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 0)))

test_suite = TestSuiteInferenceExactnessBuilder(TRACT_INFERENCES_TO_TESTS_APPROX)

# NOTE: More than >= 16 heads seems to leads to precision differences between Tract/PyTorch
n_heads = 8
hidden_dim = 256  # 4  # KO: 768 or
keys = torch.randint(2, (1, 2, hidden_dim)).float()
values = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float()
queries = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float() * 2
test_suite.add(
    (keys, values, queries),
    torch.nn.MultiheadAttention(
        hidden_dim, num_heads=n_heads, dropout=0.0, batch_first=True
    ),
    test_name="8_heads_attn",
)


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

test_suite.add(
    torch.rand((100, 1, 64)).float(),
    SelfAttn(need_weights=True),
)
test_suite.add(
    (Xmini, Xmini, Xmini),
    TernaryPrimitive(
        partial(
            F.scaled_dot_product_attention,
            attn_mask=torch.randint(high=1, size=(2, 1)).float(),
        )
    ),
)
test_suite.add(
    torch.randint(high=5, size=(3, 1, 4)).float(),
    SelfAttn(size=4, batch_size=3, need_weights=False),
)


class FScaledDotProdAttn(torch.nn.Module):
    def __init__(
        self, is_causal=False, scale=None, attn_mask=None, as_f16=False
    ):
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale
        self.attn_mask = (
            attn_mask.to(torch.float16 if as_f16 else torch.float32)
            if attn_mask is not None
            else None
        )
        self.as_f16 = as_f16

    def forward(self, x):
        # query: Tensor,
        # key: Tensor,
        # value: Tensor,
        # attn_mask: Optional[Tensor] = None,
        # dropout_p: float = 0.0,
        # is_causal: bool = False,
        # scale: Optional[float] = None
        if self.as_f16:
            x = x.to(torch.float16)
        res = F.scaled_dot_product_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=self.attn_mask,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        if self.as_f16:
            res = res.to(torch.float32)
        return res


# 3d
inp = torch.rand((1, 2, 3)).float()
test_suite.add(inp, FScaledDotProdAttn())
test_suite.add(inp, FScaledDotProdAttn(scale=1.3))
test_suite.add(inp, FScaledDotProdAttn(scale=0.3))
test_suite.add(
    inp, FScaledDotProdAttn(scale=0.3, attn_mask=torch.rand((1, 2, 2)))
)
test_suite.add(inp, FScaledDotProdAttn(attn_mask=torch.rand((1, 2, 2))))

test_suite.add(
    inp,
    FScaledDotProdAttn(is_causal=True),
    inference_conditions=lambda i: isinstance(i, TractNNEF)
    and i.version >= "0.21.4",
)

test_suite.add(
    inp,
    FScaledDotProdAttn(is_causal=True, scale=1.62),
    inference_conditions=lambda i: isinstance(i, TractNNEF)
    and i.version >= "0.21.4",
)
# 4d
for as_f16 in [False]:  # True works but difference in precision
    inp = torch.rand((1, 2, 3, 4)).float()
    test_suite.add(inp, FScaledDotProdAttn(as_f16=as_f16))
    test_suite.add(inp, FScaledDotProdAttn(as_f16=as_f16, scale=1.3))
    test_suite.add(inp, FScaledDotProdAttn(as_f16=as_f16, scale=0.3))
    test_suite.add(
        inp,
        FScaledDotProdAttn(
            as_f16=as_f16, scale=0.3, attn_mask=torch.rand((1, 2, 3, 3))
        ),
    )
    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, attn_mask=torch.rand((1, 2, 3, 3))),
    )

    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, is_causal=True),
        inference_conditions=lambda i: isinstance(i, TractNNEF)
        and i.version >= "0.21.4",
    )

    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, is_causal=True, scale=1.62),
        inference_conditions=lambda i: isinstance(i, TractNNEF)
        and i.version >= "0.21.4",
    )


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


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_attn_layers_export(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
