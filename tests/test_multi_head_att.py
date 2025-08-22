"""Tests attention variants."""

import copy
import math
import os
from functools import partial

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.utils import torch_version

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    combine_conditions,
    set_seed,
)
from .wrapper import TernaryPrimitive

set_seed(int(os.environ.get("SEED", 0)))


def causal_supported_condition(i: InferenceTarget) -> bool:
    return isinstance(i, TractNNEF) and i.version >= "0.21.4"


def approx_supported_condition(i: InferenceTarget) -> bool:
    return isinstance(i, TractNNEF) and i.version >= "0.21.7"


def sdpa_supported_condition(i: InferenceTarget) -> bool:
    return isinstance(i, TractNNEF) and i.version >= "0.22.0"


def tract_f16_friendly_condition(i: InferenceTarget) -> bool:
    return (
        isinstance(i, TractNNEF)
        and approx_supported_condition(i)
        and i.force_attention_inner_in_f32
    )


# Enabling f32 upcasting in inner attention computation is required
# to avoid overflows and obtain closer results to PyTorch.
# Tolerance needs to be relaxed (we prioritize efficiency over strict Pytorch alignement).
def enable_attention_inner_f32(target: TractNNEF) -> TractNNEF:
    target.force_attention_inner_in_f32 = True
    target.check_io_tolerance = TractCheckTolerance.VERY
    return target


# Enabling SDPA to cover export to tract_transformers_sdpa operator
def reify_sdpa_operator(target: TractNNEF) -> TractNNEF:
    target.reify_sdpa_operator = True
    return target


defaults = TRACT_INFERENCES_TO_TESTS_APPROX
defaults_f16_friendly = [
    enable_attention_inner_f32(copy.deepcopy(t))
    for t in TRACT_INFERENCES_TO_TESTS_APPROX
    if approx_supported_condition(t)
]
sdpa = [
    reify_sdpa_operator(copy.deepcopy(t))
    for t in defaults_f16_friendly
    if sdpa_supported_condition(t)
]
test_suite = TestSuiteInferenceExactnessBuilder(
    defaults + defaults_f16_friendly + sdpa
)

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
if hasattr(F, "scaled_dot_product_attention"):
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
    inference_conditions=causal_supported_condition,
)

test_suite.add(
    inp,
    FScaledDotProdAttn(is_causal=True, scale=1.62),
    inference_conditions=causal_supported_condition,
)

# 4d
for as_f16 in [False, True]:
    enabled_conditions = []
    if as_f16:
        # Only enable f16 if:
        #  - approx is supported by the inference_target
        #  - inner_f32 is enabled in the inference_target
        enabled_conditions.append(tract_f16_friendly_condition)

    inp = torch.rand((1, 2, 3, 4)).float()
    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16),
        inference_conditions=combine_conditions(enabled_conditions),
    )
    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, scale=1.3),
        inference_conditions=combine_conditions(enabled_conditions),
    )
    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, scale=0.3),
        inference_conditions=combine_conditions(enabled_conditions),
    )
    test_suite.add(
        inp,
        FScaledDotProdAttn(
            as_f16=as_f16, scale=0.3, attn_mask=torch.rand((1, 2, 3, 3))
        ),
        inference_conditions=combine_conditions(enabled_conditions),
    )
    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, attn_mask=torch.rand((1, 2, 3, 3))),
        inference_conditions=combine_conditions(enabled_conditions),
    )

    # Only enable is_causal if inference target supports is_causal.
    enabled_conditions.append(causal_supported_condition)
    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, is_causal=True),
        inference_conditions=combine_conditions(enabled_conditions),
    )

    test_suite.add(
        inp,
        FScaledDotProdAttn(as_f16=as_f16, is_causal=True, scale=1.62),
        inference_conditions=combine_conditions(enabled_conditions),
    )


def _simulate_scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p=0.0):
    attn_weight = torch.softmax(
        (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1
    )
    return attn_weight @ V


@pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason="F.scaled_dot_product_attention is only appearing in Pytorch>=2",
)
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


@pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason="F.scaled_dot_product_attention is only appearing in Pytorch>=2",
)
@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_attn_layers_export(
    id, test_input, model, inference_target, pytestconfig
):
    """Test attention mechanisms"""
    if (
        not pytestconfig.getvalue("--run-experimental")
        and isinstance(inference_target, TractNNEF)
        and inference_target.reify_sdpa_operator
    ):
        pytest.skip("reify_sdpa_operator activated only by --run-experimental")
        return

    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
