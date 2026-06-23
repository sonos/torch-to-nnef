"""Tests for MoE FFN export to tract_moe_ffn operator."""

import pytest
import torch
from torch import nn

from torch_to_nnef.op.custom_extractors import MoEFFN

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    check_model_io_test,
    cond_tract_gt_0_22_0,
)


class MoEFFNWrapper(nn.Module):
    def __init__(self, num_experts, d_model, d_hidden, k=2, activation="silu"):
        super().__init__()
        self.moe = MoEFFN(
            num_experts=num_experts,
            d_model=d_model,
            d_hidden=d_hidden,
            k=k,
            activation=activation,
        )

    def forward(self, x):
        return self.moe(x)


class MoEFFNWithBiasWrapper(nn.Module):
    def __init__(self, num_experts, d_model, d_hidden, k=2):
        super().__init__()
        self.moe = MoEFFN(
            num_experts=num_experts,
            d_model=d_model,
            d_hidden=d_hidden,
            k=k,
            bias=True,
        )

    def forward(self, x):
        return self.moe(x)


def _skip_if_unsupported(inference_target):
    if not cond_tract_gt_0_22_0(inference_target):
        pytest.skip(
            "MoE export requires tract > 0.22.0; skipping for official releases"
        )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_basic(inference_target):
    """Export MoEFFN with 4 experts, top-2."""
    _skip_if_unsupported(inference_target)
    model = MoEFFNWrapper(num_experts=4, d_model=16, d_hidden=32, k=2)
    model.eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_with_bias(inference_target):
    """Export MoEFFN with bias terms."""
    _skip_if_unsupported(inference_target)
    model = MoEFFNWithBiasWrapper(num_experts=4, d_model=16, d_hidden=32, k=2)
    model.eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("activation", ["silu", "gelu", "relu"])
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_activations(inference_target, activation):
    """Export MoEFFN with different activations."""
    _skip_if_unsupported(inference_target)
    model = MoEFFNWrapper(
        num_experts=4, d_model=16, d_hidden=32, k=1, activation=activation
    )
    model.eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_top1(inference_target):
    """Export MoEFFN with top-1 routing."""
    _skip_if_unsupported(inference_target)
    model = MoEFFNWrapper(num_experts=8, d_model=32, d_hidden=64, k=1)
    model.eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(16, 32),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


class _GptOssMoEWrapper(nn.Module):
    """Wrap a single GptOssMLP block, returning only routed hidden states.

    gpt-oss exercises the op's extra features: a router bias, fused
    interleaved gate/up projections with biases, and the clamped SwiGLU
    activation (alpha / limit / (up + 1)).
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        # GptOssMLP returns (hidden_states, router_scores); router_scores is
        # discarded at inference, mirroring transformers' decoder layer.
        return self.mlp(x)[0]


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_gpt_oss(inference_target):
    """Export a tiny gpt-oss MoE block (biases + clamped SwiGLU)."""
    _skip_if_unsupported(inference_target)
    gpt_oss = pytest.importorskip(
        "transformers.models.gpt_oss.modeling_gpt_oss",
        reason="transformers too old for gpt-oss",
    )
    if not hasattr(gpt_oss, "GptOssMLP"):
        pytest.skip("this transformers version has no GptOssMLP")

    cfg = gpt_oss.GptOssConfig(
        hidden_size=16,
        intermediate_size=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )
    mlp = gpt_oss.GptOssMLP(cfg).eval()
    model = _GptOssMoEWrapper(mlp).eval()
    # GptOssMLP expects a 3D [batch, seq, hidden] input.
    check_model_io_test(
        model=model,
        test_input=(torch.randn(1, 6, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )
