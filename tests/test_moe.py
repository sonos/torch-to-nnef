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
