"""Tests for MoE FFN export to tract_moe_ffn operator."""

import os
import tarfile
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.op.custom_extractors import MoEFFN
from torch_to_nnef.tensor.quant import (
    fp_to_tract_q4_0_with_min_max_calibration,
)

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    check_model_io_test,
    skipif_unsupported_qtensor,
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
    # tract_moe_ffn first ships in tract 0.23.4; releases 0.23.0..0.23.3 are
    # already out without it, so default CI (official versions) must skip.
    # An explicitly provided tract (T2N_TEST_TRACT_PATH /
    # T2N_TEST_TRACT_VERSION) is trusted to have the op regardless of its
    # reported version, so the locally built dev binary (e.g. 0.23.2-pre)
    # still runs these.
    if not isinstance(inference_target, TractNNEF):
        pytest.skip("MoE export requires a tract inference target")
    explicit = (
        "T2N_TEST_TRACT_PATH" in os.environ
        or "T2N_TEST_TRACT_VERSION" in os.environ
    )
    if not explicit and inference_target.version < "0.23.4":
        pytest.skip(
            "tract_moe_ffn first ships in tract 0.23.4; set "
            "T2N_TEST_TRACT_PATH to a build that has the op to run these"
        )


def _init_moe_weights(module, seed=0):
    """Initialize a freshly built MoE block.

    transformers MoE experts allocate weights with `torch.empty` (uninitialized
    garbage) and rely on the model's `_init_weights`. A standalone block skips
    that, so we seed small finite values to get a meaningful reference.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        # Small finite values for every parameter, including biases, so the
        # bias paths are exercised numerically rather than left at zero.
        for p in module.parameters():
            nn.init.normal_(p, std=0.02)


def _read_graph_from_archive(path):
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            if member.name.endswith("graph.nnef"):
                return tf.extractfile(member).read().decode("utf-8")
    raise AssertionError("graph.nnef not found in NNEF archive")


def _assert_moe_expert_weights_q40(
    inference_target,
    path,
    expected_count,
    expected_shapes=None,
):
    """Check split expert tensors were exported as tract Q40 values."""
    if not isinstance(inference_target, TractNNEF):
        return
    expected_dtype = (
        DatBinHeader.TractCustomTypes.Q40
        if inference_target.version >= "0.21.11"
        else DatBinHeader.TractCustomTypes.Q40_LEGACY
    )
    with tempfile.TemporaryDirectory() as td, tarfile.open(path, "r:*") as tf:
        members = [
            m
            for m in tf.getmembers()
            if m.name.endswith(("_w1.dat", "_w2.dat", "_w3.dat"))
        ]
        assert len(members) == expected_count, [m.name for m in members]
        for member in members:
            tf.extract(member, td)
            header = DatBinHeader.from_dat(Path(td) / member.name)
            assert header.torch_dtype_or_custom == expected_dtype
            if expected_shapes is not None:
                suffix = member.name.rsplit("_", 1)[-1].removesuffix(".dat")
                assert header.dims == expected_shapes[suffix]


def _assert_moe_expert_weight_shapes(
    inference_target,
    path,
    expected_count,
    expected_shapes,
):
    """Check split expert tensor shapes independently of their storage dtype."""
    if not isinstance(inference_target, TractNNEF):
        return
    with tempfile.TemporaryDirectory() as td, tarfile.open(path, "r:*") as tf:
        members = [
            m
            for m in tf.getmembers()
            if m.name.endswith(("_w1.dat", "_w2.dat", "_w3.dat"))
        ]
        assert len(members) == expected_count, [m.name for m in members]
        for member in members:
            tf.extract(member, td)
            header = DatBinHeader.from_dat(Path(td) / member.name)
            suffix = member.name.rsplit("_", 1)[-1].removesuffix(".dat")
            assert header.dims == expected_shapes[suffix]


def _assert_graph_contains(inference_target, path, fragment):
    if not isinstance(inference_target, TractNNEF):
        return
    graph = _read_graph_from_archive(path)
    assert fragment in graph


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


@skipif_unsupported_qtensor
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_split_experts_q40(inference_target):
    """Export MoEFFN with split expert tensors quantized to tract Q40."""
    _skip_if_unsupported(inference_target)
    export_target = deepcopy(inference_target)
    export_target.check_io = False
    model = MoEFFNWrapper(num_experts=4, d_model=32, d_hidden=64, k=2)
    model.moe._t2n_quantize_moe_experts_q40 = True
    model.eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 32),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=export_target,
        callback_post_export=partial(
            _assert_moe_expert_weights_q40,
            expected_count=2,
        ),
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_split_experts_linear_layout(inference_target):
    """Export MoE experts in native linear-filter layout."""
    _skip_if_unsupported(inference_target)
    model = MoEFFNWrapper(num_experts=4, d_model=32, d_hidden=64, k=2)
    model.moe._t2n_moe_expert_layout = "linear"
    model.eval()

    def _assert_linear_layout(inference_target, path):
        _assert_graph_contains(
            inference_target,
            path,
            "expert_layout = 'linear'",
        )
        _assert_moe_expert_weight_shapes(
            inference_target,
            path,
            expected_count=2,
            expected_shapes={
                "w1": [4, 64, 32],
                "w2": [4, 32, 64],
            },
        )

    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 32),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
        callback_post_export=_assert_linear_layout,
    )


@skipif_unsupported_qtensor
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_split_experts_q40_linear_layout(inference_target):
    """Export Q40 experts after independently selecting linear layout."""
    _skip_if_unsupported(inference_target)
    export_target = deepcopy(inference_target)
    export_target.check_io = False
    model = MoEFFNWrapper(num_experts=4, d_model=32, d_hidden=64, k=2)
    model.moe._t2n_quantize_moe_experts_q40 = True
    model.moe._t2n_moe_expert_layout = "linear"
    model.eval()

    def _assert_linear_q40(inference_target, path):
        _assert_graph_contains(
            inference_target,
            path,
            "expert_layout = 'linear'",
        )
        _assert_moe_expert_weights_q40(
            inference_target,
            path,
            expected_count=2,
            expected_shapes={
                "w1": [4, 64, 32],
                "w2": [4, 32, 64],
            },
        )

    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 32),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=export_target,
        callback_post_export=_assert_linear_q40,
    )


@skipif_unsupported_qtensor
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_custom_q40_quantizer(inference_target):
    """Allow callers to provide a calibrated Q40 quantizer for MoE experts."""
    _skip_if_unsupported(inference_target)
    export_target = deepcopy(inference_target)
    export_target.check_io = False
    calls = []

    def quantizer(tensor, marker):
        calls.append((tuple(tensor.shape), marker, tensor.is_contiguous()))
        return fp_to_tract_q4_0_with_min_max_calibration(tensor)

    model = MoEFFNWrapper(num_experts=4, d_model=32, d_hidden=64, k=2)
    model.moe._t2n_quantize_moe_experts_q40 = True
    model.moe._t2n_moe_expert_layout = "linear"
    model.moe._t2n_quantize_moe_experts_q40_quantizer = quantizer
    model.moe._t2n_quantize_moe_experts_q40_kwargs = {"marker": "calibrated"}
    model.eval()

    check_model_io_test(
        model=model,
        test_input=(torch.randn(8, 32),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=export_target,
    )

    assert calls == [
        ((4, 64, 32), "calibrated", True),
        ((4, 32, 64), "calibrated", True),
    ]


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
    mlp = gpt_oss.GptOssMLP(cfg)
    _init_moe_weights(mlp)
    model = _GptOssMoEWrapper(mlp.eval()).eval()
    # GptOssMLP expects a 3D [batch, seq, hidden] input.
    check_model_io_test(
        model=model,
        test_input=(torch.randn(1, 6, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


class _BlockWrapper(nn.Module):
    """Wrap a transformers MoE block returning a single hidden-state tensor."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        out = self.block(x)
        # Older blocks return (hidden_states, router_logits); keep hidden only.
        return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_qwen3(inference_target):
    """Export a tiny Qwen3 MoE block (concatenated gate/up, plain SwiGLU)."""
    _skip_if_unsupported(inference_target)
    qwen3 = pytest.importorskip(
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        reason="transformers too old for qwen3-moe",
    )
    cfg = qwen3.Qwen3MoeConfig(
        hidden_size=24,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
        # real Qwen3-MoE checkpoints renormalize the top-k gates
        norm_topk_prob=True,
    )
    block = qwen3.Qwen3MoeSparseMoeBlock(cfg)
    _init_moe_weights(block)
    model = _BlockWrapper(block.eval()).eval()
    # The block expects a 3D [batch, seq, hidden] input.
    check_model_io_test(
        model=model,
        test_input=(torch.randn(1, 5, 24),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_mixtral(inference_target):
    """Export a tiny Mixtral MoE block (the canonical fused MoE)."""
    _skip_if_unsupported(inference_target)
    mixtral = pytest.importorskip(
        "transformers.models.mixtral.modeling_mixtral",
        reason="transformers too old for mixtral",
    )
    cfg = mixtral.MixtralConfig(
        hidden_size=16,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    block = mixtral.MixtralSparseMoeBlock(cfg)
    _init_moe_weights(block)
    model = _BlockWrapper(block.eval()).eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(1, 5, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_olmoe(inference_target):
    """Export a tiny OLMoE block (Qwen-like fused experts, no shared expert)."""
    _skip_if_unsupported(inference_target)
    olmoe = pytest.importorskip(
        "transformers.models.olmoe.modeling_olmoe",
        reason="transformers too old for olmoe",
    )
    cfg = olmoe.OlmoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        # OLMoE-1B-7B ships with norm_topk_prob=False -> "softmax_all" gating.
        norm_topk_prob=False,
    )
    block = olmoe.OlmoeSparseMoeBlock(cfg)
    _init_moe_weights(block)
    model = _BlockWrapper(block.eval()).eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(1, 5, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_moe_ffn_qwen2_shared_expert(inference_target):
    """Export a tiny Qwen2 MoE block with its always-on shared expert."""
    _skip_if_unsupported(inference_target)
    qwen2 = pytest.importorskip(
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        reason="transformers too old for qwen2-moe",
    )
    cfg = qwen2.Qwen2MoeConfig(
        hidden_size=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    block = qwen2.Qwen2MoeSparseMoeBlock(cfg)
    _init_moe_weights(block)
    model = _BlockWrapper(block.eval()).eval()
    check_model_io_test(
        model=model,
        test_input=(torch.randn(1, 5, 16),),
        input_names=["tokens"],
        output_names=["output"],
        inference_target=inference_target,
    )
