from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.tensor.quant import QTensor
from torch_to_nnef_llm.quantize import (
    quantize_dense_projections_q40,
    quantize_lm_head_q40,
)


class _TinyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 64, bias=False)  # 2048 elems
        self.k_proj = nn.Linear(32, 32, bias=False)  # 1024 elems
        self.norm_scale = nn.Parameter(torch.ones(64))  # rank 1


class TinyCausal(nn.Module):
    """Minimal causal-LM-shaped model for the quantization passes.

    Weight in-features are multiples of 32 (tract q4_0 block length);
    the projections are nested (`self_attn.q_proj.weight`) so the
    leading-dot suffix matching mirrors real named_parameters paths.
    """

    def __init__(self, tie_word_embeddings=False):
        super().__init__()
        self.config = SimpleNamespace(
            tie_word_embeddings=tie_word_embeddings
        )
        self.embed_tokens = nn.Embedding(64, 32)
        self.self_attn = _TinyAttn()
        self.lm_head = nn.Linear(32, 64, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def q_proj(self):
        return self.self_attn.q_proj

    @property
    def k_proj(self):
        return self.self_attn.k_proj

    @property
    def norm_scale(self):
        return self.self_attn.norm_scale


def _is_q40(param) -> bool:
    return isinstance(param, QTensor) or isinstance(
        getattr(param, "data", None), QTensor
    )


def test_dense_groups_matched_and_counted():
    model = TinyCausal()
    counts = quantize_dense_projections_q40(
        model,
        {"attn": (".q_proj.weight", ".k_proj.weight")},
        min_numel=1,
    )
    assert counts == {"attn": 2}
    assert _is_q40(model.q_proj.weight)
    assert _is_q40(model.k_proj.weight)
    assert not _is_q40(model.lm_head.weight)


def test_dense_min_numel_threshold_skips():
    model = TinyCausal()
    counts = quantize_dense_projections_q40(
        model,
        {"attn": (".q_proj.weight", ".k_proj.weight")},
        min_numel=2048,
    )
    assert counts == {"attn": 1}
    assert _is_q40(model.q_proj.weight)
    assert not _is_q40(model.k_proj.weight)


def test_dense_ndim_guard_skips_rank1():
    model = TinyCausal()
    counts = quantize_dense_projections_q40(
        model,
        {"norm": (".norm_scale",), "attn": (".q_proj.weight",)},
        min_numel=1,
    )
    # rank-1 norm_scale matched the suffix but was skipped by the guard
    assert counts == {"attn": 1}
    assert not _is_q40(model.norm_scale)


def test_dense_group_subset_selection():
    model = TinyCausal()
    counts = quantize_dense_projections_q40(
        model,
        {"q": (".q_proj.weight",), "k": (".k_proj.weight",)},
        group_names=["q"],
        min_numel=1,
    )
    assert counts == {"q": 1}
    assert not _is_q40(model.k_proj.weight)


def test_dense_unknown_group_raises():
    model = TinyCausal()
    with pytest.raises(T2NErrorMisuse, match="unknown dense quant groups"):
        quantize_dense_projections_q40(
            model,
            {"attn": (".q_proj.weight",)},
            group_names=["attn", "typo"],
        )


def test_dense_nothing_matched_raises():
    model = TinyCausal()
    with pytest.raises(T2NErrorMisuse, match="matched no weight"):
        quantize_dense_projections_q40(
            model,
            {"attn": (".does_not_exist.weight",)},
            min_numel=1,
        )


def test_lm_head_quantized():
    model = TinyCausal()
    quantize_lm_head_q40(model)
    assert _is_q40(model.lm_head.weight)
    assert not _is_q40(model.embed_tokens.weight)


def test_lm_head_tied_embeddings_raises():
    model = TinyCausal(tie_word_embeddings=True)
    with pytest.raises(T2NErrorMisuse, match="tied to embed_tokens"):
        quantize_lm_head_q40(model)
    assert not _is_q40(model.lm_head.weight)
