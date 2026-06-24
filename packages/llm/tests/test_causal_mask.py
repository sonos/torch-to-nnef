"""Non-regression tests for the export wrapper's explicit causal mask.

Covers the ``force_causal_mask`` path (transformers > 4.52.4).

Regression context: the mask was built from ``in_cache_key_0.shape[0]`` (the
*batch* dim = 1), giving a degenerate ``[1, 1]`` mask that applies no causal
masking and no past offset. Exported LLMs then attended non-causally during
prefill -> garbage / empty generation in real (causal) inference, while the
single-step IO check did not catch it.

The mask must be ``[1, 1, S, S+P]`` (query length S, key length S+P) and
causal-with-past: query i (absolute position P+i) attends to keys ``0..P+i``.
"""

from types import SimpleNamespace

import torch

from torch_to_nnef_llm.models.handlers.default import (
    DefaultArchitectureHandler,
)

NEG = torch.finfo(torch.float32).min


def _build_mask(seq_len, past_len, n_layers=2, heads=2, head_dim=8):
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    past = []
    for _ in range(n_layers):
        # in_cache_key / in_cache_value: [batch, heads, past_len, head_dim]
        past.append(torch.zeros(1, heads, past_len, head_dim))
        past.append(torch.zeros(1, heads, past_len, head_dim))
    inputs = (input_ids, *past)
    wrapper = SimpleNamespace(force_causal_mask=True, with_dyn_cache=False)
    out = DefaultArchitectureHandler().build_forward_inputs(
        inputs=inputs, wrapper=wrapper
    )
    return out["attention_mask"]


def _assert_causal(mask, seq_len, past_len):
    assert mask is not None
    assert tuple(mask.shape) == (1, 1, seq_len, seq_len + past_len), (
        f"mask shape {tuple(mask.shape)} != "
        f"{(1, 1, seq_len, seq_len + past_len)}"
    )
    m = mask[0, 0]
    for i in range(seq_len):
        for j in range(seq_len + past_len):
            allowed = j <= past_len + i  # query i attends keys 0..P+i
            if allowed:
                assert m[i, j].item() == 0.0, (i, j, m[i, j].item())
            else:
                assert m[i, j].item() == NEG, (i, j, m[i, j].item())


def test_causal_mask_prefill_no_past():
    """Prefill (P=0): standard lower-triangular causal mask [S, S]."""
    _assert_causal(_build_mask(seq_len=5, past_len=0), seq_len=5, past_len=0)


def test_causal_mask_with_past():
    """Decode/prefill-with-past (P>0): [S, S+P] with the past offset."""
    _assert_causal(_build_mask(seq_len=4, past_len=3), seq_len=4, past_len=3)


def test_causal_mask_single_token_decode():
    """Single new token over a long past attends to everything before it."""
    mask = _build_mask(seq_len=1, past_len=6)
    assert tuple(mask.shape) == (1, 1, 1, 7)
    assert (mask[0, 0, 0] == 0.0).all()  # the new token sees all 7 keys
