"""Assignment-based export caches retain the installed HF update semantics."""

import pytest
import torch

from torch_to_nnef_llm.models.handlers.gdn_reify import (
    patch_linear_cache_for_export,
)

cache_utils = pytest.importorskip("transformers.cache_utils")
if not hasattr(cache_utils, "LinearAttentionLayer"):
    pytest.skip("transformers has no linear cache", allow_module_level=True)

# Keep the upstream implementations before runtime tests prepare export models.
_HF_CONV_UPDATE = cache_utils.LinearAttentionLayer.update_conv_state
_HF_RECURRENT_UPDATE = cache_utils.LinearAttentionLayer.update_recurrent_state


@pytest.mark.parametrize("chunk_lengths", [(4, 1, 2), (4, 5, 1), (2, 1, 5)])
def test_assignment_cache_matches_hf_without_mutating_inputs(chunk_lengths):
    reference = cache_utils.LinearAttentionLayer()
    exported = cache_utils.LinearAttentionLayer()
    patch_linear_cache_for_export()
    generator = torch.Generator().manual_seed(7)
    fed_inputs = []
    for length in chunk_lengths:
        conv = torch.randn(1, 3, length, generator=generator)
        original = conv.clone()
        fed_inputs.append((conv, original))
        expected = _HF_CONV_UPDATE(reference, conv.clone())
        actual = exported.update_conv_state(conv)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(conv, original)
        ref_stored = reference.conv_states
        stored = exported.conv_states
        if isinstance(stored, list):
            ref_stored, stored = ref_stored[0], stored[0]
        torch.testing.assert_close(stored, ref_stored)
        state = torch.randn(1, 2, 3, 3, generator=generator)
        expected_state = _HF_RECURRENT_UPDATE(reference, state.clone())
        actual_state = exported.update_recurrent_state(state)
        torch.testing.assert_close(actual_state, expected_state)
    for conv, original in fed_inputs:
        torch.testing.assert_close(conv, original)
