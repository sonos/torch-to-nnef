import pytest
import torch

from torch_to_nnef.op.custom_extractors.gdn import (
    GatedDeltaNetRecurrentReified,
)

qwen35 = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
)


def _rand_inputs(b, s_len, heads, width, seed=0, with_state=True):
    gen = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.rand(*shape, generator=gen) * 2.0 - 1.0

    return dict(
        query=r(b, s_len, heads, width),
        key=r(b, s_len, heads, width),
        value=r(b, s_len, heads, width),
        g=-r(b, s_len, heads).abs(),
        beta=r(b, s_len, heads).sigmoid(),
        initial_state=r(b, heads, width, width) * 0.1 if with_state else None,
    )


@pytest.mark.parametrize("s_len", [1, 16])
@pytest.mark.parametrize("with_state", [True, False])
def test_shim_matches_hf_recurrent_rule(s_len, with_state):
    inputs = _rand_inputs(1, s_len, 4, 32, with_state=with_state)
    shim = GatedDeltaNetRecurrentReified()
    core, state = shim(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        g=inputs["g"],
        beta=inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ref_core, ref_state = qwen35.torch_recurrent_gated_delta_rule(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(core, ref_core, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state, ref_state, atol=1e-5, rtol=1e-4)


def test_shim_matches_hf_chunked_rule():
    """The chunked variant computes the same recurrence with different
    blocking; substituting the shim for it must agree numerically."""
    inputs = _rand_inputs(1, 16, 4, 32, with_state=True)
    shim = GatedDeltaNetRecurrentReified()
    core, state = shim(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        g=inputs["g"],
        beta=inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ref_core, ref_state = qwen35.torch_chunk_gated_delta_rule(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(core, ref_core, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(state, ref_state, atol=2e-4, rtol=1e-3)
