import importlib

import pytest
import torch

from torch_to_nnef.op.custom_extractors.gdn import (
    GatedDeltaNetRecurrentReified,
)

#: HF families shipping the shared fla-style torch gated-delta-rule
#: reference implementations the reified shim must match. Each entry is
#: skipped independently when the installed transformers does not ship it.
GDN_REFERENCE_MODULES = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)


@pytest.fixture(
    params=GDN_REFERENCE_MODULES,
    ids=[path.rsplit(".", 1)[-1] for path in GDN_REFERENCE_MODULES],
)
def gdn_ref(request):
    return pytest.importorskip(request.param)


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
def test_shim_matches_hf_recurrent_rule(gdn_ref, s_len, with_state):
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
    ref_core, ref_state = gdn_ref.torch_recurrent_gated_delta_rule(
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


@pytest.mark.parametrize("s_len", [1, 16])
def test_shim_gqa_matches_hf_repeated_rule(gdn_ref, s_len):
    """The handler feeds the shim UN-repeated q/k (hk heads); HF repeats
    q/k to the value-head count before calling the rule. The shim's
    internal repeat must reproduce HF's result exactly."""
    groups = 2
    inputs = _rand_inputs(1, s_len, 4, 32, with_state=True)
    # keep every 'groups'-th q/k head: repeat_interleave of the kept heads
    # reconstructs the reference tensors exactly
    q_small = inputs["query"][:, :, ::groups].contiguous()
    k_small = inputs["key"][:, :, ::groups].contiguous()
    q_ref = q_small.repeat_interleave(groups, dim=2)
    k_ref = k_small.repeat_interleave(groups, dim=2)
    shim = GatedDeltaNetRecurrentReified()
    core, state = shim(
        q_small,
        k_small,
        inputs["value"],
        g=inputs["g"],
        beta=inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ref_core, ref_state = gdn_ref.torch_recurrent_gated_delta_rule(
        q_ref,
        k_ref,
        inputs["value"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(core, ref_core, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state, ref_state, atol=1e-5, rtol=1e-4)


def test_shim_matches_hf_chunked_rule(gdn_ref):
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
    ref_core, ref_state = gdn_ref.torch_chunk_gated_delta_rule(
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


def test_reference_rule_sources_stay_in_sync():
    """The registered families must keep sharing ONE reference torch
    implementation: if HF diverges the sources, the per-family tests above
    could both pass while the shared shim silently mismatches a third
    caller, so fail loudly here to force a review."""
    import inspect

    mods = []
    for path in GDN_REFERENCE_MODULES:
        try:
            mods.append(importlib.import_module(path))
        except ImportError:
            pass
    if len(mods) < 2:
        pytest.skip("fewer than two GDN families in installed transformers")
    ref = mods[0]
    for other in mods[1:]:
        for fn_name in (
            "torch_recurrent_gated_delta_rule",
            "torch_chunk_gated_delta_rule",
            "torch_causal_conv1d_update",
        ):
            assert inspect.getsource(
                getattr(ref, fn_name)
            ) == inspect.getsource(getattr(other, fn_name)), (
                f"{fn_name} diverged between {ref.__name__} "
                f"and {other.__name__}"
            )
