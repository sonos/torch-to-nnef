import contextlib
import importlib
import re
import tarfile

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
    """The shim's internal GQA repeat must reproduce HF's result exactly.

    The handler feeds the shim UN-repeated q/k (hk heads), while HF repeats
    q/k to the value-head count before calling the rule.
    """
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
    """Substituting the shim for the chunked variant must agree numerically.

    The chunked variant computes the same recurrence with different blocking.
    """
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
    """The registered families must keep sharing ONE reference implementation.

    If HF diverges the sources, the per-family tests above could both pass
    while the shared shim silently mismatches a third caller, so fail loudly
    here to force a review.
    """
    import inspect

    mods = []
    for path in GDN_REFERENCE_MODULES:
        with contextlib.suppress(ImportError):
            mods.append(importlib.import_module(path))
    if len(mods) < 2:
        pytest.skip("fewer than two GDN families in installed transformers")
    ref = mods[0]
    conv_update_name = (
        "torch_causal_conv1d_update"
        if hasattr(ref, "torch_causal_conv1d_update")
        else "causal_conv1d_update"
    )
    for other in mods[1:]:
        for fn_name in (
            "torch_recurrent_gated_delta_rule",
            "torch_chunk_gated_delta_rule",
            conv_update_name,
        ):
            assert inspect.getsource(
                getattr(ref, fn_name)
            ) == inspect.getsource(getattr(other, fn_name)), (
                f"{fn_name} diverged between {ref.__name__} "
                f"and {other.__name__}"
            )


class _CachedGdnExport(torch.nn.Module):
    """Exercise the real HF decode path with explicit cache inputs/outputs."""

    def __init__(self, gdn, config):
        super().__init__()
        self.gdn = gdn
        self.config = config

    def forward(self, hidden, conv_state, recurrent_state):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.config)
        cache.update_conv_state(conv_state, 0)
        cache.update_recurrent_state(recurrent_state, 0)
        output = self.gdn(hidden, cache_params=cache)
        layer = cache.layers[0]
        conv = layer.conv_states
        recurrent = layer.recurrent_states
        # Transformers has used both tensor and list-backed cache slots.
        if not isinstance(conv, torch.Tensor):
            conv = conv[0]
        if not isinstance(recurrent, torch.Tensor):
            recurrent = recurrent[0]
        return output, conv, recurrent


@pytest.mark.parametrize(
    "family, config_name, module_name, neutralize_gqa_repeat",
    [
        (
            "qwen3_5_moe",
            "Qwen3_5MoeTextConfig",
            "Qwen3_5MoeGatedDeltaNet",
            True,
        ),
        ("qwen3_5", "Qwen3_5TextConfig", "Qwen3_5GatedDeltaNet", True),
        ("qwen3_next", "Qwen3NextConfig", "Qwen3NextGatedDeltaNet", False),
    ],
)
def test_real_gdn_exports_reified_operations(
    family, config_name, module_name, neutralize_gqa_repeat
):
    """A successful export must contain the fused ops, not an unrolled rule."""
    from tests.utils import (
        TRACT_INFERENCES_TO_TESTS_APPROX,
        check_model_io_test,
    )

    from torch_to_nnef_llm.models.handlers.gdn_reify import (
        reify_gated_delta_net,
    )

    modeling = pytest.importorskip(
        f"transformers.models.{family}.modeling_{family}"
    )
    configs = pytest.importorskip(
        f"transformers.models.{family}.configuration_{family}"
    )
    if not TRACT_INFERENCES_TO_TESTS_APPROX:
        pytest.skip("tract checks disabled")
    target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    if target.version < "0.23.10":
        pytest.skip("GDN export regression requires tract >= 0.23.10")
    target = target.with_dynamic_axes({"input_0": {1: "S"}})
    config = getattr(configs, config_name)(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    torch.manual_seed(0)
    gdn = getattr(modeling, module_name)(config, 0).eval()
    reify_gated_delta_net(gdn, neutralize_gqa_repeat=neutralize_gqa_repeat)

    def assert_reified_operations(_target, export_path):
        with tarfile.open(export_path) as archive:
            graph_file = archive.extractfile("graph.nnef")
            assert graph_file is not None
            graph = graph_file.read().decode("utf8")
        # Inspect calls in the serialized graph, not fragment declarations.
        graph_body = re.split(r"\bgraph\s+\w+\s*\(", graph, maxsplit=1)[1]
        for op in (
            "tract_transformers_gdn_recurrent",
            "tract_transformers_causal_conv1d_update",
        ):
            assert len(re.findall(rf"=\s*{op}\s*\(", graph_body)) == 1, graph

    check_model_io_test(
        _CachedGdnExport(gdn, config),
        (
            torch.randn(1, 1, 32),
            torch.randn(1, 64, 4),
            torch.randn(1, 4, 8, 8),
        ),
        target,
        callback_post_export=assert_reified_operations,
    )
