import pytest
import torch

from torch_to_nnef_llm.models.handlers import (
    Qwen3NextArchitectureHandler,
    Qwen35MoeArchitectureHandler,
)
from torch_to_nnef_llm.models.handlers.registry import get_handler
from torch_to_nnef.op.custom_extractors.gdn import (
    GatedDeltaNetRecurrentReified,
)

qwen3_next = pytest.importorskip(
    "transformers.models.qwen3_next.modeling_qwen3_next"
)


def _tiny_model(linear_num_key_heads=2, linear_num_value_heads=4):
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        layer_types=["full_attention", "linear_attention"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen3NextForCausalLM(config).eval()


def _prefill_then_decode_logits(model):
    """Run a cached prefill + one-token decode, returning decode logits.

    The decode step (seq_len == 1 with previous state) exercises BOTH
    reified paths: the recurrent gated delta rule and the causal conv
    update."""
    from transformers.cache_utils import DynamicCache

    prompt = torch.tensor([[1, 4, 5, 6]], dtype=torch.long)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=prompt, past_key_values=cache, use_cache=True)
        out = model(
            input_ids=torch.tensor([[7]], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
    return out.logits


def test_qwen3_next_handler_is_registered():
    assert get_handler("qwen3_next") is Qwen3NextArchitectureHandler
    assert issubclass(
        Qwen3NextArchitectureHandler, Qwen35MoeArchitectureHandler
    )


def test_prepare_swaps_rules_but_keeps_num_k_heads():
    """Qwen3-Next reads num_k_heads in fix_query_key_value_ordering at
    every forward, so the GQA neutralization MUST stay off for it."""
    model = _tiny_model(linear_num_key_heads=2, linear_num_value_heads=4)
    Qwen3NextArchitectureHandler().prepare_model_for_export(model)

    gdn_modules = [
        m
        for m in model.modules()
        if isinstance(m, qwen3_next.Qwen3NextGatedDeltaNet)
    ]
    assert gdn_modules, "tiny model must contain a GDN layer"
    for m in gdn_modules:
        assert isinstance(
            m.recurrent_gated_delta_rule, GatedDeltaNetRecurrentReified
        )
        assert m.chunk_gated_delta_rule is m.recurrent_gated_delta_rule
        assert callable(m.causal_conv1d_update)
        # unlike Qwen3.5, num_k_heads must NOT be aliased to num_v_heads
        assert m.num_k_heads == 2
        assert m.num_v_heads == 4


def test_reified_qwen3_next_decode_matches_hf_eager():
    """Prefill + cached decode logits must match the unpatched HF model
    (the GQA path 2 key heads vs 4 value heads included)."""
    model = _tiny_model(linear_num_key_heads=2, linear_num_value_heads=4)
    ref_logits = _prefill_then_decode_logits(model)

    Qwen3NextArchitectureHandler().prepare_model_for_export(model)
    reified_logits = _prefill_then_decode_logits(model)

    torch.testing.assert_close(
        reified_logits, ref_logits, atol=1e-4, rtol=1e-4
    )


def test_qwen3_next_moe_block_has_registered_extractor():
    """The MoE blocks of the tiny model must dispatch through the
    tract_moe_ffn extractor machinery (adapter + registered extractor)."""
    from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
    from torch_to_nnef.op.custom_extractors.moe import _get_adapter

    model = _tiny_model()
    moe_blocks = [
        m
        for m in model.modules()
        if type(m).__name__ == "Qwen3NextSparseMoeBlock"
    ]
    assert moe_blocks, "tiny model must contain a sparse MoE block"
    for block in moe_blocks:
        adapter = _get_adapter(block)
        assert adapter.top_k(block) == 2
        assert adapter.shared_expert(block) is not None
        assert ModuleInfoExtractor.get_by_module(block) is not None
