from .gdn_reify import reify_gated_delta_net
from .qwen35_moe import Qwen35MoeArchitectureHandler
from .registry import register_handler


@register_handler
class Qwen3NextArchitectureHandler(Qwen35MoeArchitectureHandler):
    """Handler for Qwen3-Next hybrid models (GDN + full attention).

    Qwen3-Next is the modular source Qwen3.5-MoE derives from in HF
    transformers, so the whole hybrid-cache handler transfers verbatim:

    - `Qwen3NextGatedDeltaNet` binds the same module-level
      `recurrent_gated_delta_rule` / `chunk_gated_delta_rule` /
      `causal_conv1d_update` functions with identical signatures and
      tensor layouts (`torch_recurrent_gated_delta_rule(query, key,
      value, g, beta, initial_state, output_final_state,
      use_qk_l2norm_in_kernel)` on `[b, S, h, w]` tensors,
      `torch_causal_conv1d_update(hidden_states, conv_state, weight,
      bias, activation)` on `[b, C, S]` / `[b, C, k]`).
    - The hybrid cache uses the same `cache_utils.LinearAttentionLayer`
      (`layer_types` mixes "linear_attention" / "full_attention") and
      the same `linear_*` config fields drive the state shapes.

    ONE deliberate difference: the GQA q/k repeat neutralization stays
    OFF. Qwen3-Next fuses its input projections (in_proj_qkvz /
    in_proj_ba) and `fix_query_key_value_ordering` reads `num_k_heads`
    at every forward, so aliasing `num_k_heads = num_v_heads` (the
    Qwen3.5 trick) would corrupt the qkvz split. The reified shim
    instead receives HF's already-repeated q/k; the exported op resolves
    GQA groups to 1, which is correct but materializes the (activation
    sized, weight-free) broadcast in the graph.
    """

    ARCH_NAMES = ("qwen3_next",)

    def prepare_model_for_export(self, model) -> None:
        reify_gated_delta_net(model, neutralize_gqa_repeat=False)
