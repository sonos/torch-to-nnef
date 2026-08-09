import logging
import typing as T

import torch

from .base import IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler

LOGGER = logging.getLogger(__name__)


@register_handler
class Qwen35MoeArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Qwen3.5 MoE text models with hybrid attention caches."""

    ARCH_NAMES = ("qwen3_5_moe", "qwen3_5_moe_text")

    @staticmethod
    def _materialize_offloaded(module, attr) -> int:
        """Replace a disk-offloaded param with its real tensor.

        The GDN path uses these small params in trace-time arithmetic
        (`F.conv1d` on the conv weight, `A_log.exp()`, `dt_bias` add)
        that the offload meta-tracing strategy does not support; they
        are a few KB per layer, so materializing them is free.
        """
        # pylint: disable-next=import-outside-toplevel
        import torch

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.tensor.offload import OffloadedTensor

        tensor = getattr(module, attr, None)
        if tensor is None:
            return 0
        inner = (
            tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor
        )
        if not isinstance(inner, OffloadedTensor):
            return 0
        real = inner.reload()
        if real.dtype != inner.dtype:
            real = real.to(inner.dtype)
        param = torch.nn.Parameter(real, requires_grad=False)
        setattr(module, attr, param)
        return 1

    @staticmethod
    def _patch_linear_cache_for_export() -> None:
        """Make HF linear-attention cache updates assignment-based.

        `LinearAttentionLayer.update_conv_state` / `update_recurrent_state`
        `copy_` into lazily created buffers (to keep cudagraph static
        addresses). Under the opaque fake-tensor tracing strategy the
        model's activation stream is fake while those buffers are real,
        and an IN-PLACE op cannot absorb a fake source (jit tracer
        internal asserts); plain rebinds trace cleanly and are
        semantically identical for export.
        """
        # pylint: disable-next=import-outside-toplevel
        import torch
        # pylint: disable-next=import-outside-toplevel
        from transformers import cache_utils

        layer_cls = cache_utils.LinearAttentionLayer
        if getattr(layer_cls, "_t2n_assignment_cache_patch", False):
            return

        def update_conv_state(
            self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs
        ):
            if not self.is_conv_states_initialized[state_idx]:
                kernel = (
                    conv_states.shape[-1]
                    if conv_kernel_size is None
                    else conv_kernel_size
                )
                self.conv_kernel_size[state_idx] = kernel
                self.is_conv_states_initialized[state_idx] = True
            kernel = self.conv_kernel_size[state_idx]
            if not self.has_previous_state[state_idx]:
                full = conv_states
                self.has_previous_state[state_idx] = True
                if not self.record_past and full.shape[-1] < kernel:
                    full = torch.nn.functional.pad(
                        full, (kernel - full.shape[-1], 0), value=0
                    )
            else:
                full = torch.cat(
                    [self.conv_states[state_idx], conv_states], dim=-1
                )
            stored = (
                full[..., -kernel:] if not self.record_past else full
            )
            # Backref so the reified conv update can REBIND the cache slot
            # instead of mutating the stored tensor in place (which would
            # both corrupt the fed example input through the view above
            # and reintroduce the in-place real/fake mixing).
            stored._t2n_cache_slot = (self, state_idx)
            self.conv_states[state_idx] = stored
            return full

        def update_recurrent_state(
            self, recurrent_states, state_idx=0, **kwargs
        ):
            self.is_recurrent_states_initialized[state_idx] = True
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states

        layer_cls.update_conv_state = update_conv_state
        layer_cls.update_recurrent_state = update_recurrent_state
        layer_cls._t2n_assignment_cache_patch = True
        LOGGER.info(
            "patched LinearAttentionLayer cache updates to assignment form"
        )

    def prepare_model_for_export(self, model) -> None:
        """Reify the gated delta rule as one tract op.

        HF computes the linear-attention core with a Python loop over the
        sequence axis; traced, it unrolls at the traced length and the
        export is frozen to it. Swapping the module-bound rule functions
        with `GatedDeltaNetRecurrentReified` keeps the exported graph
        S-generic (see torch_to_nnef.op.custom_extractors.gdn).
        """
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.custom_extractors.gdn import (
            CausalConvUpdateReified,
            GatedDeltaNetRecurrentReified,
        )

        self._patch_linear_cache_for_export()

        def make_conv_update(conv_shim):
            def causal_conv1d_update(
                hidden_states, conv_state, weight, bias=None, activation=None
            ):
                if bias is not None:
                    raise NotImplementedError(
                        "reified causal conv does not support a bias"
                    )
                if activation not in (None, "silu"):
                    raise NotImplementedError(
                        f"reified causal conv is silu-only, got {activation}"
                    )
                out, final_state = conv_shim(hidden_states, conv_state, weight)
                # The state write stays OUTSIDE the reified boundary.
                # Preferred: REBIND the cache slot (no in-place op, no
                # real/fake mixing under the opaque tracing strategy).
                # Fallback for an unpatched cache: plain copy_ (the aten
                # copy handler resolves it).
                slot = getattr(conv_state, "_t2n_cache_slot", None)
                if slot is not None:
                    layer, state_idx = slot
                    final_state._t2n_cache_slot = slot
                    layer.conv_states[state_idx] = final_state
                else:
                    conv_state.copy_(final_state)
                return out

            return causal_conv1d_update

        n_reified = 0
        n_materialized = 0
        n_gqa = 0
        for module in model.modules():
            if hasattr(module, "recurrent_gated_delta_rule") and hasattr(
                module, "chunk_gated_delta_rule"
            ):
                shim = GatedDeltaNetRecurrentReified()
                module.recurrent_gated_delta_rule = shim
                module.chunk_gated_delta_rule = shim
                # Neutralize HF's pre-rule q/k repeat_interleave (GQA):
                # `forward` only repeats when num_v_heads // num_k_heads > 1
                # and num_k_heads has no other post-__init__ reader, so
                # overriding it makes the repeat an identity. The shim then
                # receives hk-head q/k at the traced boundary (the exported
                # graph never materializes the broadcast; the tract op does
                # the group indexing) and repeats internally so its eager
                # math stays HF-exact.
                if getattr(module, "num_k_heads", None) is not None and (
                    module.num_v_heads != module.num_k_heads
                ):
                    module.num_k_heads = module.num_v_heads
                    n_gqa += 1
                conv_shim = CausalConvUpdateReified()
                module.t2n_conv_update_shim = conv_shim
                module.causal_conv1d_update = make_conv_update(conv_shim)
                n_reified += 1
                for owner, attr in (
                    (getattr(module, "conv1d", None), "weight"),
                    (getattr(module, "conv1d", None), "bias"),
                    (module, "A_log"),
                    (module, "dt_bias"),
                ):
                    if owner is not None:
                        n_materialized += self._materialize_offloaded(
                            owner, attr
                        )
        LOGGER.info(
            "reified the gated delta rule and causal conv in %d "
            "linear-attention modules (%d small offloaded params "
            "materialized, %d GQA q/k repeats neutralized)",
            n_reified,
            n_materialized,
            n_gqa,
        )

    @staticmethod
    def _layer_types(config_helper) -> T.Sequence[str]:
        return config_helper.decoder_conf.layer_types

    @staticmethod
    def _linear_conv_shape(decoder_conf) -> T.Tuple[int, int, int]:
        key_dim = (
            decoder_conf.linear_key_head_dim * decoder_conf.linear_num_key_heads
        )
        value_dim = (
            decoder_conf.linear_value_head_dim
            * decoder_conf.linear_num_value_heads
        )
        conv_dim = key_dim * 2 + value_dim
        return (1, conv_dim, decoder_conf.linear_conv_kernel_dim)

    @staticmethod
    def _linear_recurrent_shape(decoder_conf) -> T.Tuple[int, int, int, int]:
        return (
            1,
            decoder_conf.linear_num_value_heads,
            decoder_conf.linear_key_head_dim,
            decoder_conf.linear_value_head_dim,
        )

    @staticmethod
    def _linear_state_tensor(state):
        if isinstance(state, dict):
            if len(state) != 1:
                raise ValueError(
                    "expected a single Qwen3.5 MoE linear cache state, "
                    f"got {len(state)}"
                )
            return next(iter(state.values()))
        return state

    def build_input_spec(
        self,
        *,
        tokenizer,
        config_helper,
        inputs_dtype: torch.dtype,
        sample_text: str,
        n_input_tokens: int,
        n_past_input_tokens: int,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
    ) -> IOSpec:
        if real_kv_cache is not None:
            raise NotImplementedError(
                "real cache reuse is not supported for "
                "Qwen3.5 MoE hybrid caches"
            )

        test_input = tokenizer(sample_text, return_tensors="pt")
        assert test_input.input_ids.shape[1] >= n_input_tokens

        decoder_conf = config_helper.decoder_conf
        inputs = [test_input.input_ids[:, :n_input_tokens]]
        input_names = ["input_ids"]
        output_names = ["outputs"]
        dynamic_axes = {"input_ids": {1: "S"}}

        for layer_idx, layer_type in enumerate(
            self._layer_types(config_helper)
        ):
            if layer_type == "full_attention":
                key_shape = (
                    1,
                    config_helper.get_num_kv_heads(layer_idx),
                    n_past_input_tokens,
                    config_helper.get_head_dim(),
                )
                value_shape = key_shape
                key = torch.rand(key_shape).to(inputs_dtype)
                value = torch.rand(value_shape).to(inputs_dtype)
                inputs.extend([key, value])
                for kind in ("key", "value"):
                    in_name = f"in_cache_{kind}_{layer_idx}"
                    input_names.append(in_name)
                    output_names.append(f"out_cache_{kind}_{layer_idx}")
                    dynamic_axes[in_name] = {2: "P"}
            elif layer_type == "linear_attention":
                conv = torch.zeros(
                    self._linear_conv_shape(decoder_conf),
                    dtype=inputs_dtype,
                )
                recurrent = torch.zeros(
                    self._linear_recurrent_shape(decoder_conf),
                    dtype=inputs_dtype,
                )
                inputs.extend([conv, recurrent])
                input_names.extend(
                    [
                        f"in_cache_conv_{layer_idx}",
                        f"in_cache_recurrent_{layer_idx}",
                    ]
                )
                output_names.extend(
                    [
                        f"out_cache_conv_{layer_idx}",
                        f"out_cache_recurrent_{layer_idx}",
                    ]
                )
            else:
                raise NotImplementedError(
                    f"unsupported Qwen3.5 MoE layer type: {layer_type!r}"
                )

        return IOSpec(
            inputs=tuple(inputs),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        # pylint: disable-next=import-outside-toplevel
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=wrapper.model.config)
        layer_types = wrapper.model.config.layer_types

        cursor = 1
        first_full_key = None
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                key, value = inputs[cursor], inputs[cursor + 1]
                cursor += 2
                if first_full_key is None:
                    first_full_key = key
                cache.update(key, value, layer_idx)
            elif layer_type == "linear_attention":
                conv, recurrent = inputs[cursor], inputs[cursor + 1]
                cursor += 2
                cache.update_conv_state(conv, layer_idx)
                cache.update_recurrent_state(recurrent, layer_idx)
            else:
                raise NotImplementedError(
                    f"unsupported Qwen3.5 MoE layer type: {layer_type!r}"
                )

        attention_mask = None
        position_ids = None
        cache_position = None
        if (
            getattr(wrapper, "force_causal_mask", False)
            and first_full_key is not None
        ):
            # Past length P comes from the first full-attention KV tensor
            # [batch, heads, P, head_dim]; linear-attention state carries no
            # comparable time axis.
            attention_mask, position_ids, cache_position = (
                self.build_causal_mask_with_past(
                    seq_length=inputs[0].shape[1],
                    past_length=first_full_key.shape[2],
                    device=inputs[0].device,
                )
            )

        model_inputs = {
            "input_ids": inputs[0],
            "past_key_values": cache,
            "use_cache": True,
            "attention_mask": attention_mask,
        }
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids
            model_inputs["cache_position"] = cache_position
        return StateContext(model_inputs=model_inputs, state={})

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs: T.Any,
        state_context: StateContext,
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        del model, num_logits_to_keep
        cache = state_context.model_inputs["past_key_values"]
        outputs = [model_outputs["logits"]]
        for layer in cache.layers:
            if hasattr(layer, "keys"):
                outputs.extend([layer.keys, layer.values])
            else:
                outputs.extend(
                    [
                        self._linear_state_tensor(layer.conv_states),
                        self._linear_state_tensor(layer.recurrent_states),
                    ]
                )
        return outputs
