import typing as T

import torch

from .base import IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class Qwen35MoeArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Qwen3.5 MoE text models with hybrid attention caches."""

    ARCH_NAMES = ("qwen3_5_moe_text",)

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
                outputs.extend([layer.conv_states, layer.recurrent_states])
        return outputs
