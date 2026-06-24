import typing as T

import torch

from torch_to_nnef_llm.models.base import (
    build_past_kv_dyn_cache,
    build_past_kv_list,
)

from .base import ArchitectureHandler, InputSpec
from .registry import register_handler


@register_handler
class DefaultArchitectureHandler(ArchitectureHandler):
    """Fallback handler for standard causal decoder models."""

    ARCH_NAMES = ("default",)

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
    ) -> InputSpec:
        test_input = tokenizer(sample_text, return_tensors="pt")
        assert test_input.input_ids.shape[1] >= n_input_tokens
        (
            in_cache_names,
            out_cache_names,
            past_key_values,
            dynamic_axes,
        ) = config_helper.build_kv_cache_infos(
            n_past_input_tokens=n_past_input_tokens,
            force_inputs_dtype=inputs_dtype,
            real_kv_cache=real_kv_cache,
        )
        inputs = tuple(
            [test_input.input_ids[:, :n_input_tokens]] + past_key_values
        )
        input_names = ["input_ids"] + in_cache_names
        output_names = ["outputs"] + out_cache_names
        return InputSpec(
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> T.Dict[str, T.Any]:
        attention_mask = None
        position_ids = None
        cache_position = None
        if getattr(wrapper, "force_causal_mask", False):
            attn_mask_dtype = torch.float32
            neg = torch.finfo(attn_mask_dtype).min
            # Query length S (new tokens) from input_ids, past length P from
            # the first KV cache tensor [batch, heads, P, head_dim].
            seq_length = inputs[0].shape[1]
            past_length = inputs[1].shape[2]
            total_length = seq_length + past_length
            # Absolute positions of the new tokens: P, P+1, ..., P+S-1. These
            # drive RoPE; without them transformers (>4.52.4) infers positions
            # that ignore the past, so decode RoPE is wrong and generation
            # degenerates into repetition.
            cache_position = torch.arange(
                past_length, total_length, device=inputs[0].device
            )
            position_ids = cache_position.unsqueeze(0)
            # Build the [S, S+P] causal-with-past additive mask from token
            # POSITIONS (not a fixed-diagonal triu): query i sits at absolute
            # position P+i and may attend to keys 0..P+i. Doing it via arange
            # comparisons keeps it correct for any (S, P) at inference time;
            # a baked triu diagonal (or the previous shape[0]=batch bug) made
            # the mask degenerate and attention non-causal.
            q_pos = cache_position.unsqueeze(1)
            k_pos = torch.arange(total_length).unsqueeze(0)
            future = (k_pos > q_pos).to(attn_mask_dtype)  # 1.0 where masked
            attention_mask = (
                (torch.full([seq_length, total_length], neg) * future)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(attn_mask_dtype)
            )
        if wrapper.with_dyn_cache:
            past_key_values = build_past_kv_dyn_cache(inputs[1:])
        else:
            past_key_values = build_past_kv_list(inputs[1:])
        model_inputs = {
            "input_ids": inputs[0],
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids
            model_inputs["cache_position"] = cache_position
        return model_inputs
