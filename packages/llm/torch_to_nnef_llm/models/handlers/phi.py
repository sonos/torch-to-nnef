import typing as T

import torch

from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache

from .base import StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class PhiArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Phi-family models."""

    ARCH_NAMES = ("phi",)

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        input_ids = inputs[0]
        cache = build_past_kv_dyn_cache(inputs[1:])
        _, seq_length = input_ids.shape[:2]
        past_key_values_length = cache.get_seq_length()
        cache_position = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = cache_position.unsqueeze(0)
        inputs_embeds = wrapper.model.model.embed_tokens(input_ids)
        attention_mask = (
            torch.triu(
                torch.full(
                    [seq_length, seq_length],
                    torch.finfo(inputs_embeds.dtype).min,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(inputs_embeds.dtype)
        return StateContext(
            model_inputs={
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": cache,
                "output_attentions": False,
                "use_cache": True,
                "cache_position": cache_position,
            },
            state={},
        )

    def call_model(
        self,
        *,
        model,
        state_context: StateContext,
        wrapper,
    ) -> T.Any:
        del wrapper
        return model.model(**state_context.model_inputs)

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs: T.Any,
        state_context: StateContext,
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        hidden_states = model_outputs[0]
        logits = model.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        past_key_values = state_context.model_inputs["past_key_values"]
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy_cache = past_key_values.to_legacy_cache()
        else:
            legacy_cache = [(kv[0], kv[1]) for kv in past_key_values]
        kvs = [t for kv in legacy_cache for t in kv]
        return [logits] + kvs
