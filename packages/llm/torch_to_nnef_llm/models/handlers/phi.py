import typing as T

import torch

from torch_to_nnef_llm.models.base import (
    BaseCausalWithDynCacheAndTriu,
    build_past_kv_dyn_cache,
)

from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class PhiArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Phi-family models."""

    ARCH_NAMES = ("phi",)

    @staticmethod
    def get_wrapper_class():
        return BaseCausalWithDynCacheAndTriu

    def prepare_inputs_for_model(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> T.Dict[str, T.Any]:
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
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": cache,
            "output_attentions": False,
            "use_cache": True,
            "cache_position": cache_position,
        }
