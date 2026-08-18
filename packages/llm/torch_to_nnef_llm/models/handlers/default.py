import typing as T

import torch

from torch_to_nnef_llm.models.base import (
    build_past_kv_dyn_cache,
    build_past_kv_list,
)

from .base import ArchitectureHandler, IOSpec, StateContext
from .registry import register_handler


@register_handler
class DefaultArchitectureHandler(ArchitectureHandler):
    """Fallback handler for standard causal decoder models."""

    ARCH_NAMES: T.Tuple[str, ...] = ("default",)

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
        return IOSpec(
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
    ) -> StateContext:
        attention_mask = None
        position_ids = None
        cache_position = None
        if getattr(wrapper, "force_causal_mask", False):
            # Query length S (new tokens) from input_ids, past length P from
            # the first KV cache tensor [batch, heads, P, head_dim].
            attention_mask, position_ids, cache_position = (
                self.build_causal_mask_with_past(
                    seq_length=inputs[0].shape[1],
                    past_length=inputs[1].shape[2],
                    device=inputs[0].device,
                )
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
        return StateContext(
            model_inputs=model_inputs,
            state={},
        )
