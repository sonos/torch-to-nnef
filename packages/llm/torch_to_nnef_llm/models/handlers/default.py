import typing as T

import torch

from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache, build_past_kv_list
from torch_to_nnef_llm.models.base import BaseCausal

from .base import ArchitectureHandler, InputSpec


class DefaultArchitectureHandler(ArchitectureHandler):
    """Fallback handler for standard causal decoder models"""

    ARCH_NAMES = ("default",)

    @staticmethod
    def get_wrapper_class():
        return BaseCausal

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
        inputs = tuple([test_input.input_ids[:, :n_input_tokens]] + past_key_values)
        input_names = ["input_ids"] + in_cache_names
        output_names = ["outputs"] + out_cache_names
        return InputSpec(
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    def prepare_inputs_for_model(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> T.Dict[str, T.Any]:
        if wrapper.with_dyn_cache:
            past_key_values = build_past_kv_dyn_cache(inputs[1:])
        else:
            past_key_values = build_past_kv_list(inputs[1:])
        return {
            "input_ids": inputs[0],
            "past_key_values": past_key_values,
            "use_cache": True,
        }
