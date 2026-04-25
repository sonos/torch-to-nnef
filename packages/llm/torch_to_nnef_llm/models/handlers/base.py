import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class IOSpec:
    """Defines exported inputs/outputs for one graph or graph state bundle."""

    inputs: T.Tuple[torch.Tensor, ...]
    input_names: T.List[str]
    output_names: T.List[str]
    dynamic_axes: T.Dict[str, T.Dict[int, str]]


class ArchitectureHandler(ABC):
    """Base type for architecture-specific export behavior."""

    ARCH_NAMES: T.Tuple[str, ...] = ()
    with_dyn_cache: bool = True

    @staticmethod
    def get_auto_model_class(transformers):
        """Return the HF model class to load for this architecture."""
        return transformers.AutoModelForCausalLM

    @abstractmethod
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
        """Build exported inputs plus names/dynamic axes for the decoder."""

    def build_decoder_state_spec(
        self,
        *,
        config_helper,
        inputs_dtype: torch.dtype,
        n_input_tokens: int,
        n_past_input_tokens: int,
    ) -> IOSpec:
        """Build any additional decoder state tensors."""
        return IOSpec(
            inputs=(),
            input_names=[],
            output_names=[],
            dynamic_axes={},
        )

    @abstractmethod
    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> T.Dict[str, T.Any]:
        """Convert exported inputs into kwargs expected by the HF model."""

    def call_model(
        self,
        *,
        model,
        model_inputs: T.Dict[str, T.Any],
        wrapper,
    ) -> T.Any:
        """Run the underlying model with prepared inputs."""
        return model(
            **model_inputs,
            **wrapper.forward_kwargs,
        )

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs: T.Any,
        model_inputs: T.Dict[str, T.Any],
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        """Build exported outputs matching IOSpec.output_names."""
        del model, num_logits_to_keep
        if self.with_dyn_cache:
            past_key_values = model_inputs["past_key_values"]
            if hasattr(past_key_values, "to_legacy_cache"):
                pkv = past_key_values.to_legacy_cache()
            else:
                pkv = [(kv[0], kv[1]) for kv in past_key_values]
            kvs = [t for kv in pkv for t in kv]
        else:
            kvs = [
                k_or_v
                for kv in model_outputs["past_key_values"]
                for k_or_v in kv
            ]
        return [model_outputs["logits"]] + kvs

    def prepare_additional_outputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        prepared_inputs: T.Dict[str, T.Any],
        hf_outputs,
        wrapper,
    ) -> T.List[torch.Tensor]:
        """Return additional output state tensors beyond logits and KV cache."""
        return []
