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


@dataclass
class StateContext:
    """Carry model kwargs plus handler-private state across the forward path."""

    model_inputs: T.Dict[str, T.Any]
    state: T.Dict[str, T.Any]


class ArchitectureHandler(ABC):
    """Base type for architecture-specific export behavior."""

    ARCH_NAMES: T.Tuple[str, ...] = ()
    with_dyn_cache: bool = True

    @staticmethod
    def get_auto_model_class(transformers):
        """Return the HF model class to load for this architecture."""
        return transformers.AutoModelForCausalLM

    def prepare_model_for_export(self, model) -> None:
        """Apply architecture-specific model tweaks before wrapping."""

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

    @abstractmethod
    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        """Convert exported inputs into kwargs expected by the HF model."""

    def call_model(
        self,
        *,
        model,
        state_context: StateContext,
        wrapper,
    ) -> T.Any:
        """Run the underlying model with prepared inputs."""
        return model(
            **state_context.model_inputs,
            **wrapper.forward_kwargs,
        )

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs: T.Any,
        state_context: StateContext,
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        """Build exported outputs matching IOSpec.output_names."""
        del model, num_logits_to_keep
        if self.with_dyn_cache:
            past_key_values = state_context.model_inputs["past_key_values"]
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
