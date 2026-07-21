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

    @staticmethod
    def build_causal_mask_with_past(
        *,
        seq_length: int,
        past_length: int,
        device: torch.device,
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the additive causal-with-past mask plus RoPE positions.

        Returns ``(attention_mask, position_ids, cache_position)`` for
        ``seq_length`` new query tokens attending over ``past_length`` cached
        keys. Shared by every handler that fakes a decode step so the subtle
        position math lives in exactly one place.
        """
        attn_mask_dtype = torch.float32
        neg = torch.finfo(attn_mask_dtype).min
        # Query length S (new tokens) and past length P; the causal window is
        # over S+P keys.
        total_length = seq_length + past_length
        # Absolute positions of the new tokens: P, P+1, ..., P+S-1. These
        # drive RoPE; without them transformers (>4.52.4) infers positions
        # that ignore the past, so decode RoPE is wrong and generation
        # degenerates into repetition.
        cache_position = torch.arange(past_length, total_length, device=device)
        position_ids = cache_position.unsqueeze(0)
        # Build the [S, S+P] causal-with-past additive mask from token
        # POSITIONS (not a fixed-diagonal triu): query i sits at absolute
        # position P+i and may attend to keys 0..P+i. Doing it via arange
        # comparisons keeps it correct for any (S, P) at inference time;
        # a baked triu diagonal (or a shape[0]=batch bug) made the mask
        # degenerate and attention non-causal.
        q_pos = cache_position.unsqueeze(1)
        k_pos = torch.arange(total_length).unsqueeze(0)
        future = (k_pos > q_pos).to(attn_mask_dtype)  # 1.0 where masked
        attention_mask = (
            (torch.full([seq_length, total_length], neg) * future)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(attn_mask_dtype)
        )
        return attention_mask, position_ids, cache_position

    def prepare_model_for_export(self, model) -> None:
        """Apply architecture-specific model tweaks before wrapping."""
        return None

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
