import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from torch_to_nnef.exceptions import T2NErrorConsistency


def resolve_submodule(root: T.Any, dotted_path: str) -> T.Any:
    """``getattr`` along a dotted path with an actionable error on a miss.

    Encoder handlers reach into the transformers-internal module layout
    (``model.visual``, ``vision_tower``, ``audio_tower`` ...), which drifts
    between transformers releases. A bare ``AttributeError`` surfacing deep in
    tracing is opaque; this names the missing attribute and the transformers
    version in play so a handler can be pointed at the renamed path.
    """
    obj = root
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            # pylint: disable-next=import-outside-toplevel
            import transformers

            raise T2NErrorConsistency(
                f"expected attribute path {dotted_path!r} on "
                f"{type(root).__name__}, but {type(obj).__name__} has no "
                f"{part!r}: the transformers module layout for this "
                f"architecture likely changed (transformers "
                f"{transformers.__version__}); update the handler's "
                "get_encoder_module."
            )
        obj = getattr(obj, part)
    return obj


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


@dataclass
class EmbeddingContract:
    """Tie one encoder output stream to the decoder input it feeds.

    The encoder graph emits a tensor named ``out_<modality>_embeddings`` of
    shape ``[dynamic_axis, hidden_size]``; the decoder graph consumes the
    matching ``in_<modality>_embeddings`` input of the same shape and dynamic
    symbol. This object is the single shared truth between the two graphs.

    ``injection_layers`` is empty for the common case (a single splice at the
    decoder embedding layer). It is non-empty only for multi-layer schemes such
    as Qwen3-VL DeepStack, where extra residual embeddings are added at the
    listed decoder layer indices; the i-th stream (``out/in_<modality>_
    deepstack_<i>``) is injected at ``injection_layers[i]``. Those per-layer
    streams carry ``deepstack_dynamic_axis`` as their token axis (falls back to
    ``dynamic_axis`` when unset).
    """

    modality: str
    hidden_size: int
    placeholder_token_id_attr: str
    dynamic_axis: str
    injection_layers: T.Tuple[int, ...] = ()
    deepstack_dynamic_axis: str = ""

    @property
    def input_name(self) -> str:
        return f"in_{self.modality}_embeddings"

    @property
    def output_name(self) -> str:
        return f"out_{self.modality}_embeddings"

    def deepstack_output_name(self, index: int) -> str:
        """Encoder output name for the i-th DeepStack residual stream."""
        return deepstack_output_name(self.modality, index)

    def deepstack_input_name(self, index: int) -> str:
        """Decoder input name for the i-th DeepStack residual stream."""
        return deepstack_input_name(self.modality, index)


def deepstack_output_name(modality: str, index: int) -> str:
    """Encoder output name for a modality's i-th DeepStack residual stream.

    Single source of truth for the DeepStack stream naming shared by the
    encoder graph (outputs), the decoder graph (inputs, via
    :func:`deepstack_input_name`) and the ``multimodal.json`` manifest.
    """
    return f"out_{modality}_deepstack_{index}"


def deepstack_input_name(modality: str, index: int) -> str:
    """Decoder input name for a modality's i-th DeepStack residual stream."""
    return f"in_{modality}_deepstack_{index}"


def reset_special_ids_to_filler(
    input_ids: torch.Tensor,
    special_ids: T.Iterable[int],
    vocab_size: int,
) -> None:
    """Rewrite in place any id colliding with a placeholder/special id.

    Sample ``input_ids`` are drawn at random, then the handler places the
    modality placeholder token at known positions; a random draw that happens
    to land on a special id would add a stray placeholder slot and break the
    feature/slot count. A hard-coded reset value (e.g. ``1``) silently fails
    when a special id equals it, so pick the first id that is itself not
    special.
    """
    specials = {int(s) for s in special_ids}
    if vocab_size <= len(specials):
        return
    filler = next(t for t in range(vocab_size) if t not in specials)
    for special in specials:
        input_ids[input_ids == special] = filler


def scatter_features_by_mask(
    *,
    inputs_embeds: torch.Tensor,
    token_mask: torch.Tensor,
    features: torch.Tensor,
    additive: bool = False,
) -> torch.Tensor:
    """Place flat ``[N, hidden]`` features where ``token_mask`` is set.

    The i-th True slot (row-major) receives ``features[i]``. Written with a
    gather (not boolean assignment) so it survives tracing to tract: boolean
    advanced indexing has a data-dependent output shape that NNEF shape
    inference cannot resolve.

    ``additive=False`` replaces the embedding at each masked position (the
    embedding-injection splice); ``additive=True`` adds the feature on top (the
    DeepStack residual stream). Shared by every encoder/decoder handler.
    """
    if features.numel() == 0:
        return inputs_embeds
    batch_size, seq_length = token_mask.shape
    token_counts = token_mask.to(torch.long).sum(dim=-1)
    total_tokens = int(token_counts.sum().item())
    if total_tokens == 0:
        return inputs_embeds
    if total_tokens != features.shape[0]:
        raise ValueError(
            f"feature/slot count mismatch: got {features.shape[0]} "
            f"feature(s) for {total_tokens} placeholder slot(s) in input_ids"
        )
    start_offsets = torch.cumsum(token_counts, dim=0) - token_counts
    slot_ids = token_mask.to(torch.long).cumsum(dim=-1)
    slot_ids = slot_ids + start_offsets.unsqueeze(-1)
    slot_ids = torch.where(token_mask, slot_ids, torch.zeros_like(slot_ids))
    zero_feature = torch.zeros(
        (1, features.shape[-1]),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    feature_bank = torch.cat(
        [
            zero_feature,
            features.to(inputs_embeds.device, inputs_embeds.dtype),
        ],
        dim=0,
    )
    gathered = feature_bank.index_select(0, slot_ids.reshape(-1)).view(
        batch_size, seq_length, inputs_embeds.shape[-1]
    )
    if additive:
        return inputs_embeds + gathered
    float_mask = token_mask.unsqueeze(-1).to(inputs_embeds.dtype)
    return inputs_embeds * (1 - float_mask) + gathered * float_mask


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

    def cleanup(self, *, state_context: "StateContext", wrapper) -> None:
        """Undo any model mutation made in ``build_forward_inputs``.

        Runs in a ``finally`` after every traced forward (see
        :class:`~torch_to_nnef_llm.models.base.BaseCausal`), so a handler that
        installs hooks or overrides model attributes to prepare a step restores
        the model even when the forward raises. No-op by default.
        """
        del state_context, wrapper

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


class EncoderHandler(ABC):
    """Base type for modality-encoder export behavior.

    Mirrors :class:`ArchitectureHandler` but for the encoder graph (vision
    tower or audio tower + projector). The encoder graph turns raw modality
    input (``pixel_values``, ``input_features`` ...) into embeddings whose
    output names and shapes satisfy the :class:`EmbeddingContract` shared with
    the decoder handler.
    """

    #: Coarse modality bucket, "vision" or "audio". Documentation-only.
    MODALITY: str = ""
    #: ``config.model_type`` values this encoder handler serves.
    ARCH_NAMES: T.Tuple[str, ...] = ()
    #: tract check_io tolerance for this encoder graph. Vision/audio towers
    #: accumulate more f32 attention drift than the LLM decoder, so they
    #: usually need a looser preset than the decoder default ("approximate").
    CHECK_IO_TOLERANCE: str = "very"

    @staticmethod
    def get_auto_model_class(transformers):
        """Return the HF model class to load for this architecture."""
        return transformers.AutoModel

    def prepare_model_for_export(self, model) -> None:
        """Apply arch tweaks before wrapping (eager attn, merge LoRA)."""
        return None

    def cleanup(self, *, state_context: "StateContext", wrapper) -> None:
        """Undo any model mutation made in ``build_forward_inputs``.

        Symmetric with :meth:`ArchitectureHandler.cleanup`; runs in a
        ``finally`` after the encoder forward. No-op by default (encoders are
        pure today).
        """
        del state_context, wrapper

    @abstractmethod
    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        """Return the submodule to trace (tower + projector)."""

    @abstractmethod
    def build_input_spec(
        self,
        *,
        config_helper,
        inputs_dtype: torch.dtype,
    ) -> IOSpec:
        """Build raw-modality inputs plus names/dynamic axes for the encoder."""

    @abstractmethod
    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        """Convert exported inputs into kwargs for the encoder module."""

    def call_encoder(
        self,
        *,
        model,
        state_context: StateContext,
        wrapper,
    ) -> T.Any:
        """Run the encoder module with prepared inputs."""
        return model(
            **state_context.model_inputs,
            **wrapper.forward_kwargs,
        )

    @abstractmethod
    def build_forward_outputs(
        self,
        *,
        model_outputs: T.Any,
        state_context: StateContext,
    ) -> T.List[torch.Tensor]:
        """Build exported embeddings matching the encoder output names."""

    @abstractmethod
    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        """Return the embedding contracts this encoder produces."""


@dataclass
class MultiModalArchitectureHandler:
    """Pair a decoder handler with the encoder handler(s) feeding it.

    Owns the set of :class:`EmbeddingContract` linking each encoder output to
    the decoder input it feeds. Absent for text-only models, so the plain
    decoder path is unchanged.
    """

    decoder_handler: ArchitectureHandler
    encoder_handlers: T.List[EncoderHandler]

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        out: T.List[EmbeddingContract] = []
        for handler in self.encoder_handlers:
            out.extend(handler.contracts(config_helper))
        return out
