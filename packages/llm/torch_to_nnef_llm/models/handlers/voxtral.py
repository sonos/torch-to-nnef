"""Joint-export handlers for Voxtral audio-language models.

Proves the encoder abstraction generalizes from vision to audio with no
modality-specific escape hatch: the same :class:`EncoderHandler` /
:class:`EmbeddingContract`, just ``modality="audio"``.

- :class:`VoxtralAudioEncoderHandler`: the Whisper-style audio tower (conv1d
  subsample + transformer) plus the multimodal projector. Input is log-mel
  ``input_features`` (the STFT/mel front-end stays in the HF FeatureExtractor,
  outside the graph); output is audio embeddings for the decoder splice.
- :class:`VoxtralArchitectureHandler`: the decoder graph, injecting audio
  embeddings into the token stream at ``audio_token_id`` (single splice; the
  backbone is a plain Mistral/Ministral causal LM).
"""

import typing as T

import torch

from .base import (
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    StateContext,
    reset_special_ids_to_filler,
    resolve_submodule,
    scatter_features_by_mask,
)
from .default import DefaultArchitectureHandler
from .registry import register_encoder_handler, register_handler


def _num_audio_tokens(audio_config) -> int:
    """Audio embeddings one 30s mel chunk becomes after tower + projector."""
    return (
        audio_config.max_source_positions
        * audio_config.hidden_size
        // audio_config.intermediate_size
    )


class VoxtralAudioEncoder(torch.nn.Module):
    """Audio tower + projector traced as one encoder graph.

    Input log-mel ``input_features`` ``[1, num_mel_bins, num_frames]`` (fixed
    30s chunk); output ``[num_audio_tokens, text_hidden]``.
    """

    def __init__(self, audio_tower, projector, intermediate_size: int):
        super().__init__()
        self.audio_tower = audio_tower
        self.projector = projector
        self.intermediate_size = intermediate_size

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden = self.audio_tower(input_features).last_hidden_state
        hidden = hidden.reshape(-1, self.intermediate_size)
        return self.projector(hidden)


@register_encoder_handler
class VoxtralAudioEncoderHandler(EncoderHandler):
    """Encoder handler for the Voxtral audio tower."""

    MODALITY = "audio"
    ARCH_NAMES = ("voxtral",)
    MODEL_INPUT_NAME = "input_features"

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        return VoxtralAudioEncoder(
            resolve_submodule(hf_model, "audio_tower"),
            resolve_submodule(hf_model, "multi_modal_projector"),
            hf_model.config.audio_config.intermediate_size,
        )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        audio_conf = config_helper.conf.audio_config
        num_frames = audio_conf.max_source_positions * 2  # conv2 stride 2
        input_features = torch.randn(
            (1, audio_conf.num_mel_bins, num_frames), dtype=inputs_dtype
        )
        return IOSpec(
            inputs=(input_features,),
            input_names=["input_features"],
            output_names=["out_audio_embeddings"],
            dynamic_axes={},  # Whisper pos-emb fixes the 30s chunk length
        )

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        return [
            EmbeddingContract(
                modality="audio",
                hidden_size=config_helper.decoder_conf.hidden_size,
                placeholder_token_id_attr="audio_token_id",
                dynamic_axis="AUD",
            )
        ]


@register_handler
class VoxtralArchitectureHandler(DefaultArchitectureHandler):
    """Decoder handler for Voxtral: audio embeddings as input."""

    ARCH_NAMES = ("voxtral",)
    STATE_INPUT_NAMES = ["in_audio_embeddings"]
    STATE_OUTPUT_NAMES = ["out_audio_embeddings"]

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.VoxtralForConditionalGeneration

    def prepare_model_for_export(self, model) -> None:
        for conf in (
            model.config,
            getattr(model.config, "text_config", None),
            getattr(model.config, "audio_config", None),
        ):
            if conf is not None:
                conf._attn_implementation = "eager"

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
        del tokenizer, sample_text
        num_audio_tokens = _num_audio_tokens(config_helper.conf.audio_config)
        effective_seq_len = max(n_input_tokens, num_audio_tokens + 2)
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
        hidden_size = config_helper.decoder_conf.hidden_size
        vocab_size = config_helper.decoder_conf.vocab_size
        audio_token_id = config_helper.conf.audio_token_id

        input_ids = torch.randint(0, vocab_size, (1, effective_seq_len))
        reset_special_ids_to_filler(input_ids, {audio_token_id}, vocab_size)
        for idx in range(num_audio_tokens):
            input_ids[:, 1 + idx] = audio_token_id
        audio_embeddings = torch.randn(
            (num_audio_tokens, hidden_size), dtype=inputs_dtype
        )

        return IOSpec(
            inputs=(input_ids, *past_key_values, audio_embeddings),
            input_names=["input_ids"] + in_cache_names + self.STATE_INPUT_NAMES,
            output_names=["outputs"]
            + out_cache_names
            + self.STATE_OUTPUT_NAMES,
            dynamic_axes={**dynamic_axes, "in_audio_embeddings": {0: "AUD"}},
        )

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        hf_model = wrapper.model
        input_ids = inputs[0]
        audio_embeddings = inputs[-1]
        cache_tensors = inputs[1:-1]

        base_ctx = super().build_forward_inputs(
            inputs=(input_ids, *cache_tensors), wrapper=wrapper
        )
        inputs_embeds = hf_model.get_input_embeddings()(input_ids)
        inputs_embeds = scatter_features_by_mask(
            inputs_embeds=inputs_embeds,
            token_mask=input_ids == hf_model.config.audio_token_id,
            features=audio_embeddings,
        )
        base_ctx.model_inputs["input_ids"] = None
        base_ctx.model_inputs["inputs_embeds"] = inputs_embeds
        base_ctx.model_inputs["input_features"] = None
        base_ctx.state = {"audio_embeddings": audio_embeddings}
        return base_ctx

    def build_forward_outputs(
        self, *, model, model_outputs, state_context, num_logits_to_keep
    ) -> T.List[torch.Tensor]:
        outputs = super().build_forward_outputs(
            model=model,
            model_outputs=model_outputs,
            state_context=state_context,
            num_logits_to_keep=num_logits_to_keep,
        )
        return outputs + [state_context.state["audio_embeddings"]]
