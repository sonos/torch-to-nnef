"""Joint-export handlers for Gemma 4 vision-language models.

Gemma 4 is a three-branch multimodal model (vision + audio + video towers over
a per-layer-input, MoE-capable text decoder). This module implements the
**vision** branch; audio/video/MoE are handled separately.

- :class:`Gemma4VisionEncoderHandler`: the vision tower + ``embed_vision``
  projector. The tower consumes flattened patches plus 2D patch positions and a
  padding mask; for a fixed (baked) full square grid those positions are
  constant, so we bake them and expose only ``pixel_values`` (same trick as the
  Qwen towers).
- :class:`Gemma4ArchitectureHandler`: the decoder graph. Gemma 4 needs two
  things a plain splice does not: (1) *per-layer input embeddings* computed from
  the original token ids (image ids mapped to ``pad``) **before** the image
  soft tokens are spliced in: otherwise the language model would try to
  reverse the embedding of a spliced token and fail; (2) a direct call into
  ``language_model`` (not the top-level model) so those pre-computed per-layer
  inputs can be passed through. Small models use a conventional causal mask
  (only the larger ``use_bidirectional_attention == "vision"`` variants need the
  Gemma 3 style image-span bidirectional mask).
"""

import typing as T

import torch

from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache

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


def _sample_grid_side(vision_conf) -> int:
    """Side (in patches) of the baked square image, a small multiple of pool.

    The pooler requires the patch count to be ``k**2 * output_length`` with
    ``k == pooling_kernel_size``, so a square grid side must be a multiple of
    the pooling kernel; ``2 * pool`` keeps the sample tiny (4 soft tokens).
    """
    return 2 * int(vision_conf.pooling_kernel_size)


def _num_soft_tokens(vision_conf) -> int:
    side = _sample_grid_side(vision_conf)
    pool = int(vision_conf.pooling_kernel_size)
    return (side * side) // (pool * pool)


def _grid_position_ids(side: int) -> torch.Tensor:
    """Row-major ``(x, y)`` patch coordinates for a ``side x side`` grid."""
    coords = [[i % side, i // side] for i in range(side * side)]
    return torch.tensor([coords], dtype=torch.long)


class Gemma4VisionEncoder(torch.nn.Module):
    """Vision tower + ``embed_vision`` projector with baked patch positions.

    Input ``pixel_values`` is the flattened-patch tensor
    ``[1, num_patches, 3 * patch_size**2]``; output is the projected soft
    tokens ``[num_soft_tokens, text_hidden]`` ready for the decoder splice. The
    2D patch positions are baked constant (full square grid, no padding), so
    the tower's one-hot position embedding and 2D pooling fold to constants.

    The tower forward is inlined here rather than calling
    :meth:`Gemma4VisionModel.forward` to sidestep two trace hazards that only
    matter for padded / variable-resolution inputs, neither of which occurs for
    a full square grid:

    - ``create_bidirectional_mask`` (breaks tracing via the transformers masking
      utils): with no padding every patch attends every patch, i.e. the mask is
      uniform, so the layers run with ``attention_mask=None``.
    - the pooler's data-dependent ``hidden_states[pooler_mask]`` padding strip:
      with no padding ``pooler_mask`` is all-true, so it reduces to a reshape.
    """

    def __init__(self, vision_tower, embed_vision, pixel_position_ids):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.register_buffer(
            "pixel_position_ids", pixel_position_ids, persistent=False
        )
        # Precompute the 2D-pooling weights. They depend only on the baked
        # positions + pooling kernel, so they are constant; folding them here
        # avoids the pooler's one-hot / floor-divide (which the IR cannot
        # translate) and leaves a single constant matmul in the graph.
        self.register_buffer(
            "_pool_weights", self._pooling_weights(), persistent=False
        )

    def _pooling_weights(self) -> torch.Tensor:
        pos_ids = self.pixel_position_ids
        num_patches = pos_ids.shape[1]
        pool = int(self.vision_tower.config.pooling_kernel_size)
        length = num_patches // (pool * pool)
        clamped = pos_ids.clamp(min=0)[0]  # [num_patches, 2]
        max_x = int(clamped[:, 0].max().item()) + 1
        kernel_idxs = clamped // pool
        kernel_idxs = kernel_idxs[:, 0] + (max_x // pool) * kernel_idxs[:, 1]
        weights = torch.nn.functional.one_hot(
            kernel_idxs.long(), length
        ).float()
        return weights / (pool * pool)  # [num_patches, length]

    def _patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Patch embedding with the 2D position lookup as a gather.

        The tower computes position embeddings as ``one_hot(pos) @ table``;
        with baked constant positions that is an embedding lookup, so we emit an
        ``index_select`` (which tract handles) instead of the one-hot matmul
        (which the IR cannot translate).
        """
        embedder = self.vision_tower.patch_embedder
        pos_ids = self.pixel_position_ids
        table = embedder.position_embedding_table  # [2, pos_emb_size, hidden]
        clamped = pos_ids.clamp(min=0)
        pos_x = clamped[..., 0].reshape(-1)
        pos_y = clamped[..., 1].reshape(-1)
        pe = table[0].index_select(0, pos_x) + table[1].index_select(0, pos_y)
        pe = pe.unsqueeze(0)  # [1, num_patches, hidden]
        scaled = 2 * (pixel_values - 0.5)
        hidden = embedder.input_proj(
            scaled.to(embedder.input_proj.weight.dtype)
        )
        return hidden + pe

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tower = self.vision_tower
        pos_ids = self.pixel_position_ids

        hidden = self._patch_embed(pixel_values)
        position_embeddings = tower.encoder.rotary_emb(hidden, pos_ids)
        for layer in tower.encoder.layers[: tower.encoder.num_layers]:
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,  # no padding -> full (bidirectional) attn
                position_ids=pos_ids,
            )

        # 2D avg-pool via the precomputed constant weights (see __init__), then
        # scale as the pooler does. No padding -> the model's boolean strip is a
        # reshape.
        pooled = (
            self._pool_weights.float().transpose(0, 1) @ hidden.float()
        ).to(hidden.dtype)
        pooled = pooled * (tower.pooler.hidden_size**0.5)
        pooled = pooled.reshape(-1, pooled.shape[-1])
        if tower.config.standardize:
            pooled = (pooled - tower.std_bias) * tower.std_scale
        return self.embed_vision(inputs_embeds=pooled)


@register_encoder_handler
class Gemma4VisionEncoderHandler(EncoderHandler):
    """Encoder handler for the Gemma 4 vision tower."""

    MODALITY = "vision"
    ARCH_NAMES = ("gemma4",)
    MODEL_INPUT_NAME = "pixel_values"

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        vision_conf = hf_model.config.vision_config
        side = _sample_grid_side(vision_conf)
        return Gemma4VisionEncoder(
            resolve_submodule(hf_model, "model.vision_tower"),
            resolve_submodule(hf_model, "model.embed_vision"),
            _grid_position_ids(side),
        )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        vision_conf = config_helper.conf.vision_config
        side = _sample_grid_side(vision_conf)
        num_patches = side * side
        patch_dim = 3 * vision_conf.patch_size * vision_conf.patch_size
        # patch pixels are scaled to [-1, 1] inside the tower (2*(x-0.5)); a
        # [0, 1) sample matches the processor's normalized range.
        pixel_values = torch.rand(
            (1, num_patches, patch_dim), dtype=inputs_dtype
        )
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            # positions/patch count are baked constant -> no dynamic axis.
            dynamic_axes={},
        )

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        return [
            EmbeddingContract(
                modality="image",
                hidden_size=config_helper.decoder_conf.hidden_size,
                placeholder_token_id_attr="image_token_id",
                dynamic_axis="IMG",
            )
        ]


@register_encoder_handler
class Gemma4VideoEncoderHandler(Gemma4VisionEncoderHandler):
    """Video encoder: the same vision tower applied to video frames.

    ``get_video_features`` just flattens frames through the vision tower, so a
    single baked frame is identical to the image path (same weights, same
    graph); only the placeholder token it feeds differs. Registered under the
    distinct ``"video"`` modality bucket so both encoders coexist.
    """

    MODALITY = "video"

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        spec = super().build_input_spec(
            config_helper=config_helper, inputs_dtype=inputs_dtype
        )
        return IOSpec(
            inputs=spec.inputs,
            input_names=spec.input_names,
            output_names=["out_video_embeddings"],
            dynamic_axes=spec.dynamic_axes,
        )

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        return [
            EmbeddingContract(
                modality="video",
                hidden_size=config_helper.decoder_conf.hidden_size,
                placeholder_token_id_attr="video_token_id",
                dynamic_axis="VIDEO",
            )
        ]


def _audio_conv_len(length: int) -> int:
    """Output length of one stride-2, padding-1, kernel-3 conv."""
    return (length - 1) // 2 + 1


def _sample_audio_frames(audio_conf) -> int:
    """Baked mel-frame count for the audio sample (a few attention chunks)."""
    return 8 * int(audio_conf.attention_chunk_size)


def _num_audio_soft_tokens(audio_conf) -> int:
    frames = _sample_audio_frames(audio_conf)
    return _audio_conv_len(_audio_conv_len(frames))  # two /2 subsamples


class Gemma4AudioEncoder(torch.nn.Module):
    """Audio conformer tower + ``embed_audio`` projector, baked feature mask.

    Input ``input_features`` is log-mel ``[1, frames, num_mel_bins]`` (the
    STFT/mel front-end stays in the HF feature extractor); output is the
    projected soft tokens ``[num_audio_tokens, text_hidden]``. The valid-frame
    mask is baked (a full, no-padding chunk), so the tower's padding strip is a
    reshape.

    NOTE: this branch is validated numerically in PyTorch but is **not yet
    registered** for joint export: the Universal-Speech-Model conformer uses
    ``unfold`` / chunked 5D local attention / a relative-shift trick that
    ``tract`` does not yet cover. Wire it up (register + add ``"audio"`` to the
    decoder ``MODALITIES``) once those ops land.
    """

    def __init__(self, audio_tower, embed_audio, input_features_mask):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio
        self.register_buffer(
            "input_features_mask", input_features_mask, persistent=False
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        outputs = self.audio_tower(input_features, self.input_features_mask)
        pooled = self.embed_audio(inputs_embeds=outputs.last_hidden_state)
        # no padding -> every frame valid -> the model's boolean strip keeps all
        return pooled.reshape(-1, pooled.shape[-1])


class Gemma4AudioEncoderHandler(EncoderHandler):
    """Encoder handler for the Gemma 4 audio tower (tract-pending, see above).

    Deliberately not decorated with ``@register_encoder_handler``: the recipe
    is validated in PyTorch (see the audio chain-parity test) but the conformer
    tower cannot export to tract yet.
    """

    MODALITY = "audio"
    ARCH_NAMES = ("gemma4",)
    MODEL_INPUT_NAME = "input_features"

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        audio_conf = hf_model.config.audio_config
        frames = _sample_audio_frames(audio_conf)
        mask = torch.ones(1, frames, dtype=torch.bool)
        return Gemma4AudioEncoder(
            resolve_submodule(hf_model, "model.audio_tower"),
            resolve_submodule(hf_model, "model.embed_audio"),
            mask,
        )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        audio_conf = config_helper.conf.audio_config
        frames = _sample_audio_frames(audio_conf)
        num_mel_bins = audio_conf.subsampling_conv_channels[0]
        input_features = torch.randn(
            (1, frames, num_mel_bins), dtype=inputs_dtype
        )
        return IOSpec(
            inputs=(input_features,),
            input_names=["input_features"],
            output_names=["out_audio_embeddings"],
            dynamic_axes={},  # fixed baked chunk length
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
class Gemma4ArchitectureHandler(DefaultArchitectureHandler):
    """Decoder handler for Gemma 4: modality embeddings + per-layer inputs.

    Splices one soft-token stream per modality (image + video today; both come
    from the vision tower). Audio is a separate branch pending tract support for
    the conformer encoder, but the decoder splice is modality-neutral and would
    extend by adding an entry to ``MODALITIES``.
    """

    ARCH_NAMES = ("gemma4",)
    #: (contract modality, placeholder-token-id config attr), one decoder state
    #: input/output per modality.
    MODALITIES = (("image", "image_token_id"), ("video", "video_token_id"))

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Gemma4ForConditionalGeneration

    def prepare_model_for_export(self, model) -> None:
        for conf in (
            model.config,
            getattr(model.config, "text_config", None),
            getattr(model.config, "vision_config", None),
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
        conf = config_helper.conf
        # image + video both come from the vision tower -> same soft-token count
        num_tokens = _num_soft_tokens(conf.vision_config)
        total_mm = num_tokens * len(self.MODALITIES)
        effective_seq_len = max(n_input_tokens, total_mm + 2)
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
        specials = {getattr(conf, attr) for _, attr in self.MODALITIES}

        input_ids = torch.randint(0, vocab_size, (1, effective_seq_len))
        reset_special_ids_to_filler(input_ids, specials, vocab_size)
        # place each modality's placeholders at a distinct span (pos 0 stays a
        # real text token so decode still has one).
        embeddings, state_in, state_out = [], [], []
        dyn_axes = dict(dynamic_axes)
        pos = 1
        for modality, attr in self.MODALITIES:
            token_id = getattr(conf, attr)
            input_ids[:, pos : pos + num_tokens] = token_id
            pos += num_tokens
            embeddings.append(
                torch.randn((num_tokens, hidden_size), dtype=inputs_dtype)
            )
            in_name = f"in_{modality}_embeddings"
            state_in.append(in_name)
            state_out.append(f"out_{modality}_embeddings")
            dyn_axes[in_name] = {0: modality.upper()}

        return IOSpec(
            inputs=(input_ids, *past_key_values, *embeddings),
            input_names=["input_ids"] + in_cache_names + state_in,
            output_names=["outputs"] + out_cache_names + state_out,
            dynamic_axes=dyn_axes,
        )

    @staticmethod
    def _build_mask_mapping(
        *,
        query_length: int,
        past_length: int,
        sliding_window: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> T.Dict[str, torch.Tensor]:
        """Conventional causal + sliding-window masks, causal-with-past.

        Built from token positions with arange comparisons so it stays correct
        for any (query, past) at inference and exports to tract. Small Gemma 4
        models are plain causal (no image-span bidirectionality: that only
        applies to ``use_bidirectional_attention == "vision"`` variants).
        """
        total = past_length + query_length
        q_pos = (past_length + torch.arange(query_length, device=device)).view(
            query_length, 1
        )
        k_pos = torch.arange(total, device=device).view(1, total)
        causal = (k_pos <= q_pos).to(dtype)
        within_window = (k_pos > (q_pos - sliding_window)).to(dtype)
        sliding = causal * within_window
        neg = torch.finfo(dtype).min

        def to_additive(visible: torch.Tensor) -> torch.Tensor:
            return ((1.0 - visible) * neg).unsqueeze(0).unsqueeze(0)

        return {
            "full_attention": to_additive(causal),
            "sliding_attention": to_additive(sliding),
        }

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        hf_model = wrapper.model
        n_mod = len(self.MODALITIES)
        modality_embeds = list(inputs[-n_mod:])
        input_ids = inputs[0]
        cache_tensors = inputs[1:-n_mod]

        language_model = hf_model.model.language_model
        conf = hf_model.config
        text_config = conf.text_config
        pad_token_id = text_config.pad_token_id

        masks = [
            input_ids == getattr(conf, attr) for _, attr in self.MODALITIES
        ]
        multimodal_mask = masks[0]
        for mask in masks[1:]:
            multimodal_mask = multimodal_mask | mask
        # placeholder ids are out of the text vocab used for embedding /
        # per-layer lookup; map them to pad first (matches Gemma4Model.forward).
        llm_input_ids = torch.where(
            multimodal_mask,
            torch.full_like(input_ids, pad_token_id),
            input_ids,
        )
        inputs_embeds = hf_model.model.get_input_embeddings()(llm_input_ids)
        # per-layer inputs MUST come from the token ids, before the splice: the
        # language model cannot recover ids from a spliced soft-token embedding.
        per_layer_inputs = language_model.get_per_layer_inputs(
            llm_input_ids, None
        )
        for mask, features in zip(masks, modality_embeds, strict=True):
            inputs_embeds = scatter_features_by_mask(
                inputs_embeds=inputs_embeds,
                token_mask=mask,
                features=features,
            )
        past_key_values = build_past_kv_dyn_cache(cache_tensors)

        past_length = cache_tensors[0].shape[-2] if cache_tensors else 0
        query_length = input_ids.shape[1]
        sliding_window = int(getattr(text_config, "sliding_window", 4096) or 0)
        mask_mapping = self._build_mask_mapping(
            query_length=query_length,
            past_length=past_length,
            sliding_window=sliding_window or (past_length + query_length),
            dtype=inputs_embeds.dtype,
            device=input_ids.device,
        )
        position_ids = (
            past_length + torch.arange(query_length, device=input_ids.device)
        ).unsqueeze(0)

        return StateContext(
            model_inputs={
                "inputs_embeds": inputs_embeds,
                "per_layer_inputs": per_layer_inputs,
                "attention_mask": mask_mapping,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            },
            state={"modality_embeds": modality_embeds},
        )

    def call_model(self, *, model, state_context, wrapper) -> T.Any:
        outputs = model.model.language_model(**state_context.model_inputs)
        logits = model.lm_head(outputs.last_hidden_state)
        return {"logits": logits, "past_key_values": outputs.past_key_values}

    def build_forward_outputs(
        self, *, model, model_outputs, state_context, num_logits_to_keep
    ) -> T.List[torch.Tensor]:
        outputs = super().build_forward_outputs(
            model=model,
            model_outputs=model_outputs,
            state_context=state_context,
            num_logits_to_keep=num_logits_to_keep,
        )
        return outputs + list(state_context.state["modality_embeds"])
