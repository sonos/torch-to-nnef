"""Joint export of NemotronH Omni: RADIO vision, Parakeet audio, NemotronH LM.

- :class:`NemotronHOmniVisionEncoderHandler`: the RADIO tower, its pixel
  shuffle and the vision projector, for one image of any patch grid. Patches
  come grouped by 2x2 merge cell, `[MH, MW, 2, 2, patch_dim]`: RADIO attention
  has no relative positions, so the tower runs on tokens in that order (each
  with its own position embedding) and the pixel shuffle is a plain reshape.
- :class:`NemotronHOmniAudioEncoderHandler`: the Parakeet encoder and the audio
  projector, for one clip of any length.
- :class:`NemotronHOmniArchitectureHandler`: the NemotronH decoder (explicit
  Mamba-2 and KV states), with image and audio embeddings spliced at their
  placeholder tokens.
"""

import types
import typing as T

import torch
from torch import nn

from torch_to_nnef.tensor.offload import OffloadedTensor

from .base import (
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    StateContext,
    reset_special_ids_to_filler,
    scatter_features_by_mask,
)
from .gdn_reify import _materialize_offloaded
from .nemotron_h import (
    NemotronHArchitectureHandler,
    decoder_forward,
    install_reified_mamba,
    past_length,
)
from .registry import register_encoder_handler, register_handler

MERGE = 2


def _load_back(*modules) -> None:
    """Load disk-offloaded parameters and buffers of `modules` back in RAM.

    The towers are small next to the decoder and trace like plain modules
    once they hold real tensors, which the offload tracing strategy needs
    for their concatenations and position arithmetic.
    """
    for root in modules:
        if root is None:
            continue
        for module in root.modules():
            for name, _ in list(module.named_parameters(recurse=False)):
                _materialize_offloaded(module, name)
            for name, buffer in list(module.named_buffers(recurse=False)):
                if isinstance(buffer, OffloadedTensor):
                    setattr(module, name, buffer.reload().to(buffer.dtype))


def _merge_cell_positions(probe: torch.Tensor) -> torch.Tensor:
    """Patch indices `[n, 2]` along one axis of `n` merge cells.

    Built from a slice of the input (cumsum of ones) rather than from its
    shape: shape-derived integers trace as i64 casts that tract cannot keep
    symbolic.
    """
    cell = torch.cumsum(torch.ones_like(probe).float(), 0) - 1
    return torch.stack([MERGE * cell, MERGE * cell + 1], dim=-1)


def _bilinear_weights(index, size, table_size):
    """Bilinear weights of `index` positions after resizing to `size`.

    Row `i` holds, as tent functions, the align_corners=False weights of table
    position `index[i]` once the table is resized to `size`.
    """
    src = ((index + 0.5) * (table_size / size) - 0.5).clamp(
        0.0, float(table_size - 1)
    )
    # cast, not dtype=: the exported arange keeps an integer type otherwise
    grid = torch.arange(table_size, device=index.device).float()
    return torch.relu(1 - (src[..., None] - grid).abs())


class NemotronHOmniVisionEncoder(nn.Module):
    """RADIO tower + pixel shuffle + projector for one image.

    Input `[MH, MW, 2, 2, patch_dim]` (normalized, channel-major patches);
    output `[MH * MW, text_hidden]`. The position embedding is RADIO's CPE:
    the table bilinearly resized to the square of the longest side, then
    cropped to the grid.
    """

    def __init__(self, hf_model):
        super().__init__()
        vision = hf_model.vision_model
        embeddings = vision.embeddings
        self.patch_projection = embeddings.patch_projection
        self.position_embedding = embeddings.position_embedding
        self.cls_register_token = embeddings.cls_register_token
        self.table_size = embeddings.max_rows
        self.encoder = vision.encoder
        self.num_skip = vision.config.num_summary_tokens
        self.cell_width = MERGE * MERGE * vision.config.hidden_size
        self.final_layernorm = hf_model.vision_final_layernorm
        self.projector = hf_model.multi_modal_projector

    def position_embeddings(self, pixel_values):
        rows = _merge_cell_positions(pixel_values[:, 0, 0, 0, 0])
        cols = _merge_cell_positions(pixel_values[0, :, 0, 0, 0])
        size = torch.maximum(rows[-1, -1:], cols[-1, -1:]) + 1
        r = self.table_size
        table = self.position_embedding.reshape(r, r, -1).float()
        row_w = _bilinear_weights(rows, size, r)
        col_w = _bilinear_weights(cols, size, r)
        by_row = torch.matmul(row_w.reshape(-1, r), table.reshape(r, -1))
        by_row = by_row.reshape(-1, MERGE, r, table.shape[-1])
        return torch.einsum("ahkc,bwk->abhwc", by_row, col_w)

    def forward(self, pixel_values):
        # 2-D: the f32-accumulation linear rewrite takes ranks 2 to 4
        patches = self.patch_projection(
            pixel_values.reshape(-1, pixel_values.shape[-1])
        )
        pos = self.position_embeddings(pixel_values).to(patches.dtype)
        tokens = patches + pos.reshape(-1, patches.shape[-1])
        tokens = torch.cat([self.cls_register_token, tokens])[None]
        hidden = self.encoder(tokens).last_hidden_state[0]
        if self.final_layernorm is not None:
            hidden = self.final_layernorm(hidden)
        # Drop the summary tokens after grouping by merge cell, padding them to
        # a whole number of cells first: the length is then a multiple of the
        # cell size tract can divide by, and nothing divides the slice after.
        pad = -self.num_skip % (MERGE * MERGE)
        zeros = torch.zeros(pad, hidden.shape[-1], dtype=hidden.dtype)
        hidden = torch.cat([zeros.to(hidden.device), hidden])
        cells = hidden.reshape(-1, self.cell_width)
        return self.projector(cells[(self.num_skip + pad) // (MERGE * MERGE) :])


class NemotronHOmniAudioEncoder(nn.Module):
    """Parakeet encoder + projector for one clip `[1, frames, mel_bins]`."""

    def __init__(self, hf_model):
        super().__init__()
        self.audio_tower = hf_model.audio_tower
        self.projector = hf_model.embed_audio

    def forward(self, input_features):
        hidden = self.audio_tower(input_features=input_features)
        return self.projector(hidden.last_hidden_state)[0]


def _traceable_relative_positions(self, hidden_states):
    """Parakeet relative positions, same values as HF.

    HF reshapes them with the traced length, which then stays in the graph.
    """
    seq_length = hidden_states.shape[1]
    position_ids = torch.arange(
        seq_length - 1, -seq_length, -1, device=hidden_states.device
    ).float()
    freqs = position_ids[:, None] * self.inv_freq.float()
    pos = torch.stack([freqs.sin(), freqs.cos()], dim=-1).flatten(-2)
    return pos[None].to(hidden_states.dtype)


@register_encoder_handler
class NemotronHOmniVisionEncoderHandler(EncoderHandler):
    """RADIO vision tower and projector of NemotronH Omni."""

    MODALITY = "vision"
    ARCH_NAMES = ("nemotron_h_omni",)
    MODEL_INPUT_NAME = "pixel_values"
    TRACT_ASSERTS = ("MH >= 1", "MW >= 1")
    SAMPLE_MERGE_GRID = (2, 3)

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        return NemotronHOmniVisionEncoder(hf_model)

    def prepare_model_for_export(self, model) -> None:
        _load_back(
            model.vision_model,
            model.vision_final_layernorm,
            model.multi_modal_projector,
        )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        vision_conf = config_helper.conf.vision_config
        patch_dim = vision_conf.num_channels * vision_conf.patch_size**2
        pixel_values = torch.randn(
            (*self.SAMPLE_MERGE_GRID, MERGE, MERGE, patch_dim),
            dtype=inputs_dtype,
        )
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            dynamic_axes={"pixel_values": {0: "MH", 1: "MW"}},
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

    def manifest_input_contract(self, config_helper):
        patch = config_helper.conf.vision_config.patch_size
        return {
            "name": "pixel_values",
            "layout": ["MH", "MW", MERGE, MERGE, "patch_dim"],
            "host_prep": (
                "One image, resized to an even patch grid h x w and "
                "normalized; cut into channel-major patches of "
                f"{patch}x{patch} and group them by {MERGE}x{MERGE} cell: "
                f"[h // {MERGE}, w // {MERGE}, {MERGE}, {MERGE}, patch_dim], "
                "cells row-major, patches row-major within a cell."
            ),
        }


@register_encoder_handler
class NemotronHOmniAudioEncoderHandler(EncoderHandler):
    """Parakeet audio encoder and projector of NemotronH Omni."""

    MODALITY = "audio"
    ARCH_NAMES = ("nemotron_h_omni",)
    MODEL_INPUT_NAME = "input_features"
    # Parakeet's relative shift slices off one padded row
    TRACT_ASSERTS = ("FRAMES >= 1",)
    SAMPLE_FRAMES = 40

    @classmethod
    def is_available(cls, config) -> bool:
        return getattr(config, "audio_config", None) is not None

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        return NemotronHOmniAudioEncoder(hf_model)

    def prepare_model_for_export(self, model) -> None:
        _load_back(model.audio_tower, model.embed_audio)
        for module in model.modules():
            if type(module).__name__ == "ParakeetEncoderRelPositionalEncoding":
                module.forward = types.MethodType(
                    _traceable_relative_positions, module
                )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        audio_conf = config_helper.conf.audio_config
        input_features = torch.randn(
            (1, self.SAMPLE_FRAMES, audio_conf.num_mel_bins),
            dtype=inputs_dtype,
        )
        return IOSpec(
            inputs=(input_features,),
            input_names=["input_features"],
            output_names=["out_audio_embeddings"],
            dynamic_axes={"input_features": {1: "FRAMES"}},
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

    def manifest_input_contract(self, config_helper):
        return {
            "name": "input_features",
            "layout": [1, "FRAMES", "num_mel_bins"],
            "host_prep": (
                "One clip of log-mel features from the processor, cut to its "
                "valid frames (the feature mask): the graph takes no mask."
            ),
        }


@register_handler
class NemotronHOmniArchitectureHandler(NemotronHArchitectureHandler):
    """NemotronH Omni decoder: image and audio embeddings as inputs."""

    ARCH_NAMES = ("nemotron_h_omni",)
    SAMPLE_IMAGE_TOKENS = 2
    SAMPLE_AUDIO_TOKENS = 1

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.NemotronH_Omni_Reasoning_V3

    def prepare_model_for_export(self, model) -> None:
        install_reified_mamba(model)

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
        n_image, n_audio = self.SAMPLE_IMAGE_TOKENS, self.SAMPLE_AUDIO_TOKENS
        spec = super().build_input_spec(
            tokenizer=tokenizer,
            config_helper=config_helper,
            inputs_dtype=inputs_dtype,
            sample_text=sample_text,
            # room for the media placeholders and one text token
            n_input_tokens=max(n_input_tokens, n_image + n_audio + 1),
            n_past_input_tokens=n_past_input_tokens,
            real_kv_cache=real_kv_cache,
        )
        conf = config_helper.conf
        hidden = config_helper.decoder_conf.hidden_size
        input_ids = spec.inputs[0].clone()
        reset_special_ids_to_filler(
            input_ids,
            {conf.image_token_id, conf.audio_token_id},
            config_helper.decoder_conf.vocab_size,
        )
        # media placeholders first: the continuation probes then see all of
        # them in their prefix or none of them
        input_ids[:, :n_image] = conf.image_token_id
        input_ids[:, n_image : n_image + n_audio] = conf.audio_token_id
        return IOSpec(
            inputs=(
                input_ids,
                *spec.inputs[1:],
                torch.randn((n_image, hidden), dtype=inputs_dtype),
                torch.randn((n_audio, hidden), dtype=inputs_dtype),
            ),
            input_names=spec.input_names
            + ["in_image_embeddings", "in_audio_embeddings"],
            output_names=spec.output_names
            + ["out_image_embeddings", "out_audio_embeddings"],
            dynamic_axes={
                **spec.dynamic_axes,
                "in_image_embeddings": {0: "IMG"},
                "in_audio_embeddings": {0: "AUD"},
            },
        )

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        config = wrapper.model.config
        input_ids, states = inputs[0], list(inputs[1:-2])
        image_embeddings, audio_embeddings = inputs[-2:]
        mask, _, _ = self.build_causal_mask_with_past(
            seq_length=input_ids.shape[1],
            past_length=past_length(config.text_config, states),
            device=input_ids.device,
        )
        return StateContext(
            model_inputs={},
            state={
                "input_ids": input_ids,
                "states": states,
                "mask": mask,
                "image_embeddings": image_embeddings,
                "audio_embeddings": audio_embeddings,
            },
        )

    def call_model(self, *, model, state_context, wrapper) -> T.Any:
        del wrapper
        st = state_context.state
        lm = model.language_model
        input_ids = st["input_ids"]
        hidden = lm.model.embeddings(input_ids)
        for token_id, features in (
            (model.config.image_token_id, st["image_embeddings"]),
            (model.config.audio_token_id, st["audio_embeddings"]),
        ):
            hidden = scatter_features_by_mask(
                inputs_embeds=hidden,
                token_mask=input_ids == token_id,
                features=features,
            )
        hidden, new_states = decoder_forward(
            lm.model, lm.config, hidden, st["states"], st["mask"]
        )
        return {"logits": lm.lm_head(hidden), "new_states": new_states}

    def build_forward_outputs(
        self, *, model, model_outputs, state_context, num_logits_to_keep
    ) -> T.List[torch.Tensor]:
        outputs = super().build_forward_outputs(
            model=model,
            model_outputs=model_outputs,
            state_context=state_context,
            num_logits_to_keep=num_logits_to_keep,
        )
        st = state_context.state
        return outputs + [st["image_embeddings"], st["audio_embeddings"]]
