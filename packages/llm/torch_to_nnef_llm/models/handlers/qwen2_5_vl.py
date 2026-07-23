"""Joint-export handlers for Qwen2.5-VL vision-language models.

- :class:`Qwen25VLVisionEncoderHandler`: the vision tower (conv3d patch embed,
  2D-RoPE, window + full attention, patch merger). ``grid_thw`` is baked as a
  constant so the data-dependent window_index / cu_seqlens / rotary structure
  folds to constants at trace time, leaving only ``pixel_values`` flowing; the
  non-flash attention then splits into fixed-size windows.
- :class:`Qwen25VLArchitectureHandler`: the decoder graph. Reuses the landed
  Qwen3-VL decoder handler (image-embedding injection + mRoPE + rope_deltas);
  only the loaded model class differs (no DeepStack in Qwen2.5-VL).

Both handlers are validated in PyTorch (encoder output bit-exact vs
``get_image_features``, decoder wrapper self-consistent) and export to
tract in ``f32`` (the vision tower's bare ``NUMBERTYPE`` rotary-seqlen
scalar is handled by baking it via ``_IntSeqlenRotary``; see
``qwen3_vl``). Only the ``f16`` vision path is currently gated: not by
t2n, but by a tract ``-O`` optimizer bug on the ``einsum(acc=f32)``
accumulation pattern, fixed in tract main.
"""

import typing as T

import torch

from .base import (
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    StateContext,
    resolve_submodule,
)
from .qwen3_vl import (
    Qwen3VLArchitectureHandler,
    bake_vision_rotary_seqlen,
)
from .registry import register_encoder_handler, register_handler


class Qwen25VLVisionEncoder(torch.nn.Module):
    """Vision tower traced as one encoder graph, with grid_thw baked constant.

    Input ``pixel_values`` is the processor's flattened patch tensor of shape
    ``[num_patches, in_channels * temporal_patch * patch**2]``; output is the
    merger embeddings ``[num_patches // merge**2, out_hidden]`` ready for the
    decoder splice.
    """

    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=self.grid_thw).pooler_output


@register_encoder_handler
class Qwen25VLVisionEncoderHandler(EncoderHandler):
    """Encoder handler for the Qwen2.5-VL vision tower."""

    MODALITY = "vision"
    ARCH_NAMES = ("qwen2_5_vl",)
    #: Baked sample grid (t, h, w) in patch units; h,w multiples of merge size.
    SAMPLE_GRID_THW = (1, 8, 8)

    def _grid_tensor(self) -> torch.Tensor:
        return torch.tensor([self.SAMPLE_GRID_THW], dtype=torch.long)

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        visual = bake_vision_rotary_seqlen(
            resolve_submodule(hf_model, "model.visual")
        )
        return Qwen25VLVisionEncoder(visual, self._grid_tensor())

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        vision_conf = config_helper.conf.vision_config
        t, h, w = self.SAMPLE_GRID_THW
        num_patches = t * h * w
        patch_dim = (
            vision_conf.in_channels
            * vision_conf.temporal_patch_size
            * vision_conf.patch_size
            * vision_conf.patch_size
        )
        pixel_values = torch.randn((num_patches, patch_dim), dtype=inputs_dtype)
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            # grid_thw is baked constant, so the patch count is fixed; a dynamic
            # axis here is spurious and makes the tower's seq_len // merge_unit
            # reshape symbolically undivisible for tract.
            dynamic_axes={},
        )

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        return StateContext(model_inputs={"pixel_values": inputs[0]}, state={})

    def build_forward_outputs(
        self, *, model_outputs, state_context
    ) -> T.List[torch.Tensor]:
        return [model_outputs]

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        return [
            EmbeddingContract(
                modality="image",
                hidden_size=config_helper.conf.vision_config.out_hidden_size,
                placeholder_token_id_attr="image_token_id",
                # matches the (shared Qwen3-VL) decoder graph's
                # ``in_image_embeddings`` symbol; the encoder is fixed-shape.
                dynamic_axis="IMG_STATE",
            )
        ]


@register_handler
class Qwen25VLArchitectureHandler(Qwen3VLArchitectureHandler):
    """Decoder handler for Qwen2.5-VL (Qwen3-VL logic minus DeepStack)."""

    ARCH_NAMES = ("qwen2_5_vl",)

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Qwen2_5_VLForConditionalGeneration
