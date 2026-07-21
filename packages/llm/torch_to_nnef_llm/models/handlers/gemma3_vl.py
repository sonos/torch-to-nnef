"""Joint-export handlers for Gemma 3 vision-language models.

- :class:`Gemma3VisionEncoderHandler`: SigLIP vision tower + multimodal
  projector (4x4 avg-pool of the 64x64 patch grid to 256 soft tokens) as the
  encoder graph.
- :class:`Gemma3ArchitectureHandler`: the decoder graph. Gemma 3 attends
  *bidirectionally* within each image-token span (text stays causal) and
  interleaves full-attention and sliding-window layers, so the handler builds
  the ``{"full_attention", "sliding_attention"}`` additive mask mapping itself
  and passes it to the model, bypassing the fragile ``is_first_iteration``
  prefill heuristic in ``create_causal_mask_mapping``.
"""

import typing as T

import torch

from .base import EmbeddingContract, EncoderHandler, IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .idefics3_vl import _inject_image_features
from .registry import register_encoder_handler, register_handler


class Gemma3VisionEncoder(torch.nn.Module):
    """SigLIP tower + multimodal projector traced as one encoder graph.

    Takes tiled ``pixel_values`` ``[num_tiles, 3, 896, 896]`` and returns flat
    image embeddings ``[num_tiles * mm_tokens_per_image, text_hidden]``.
    """

    def __init__(self, vision_tower, projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        features = self.projector(hidden)
        return features.reshape(-1, features.shape[-1])


@register_encoder_handler
class Gemma3VisionEncoderHandler(EncoderHandler):
    """Encoder handler for Gemma 3 SigLIP vision towers."""

    MODALITY = "vision"
    ARCH_NAMES = ("gemma3",)

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        inner = hf_model.model
        return Gemma3VisionEncoder(
            inner.vision_tower, inner.multi_modal_projector
        )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        vision_conf = config_helper.conf.vision_config
        image_size = vision_conf.image_size
        pixel_values = torch.randn(
            (1, vision_conf.num_channels, image_size, image_size),
            dtype=inputs_dtype,
        )
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            dynamic_axes={"pixel_values": {0: "TILES"}},
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
                hidden_size=config_helper.decoder_conf.hidden_size,
                placeholder_token_id_attr="image_token_id",
                dynamic_axis="IMG",
            )
        ]


@register_handler
class Gemma3ArchitectureHandler(DefaultArchitectureHandler):
    """Decoder handler for Gemma 3: image embeddings as input."""

    ARCH_NAMES = ("gemma3",)
    STATE_INPUT_NAMES = ["in_image_embeddings"]
    STATE_OUTPUT_NAMES = ["out_image_embeddings"]

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.AutoModelForImageTextToText

    def prepare_model_for_export(self, model) -> None:
        for conf in (
            model.config,
            getattr(model.config, "text_config", None),
            getattr(model.config, "vision_config", None),
        ):
            if conf is not None:
                conf._attn_implementation = "eager"

    def _mm_tokens_per_image(self, config_helper) -> int:
        return int(config_helper.conf.mm_tokens_per_image)

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
        # A single image already needs mm_tokens_per_image (256) placeholder
        # tokens, more than EN_SAMPLE_TEXT tokenizes to, so build input_ids
        # directly instead of slicing the tokenized sample (which the default
        # handler asserts is long enough).
        del tokenizer, sample_text
        num_image_tokens = self._mm_tokens_per_image(config_helper)
        effective_seq_len = max(n_input_tokens, num_image_tokens + 2)
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
        image_token_id = config_helper.conf.image_token_id

        input_ids = torch.randint(0, vocab_size, (1, effective_seq_len))
        input_ids[input_ids == image_token_id] = 1
        for idx in range(num_image_tokens):
            input_ids[:, 1 + idx] = image_token_id
        image_embeddings = torch.randn(
            (num_image_tokens, hidden_size), dtype=inputs_dtype
        )

        return IOSpec(
            inputs=(input_ids, *past_key_values, image_embeddings),
            input_names=["input_ids"] + in_cache_names + self.STATE_INPUT_NAMES,
            output_names=["outputs"]
            + out_cache_names
            + self.STATE_OUTPUT_NAMES,
            dynamic_axes={**dynamic_axes, "in_image_embeddings": {0: "IMG"}},
        )

    @staticmethod
    def _build_mask_mapping(
        *,
        input_ids: torch.Tensor,
        image_token_id: int,
        past_length: int,
        sliding_window: int,
        dtype: torch.dtype,
    ) -> T.Dict[str, torch.Tensor]:
        """Additive full + sliding masks with bidirectional image spans.

        Text is causal; tokens of the same image group attend each other
        bidirectionally. Built from token positions with arange comparisons so
        it stays correct for any (S, P) and exports to tract.
        """
        # Built with float arithmetic (comparisons cast straight to `dtype`,
        # OR via add+clamp, AND via multiply) rather than boolean `&`/`|` on
        # tensors: t2n's shape-inference re-runs bitwise ops with a float
        # placeholder and fails ("bitwise_and not implemented for Float").
        device = input_ids.device
        seq_length = input_ids.shape[1]
        total = past_length + seq_length
        q_pos = (past_length + torch.arange(seq_length, device=device)).view(
            seq_length, 1
        )
        k_pos = torch.arange(total, device=device).view(1, total)

        causal_f = (k_pos <= q_pos).to(dtype)  # 1 where causally visible

        # Bidirectional attention within the image span. Assumes a single
        # contiguous image block (one image), the current export target;
        # multiple images would need per-image group ids (deferred).
        is_image_q = (
            (input_ids[0] == image_token_id).to(dtype).view(seq_length, 1)
        )
        is_image_all = (input_ids[0] == image_token_id).to(dtype)
        is_image_kv = torch.cat(
            [
                torch.zeros(past_length, dtype=dtype, device=device),
                is_image_all,
            ]
        ).view(1, total)
        bidir_f = is_image_q * is_image_kv  # 1 where both are image tokens

        visible_full = torch.clamp(causal_f + bidir_f, max=1.0)
        within_window_f = (k_pos > (q_pos - sliding_window)).to(dtype)
        visible_sliding = visible_full * within_window_f

        neg = torch.finfo(dtype).min

        def to_additive(visible: torch.Tensor) -> torch.Tensor:
            mask = (1.0 - visible) * neg
            return mask.unsqueeze(0).unsqueeze(0)

        return {
            "full_attention": to_additive(visible_full),
            "sliding_attention": to_additive(visible_sliding),
        }

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        hf_model = wrapper.model
        input_ids = inputs[0]
        image_embeddings = inputs[-1]
        cache_tensors = inputs[1:-1]

        base_ctx = super().build_forward_inputs(
            inputs=(input_ids, *cache_tensors), wrapper=wrapper
        )

        inputs_embeds = hf_model.get_input_embeddings()(input_ids)
        image_token_id = hf_model.config.image_token_id
        inputs_embeds = _inject_image_features(
            inputs_embeds=inputs_embeds,
            token_mask=input_ids == image_token_id,
            features=image_embeddings,
        )

        past_length = cache_tensors[0].shape[2] if cache_tensors else 0
        sliding_window = config_sliding_window(hf_model.config)
        mask_mapping = self._build_mask_mapping(
            input_ids=input_ids,
            image_token_id=image_token_id,
            past_length=past_length,
            sliding_window=sliding_window,
            dtype=inputs_embeds.dtype,
        )

        base_ctx.model_inputs["input_ids"] = None
        base_ctx.model_inputs["inputs_embeds"] = inputs_embeds
        base_ctx.model_inputs["pixel_values"] = None
        base_ctx.model_inputs["attention_mask"] = mask_mapping
        base_ctx.state = {"image_embeddings": image_embeddings}
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
        return outputs + [state_context.state["image_embeddings"]]


def config_sliding_window(config) -> int:
    text_config = getattr(config, "text_config", config)
    return int(getattr(text_config, "sliding_window", 4096) or 4096)
