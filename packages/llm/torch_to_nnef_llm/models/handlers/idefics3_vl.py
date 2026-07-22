"""Joint-export handlers for Idefics3 / SmolVLM vision-language models.

Holds both halves of the arch:

- :class:`Idefics3VisionEncoderHandler`: exports the vision tower + connector
  (SigLIP-style ViT, pixel-shuffle, modality projection) as the encoder graph.
- :class:`Idefics3ArchitectureHandler`: the decoder graph, injecting the
  encoder's image embeddings into the token stream at ``image_token_id``
  positions (single splice at the embedding layer, no DeepStack).

SmolVLM-256M reports ``model_type == "idefics3"``; SmolVLM2 reports
``"smolvlm"``. Both share this modeling, so both arch names are registered.
"""

import typing as T

import torch

from .base import EmbeddingContract, EncoderHandler, IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_encoder_handler, register_handler


def _image_seq_len(conf) -> int:
    """Number of image tokens one full tile becomes after pixel-shuffle."""
    vision_conf = conf.vision_config
    patches_per_side = vision_conf.image_size // vision_conf.patch_size
    return int((patches_per_side**2) / (conf.scale_factor**2))


def _inject_image_features(
    *,
    inputs_embeds: torch.Tensor,
    token_mask: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``features`` into ``inputs_embeds`` where ``token_mask`` is set.

    ``features`` is the flat ``[num_image_tokens, hidden]`` encoder output; the
    i-th True slot in row-major order receives ``features[i]``. Written with a
    gather (not boolean assignment) so it survives tracing to tract.
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
    float_mask = token_mask.unsqueeze(-1).to(inputs_embeds.dtype)
    return inputs_embeds * (1 - float_mask) + gathered * float_mask


class Idefics3VisionEncoder(torch.nn.Module):
    """Vision tower + connector traced as one encoder graph.

    Takes already-tiled ``pixel_values`` of shape ``[num_tiles, 3, H, W]``
    (host does the image splitting, so every tile is a full square) and returns
    flat image embeddings ``[num_tiles * image_seq_len, text_hidden]`` ready for
    the decoder splice.

    The embedding + attention paths are inlined here rather than calling
    :meth:`Idefics3VisionTransformer.forward` to sidestep two trace hazards that
    only matter for variable-resolution / padded inputs, neither of which occur
    for full square tiles:

    - ``create_bidirectional_mask`` (an sdpa-mask helper that breaks under
      tracing): with all patches valid the mask is uniform, so the encoder runs
      with no attention mask.
    - the data-dependent position-id ``bucketize`` + boolean scatter in
      :class:`SmolVLMVisionEmbeddings`: for a full square tile it reduces
      exactly to a row-major ``arange`` grid.
    """

    def __init__(self, vision_model, connector):
        super().__init__()
        self.embeddings = vision_model.embeddings
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = connector
        self.num_patches = vision_model.embeddings.num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_tiles = pixel_values.shape[0]
        # match the reference get_image_features, which casts pixel_values to
        # the tower dtype before the patch conv (the exported input dtype need
        # not match the module dtype, e.g. bf16 weights with f16 inputs).
        pixel_values = pixel_values.to(
            self.embeddings.patch_embedding.weight.dtype
        )
        patch_embeds = self.embeddings.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        position_ids = torch.arange(
            self.num_patches, device=pixel_values.device
        ).expand(num_tiles, -1)
        embeddings = embeddings + self.embeddings.position_embedding(
            position_ids
        )
        hidden = self.encoder(
            inputs_embeds=embeddings, attention_mask=None
        ).last_hidden_state
        hidden = self.post_layernorm(hidden)
        features = self.connector(hidden)
        return features.reshape(-1, features.shape[-1])


@register_encoder_handler
class Idefics3VisionEncoderHandler(EncoderHandler):
    """Encoder handler for Idefics3 / SmolVLM vision towers."""

    MODALITY = "vision"
    ARCH_NAMES = ("idefics3", "smolvlm")

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        inner = hf_model.model
        return Idefics3VisionEncoder(inner.vision_model, inner.connector)

    def _text_hidden_size(self, config_helper) -> int:
        return config_helper.decoder_conf.hidden_size

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
                hidden_size=self._text_hidden_size(config_helper),
                placeholder_token_id_attr="image_token_id",
                dynamic_axis="IMG",
            )
        ]


@register_handler
class Idefics3ArchitectureHandler(DefaultArchitectureHandler):
    """Decoder handler for Idefics3 / SmolVLM: image embeddings as input."""

    ARCH_NAMES = ("idefics3", "smolvlm")
    STATE_INPUT_NAMES = ["in_image_embeddings"]
    STATE_OUTPUT_NAMES = ["out_image_embeddings"]

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.AutoModelForImageTextToText

    def prepare_model_for_export(self, model) -> None:
        # SDPA masking trips tracing; force eager attention on every sub-config.
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
        num_image_tokens = _image_seq_len(config_helper.conf)
        effective_seq_len = max(n_input_tokens, num_image_tokens + 2)
        base_spec = super().build_input_spec(
            tokenizer=tokenizer,
            config_helper=config_helper,
            inputs_dtype=inputs_dtype,
            sample_text=sample_text,
            n_input_tokens=effective_seq_len,
            n_past_input_tokens=n_past_input_tokens,
            real_kv_cache=real_kv_cache,
        )
        hidden_size = config_helper.decoder_conf.hidden_size
        image_embeddings = torch.randn(
            (num_image_tokens, hidden_size), dtype=inputs_dtype
        )

        input_ids = base_spec.inputs[0]
        image_token_id = config_helper.conf.image_token_id
        vocab_size = config_helper.decoder_conf.vocab_size
        input_ids.random_(0, vocab_size)
        if vocab_size > 1:
            input_ids[input_ids == image_token_id] = 1
        for idx in range(num_image_tokens):
            # leave position 0 for a real text token so decode still has one
            input_ids[:, 1 + idx] = image_token_id

        return IOSpec(
            inputs=base_spec.inputs + (image_embeddings,),
            input_names=base_spec.input_names + self.STATE_INPUT_NAMES,
            output_names=base_spec.output_names + self.STATE_OUTPUT_NAMES,
            dynamic_axes={
                **base_spec.dynamic_axes,
                "in_image_embeddings": {0: "IMG"},
            },
        )

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        hf_model = wrapper.model
        input_ids = inputs[0]
        image_embeddings = inputs[-1]
        cache_tensors = inputs[1:-1]

        # Reuse the default causal-mask / position-id construction.
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
        base_ctx.model_inputs["input_ids"] = None
        base_ctx.model_inputs["inputs_embeds"] = inputs_embeds
        base_ctx.model_inputs["pixel_values"] = None
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
