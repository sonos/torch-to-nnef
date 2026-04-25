import typing as T

import torch

from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache

from .base import DecoderStateSpec, InputSpec
from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class Qwen3VLArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Qwen3-VL models."""

    ARCH_NAMES = ("qwen3_vl",)

    def __init__(self):
        super().__init__()
        self._prev_rope_deltas: T.Optional[torch.Tensor] = None
        self._last_rope_deltas: T.Optional[torch.Tensor] = None
        self._last_state_outputs: T.Tuple[torch.Tensor, ...] = ()

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Qwen3VLForConditionalGeneration

    def _ensure_seq_length(
        self,
        sequence_length: int,
        num_image_tokens: int,
        num_video_tokens: int,
    ) -> int:
        minimal = 1 + num_image_tokens + num_video_tokens + 1
        return max(sequence_length, minimal)

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
    ) -> InputSpec:
        image_grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        vision_conf = config_helper.conf.vision_config
        num_image_tokens = int(
            (image_grid.prod(-1) // (vision_conf.spatial_merge_size**2)).item()
        )
        num_video_tokens = 0
        effective_seq_len = self._ensure_seq_length(
            n_input_tokens, num_image_tokens, num_video_tokens
        )

        spec = super().build_input_spec(
            tokenizer=tokenizer,
            config_helper=config_helper,
            inputs_dtype=inputs_dtype,
            sample_text=sample_text,
            n_input_tokens=effective_seq_len,
            n_past_input_tokens=n_past_input_tokens,
            real_kv_cache=real_kv_cache,
        )

        input_ids = spec.inputs[0]
        vocab_size = config_helper.decoder_conf.vocab_size
        vision_start_token_id = getattr(
            config_helper.conf,
            "vision_start_token_id",
            config_helper.conf.image_token_id - 1,
        )
        image_token_id = config_helper.conf.image_token_id
        video_token_id = config_helper.conf.video_token_id

        input_ids.random_(0, vocab_size)
        if vocab_size > 10:
            input_ids[input_ids == image_token_id] = 1
            input_ids[input_ids == video_token_id] = 2

        if effective_seq_len >= 1:
            input_ids[:, 0] = vision_start_token_id
        if effective_seq_len >= 1 + num_image_tokens:
            for idx in range(num_image_tokens):
                position = 1 + idx
                if position < effective_seq_len:
                    input_ids[:, position] = image_token_id

        return spec

    def build_decoder_state_spec(
        self,
        *,
        config_helper,
        inputs_dtype: torch.dtype,
        n_input_tokens: int,
        n_past_input_tokens: int,
    ) -> DecoderStateSpec:
        hidden_size = config_helper.decoder_conf.hidden_size
        vision_conf = config_helper.conf.vision_config

        image_grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        num_image_tokens = int(
            (image_grid.prod(-1) // (vision_conf.spatial_merge_size**2)).item()
        )
        image_embeddings = torch.randn(
            (num_image_tokens, hidden_size), dtype=inputs_dtype
        )

        video_grid = torch.zeros((0, 3), dtype=torch.long)
        video_embeddings = torch.zeros((0, hidden_size), dtype=inputs_dtype)
        rope_deltas = torch.zeros((1, 1), dtype=torch.long)

        return DecoderStateSpec(
            inputs=(
                image_embeddings,
                video_embeddings,
                image_grid,
                video_grid,
                rope_deltas,
            ),
            input_names=[
                "in_image_embeddings",
                "in_video_embeddings",
                "in_image_grid_thw",
                "in_video_grid_thw",
                "in_rope_deltas",
            ],
            output_names=[
                "out_image_embeddings",
                "out_video_embeddings",
                "out_image_grid_thw",
                "out_video_grid_thw",
                "out_rope_deltas",
            ],
            dynamic_axes={
                "in_image_embeddings": {0: "IMG_STATE"},
                "in_video_embeddings": {0: "VID_STATE"},
                "in_image_grid_thw": {0: "IMG_GRID"},
                "in_video_grid_thw": {0: "VID_GRID"},
            },
        )

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> T.Dict[str, T.Any]:
        hf_model = wrapper.model
        input_ids = inputs[0]
        num_kv_tensors = wrapper.num_kv_tensors
        cache_tensors = inputs[1 : 1 + num_kv_tensors]
        state_tensors = inputs[1 + num_kv_tensors :]

        past_key_values = build_past_kv_dyn_cache(cache_tensors)

        (
            image_embeddings,
            video_embeddings,
            image_grid_thw,
            video_grid_thw,
            rope_deltas_state,
        ) = state_tensors
        self._last_state_outputs = (
            image_embeddings,
            video_embeddings,
            image_grid_thw,
            video_grid_thw,
        )

        embed_layer = hf_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        image_token_id = hf_model.config.image_token_id
        video_token_id = hf_model.config.video_token_id
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == image_token_id] = 1
        mm_token_type_ids[input_ids == video_token_id] = 2

        def _maybe_scatter(features: torch.Tensor, token_id: int):
            token_count = int((input_ids == token_id).sum().item())
            if token_count == 0 or features.numel() == 0:
                return None
            if features.shape[0] != token_count:
                return None
            return features.to(inputs_embeds.device, inputs_embeds.dtype)

        image_features = _maybe_scatter(image_embeddings, image_token_id)
        video_features = _maybe_scatter(video_embeddings, video_token_id)

        if image_features is not None or video_features is not None:
            image_mask, video_mask = hf_model.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
            if image_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask, image_features
                )
            if video_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    video_mask, video_features
                )

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        past_seq_len = (
            past_key_values.get_seq_length() if past_key_values else 0
        )
        cache_position = torch.arange(
            past_seq_len,
            past_seq_len + input_ids.shape[1],
            device=input_ids.device,
        )

        image_grid_arg = image_grid_thw if image_grid_thw.numel() else None
        video_grid_arg = video_grid_thw if video_grid_thw.numel() else None

        if past_seq_len == 0 or rope_deltas_state.numel() == 0:
            position_ids, rope_deltas_current = hf_model.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_arg,
                video_grid_thw=video_grid_arg,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            position_ids = position_ids.to(device=input_ids.device)
            rope_deltas_current = rope_deltas_current.to(
                device=input_ids.device, dtype=torch.long
            )
        else:
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]

            rope_deltas_current = rope_deltas_state.to(
                device=input_ids.device, dtype=torch.long
            )
            if rope_deltas_current.ndim == 1:
                rope_deltas_current = rope_deltas_current.unsqueeze(-1)

            base_positions = torch.arange(
                seq_length, device=input_ids.device, dtype=torch.long
            ).view(1, 1, -1)
            base_positions = base_positions.repeat(3, batch_size, 1)

            cache_offset = cache_position[0] if cache_position.numel() else 0
            delta = cache_offset + rope_deltas_current
            delta = delta.view(1, batch_size, 1)

            position_ids = (base_positions + delta).to(dtype=torch.long)

        prev_rope = getattr(hf_model.model, "rope_deltas", None)
        hf_model.model.rope_deltas = rope_deltas_current
        self._prev_rope_deltas = prev_rope
        self._last_rope_deltas = rope_deltas_current.detach().clone()

        return {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": image_grid_arg,
            "video_grid_thw": video_grid_arg,
            "mm_token_type_ids": mm_token_type_ids,
            "cache_position": cache_position,
            "position_ids": position_ids,
        }

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs: T.Any,
        model_inputs: T.Dict[str, T.Any],
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        outputs = super().build_forward_outputs(
            model=model,
            model_outputs=model_outputs,
            model_inputs=model_inputs,
            num_logits_to_keep=num_logits_to_keep,
        )
        rope_deltas = getattr(model_outputs, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = self._last_rope_deltas

        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = self._prev_rope_deltas
        self._prev_rope_deltas = None

        return outputs + list(self._last_state_outputs) + [rope_deltas]

    def prepare_additional_outputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        prepared_inputs: T.Dict[str, T.Any],
        hf_outputs,
        wrapper,
    ) -> T.List[torch.Tensor]:
        rope_deltas = getattr(hf_outputs, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = inputs[-1]

        hf_model = wrapper.model
        if hasattr(hf_model.model, "rope_deltas"):
            hf_model.model.rope_deltas = self._prev_rope_deltas
        self._prev_rope_deltas = None

        if rope_deltas is None and self._last_rope_deltas is not None:
            rope_deltas = self._last_rope_deltas

        return list(inputs[-5:-1]) + [rope_deltas]
