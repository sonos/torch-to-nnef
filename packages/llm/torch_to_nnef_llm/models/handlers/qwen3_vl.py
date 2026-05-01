import typing as T

import torch

from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache

from .base import IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class Qwen3VLArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Qwen3-VL models."""

    ARCH_NAMES = ("qwen3_vl",)
    STATE_INPUT_NAMES = [
        "in_image_embeddings",
        "in_video_embeddings",
        "in_image_grid_thw",
        "in_video_grid_thw",
        "in_rope_deltas",
    ]
    STATE_OUTPUT_NAMES = [
        "out_image_embeddings",
        "out_video_embeddings",
        "out_image_grid_thw",
        "out_video_grid_thw",
        "out_rope_deltas",
    ]
    SAMPLE_IMAGE_GRID_THW = (1, 4, 4)

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

    @staticmethod
    def _build_causal_attention_mask(
        *,
        batch_size: int,
        query_length: int,
        past_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        total_kv_len = past_seq_len + query_length
        kv_positions = torch.arange(total_kv_len, device=device).view(
            1, 1, total_kv_len
        )
        query_positions = (
            past_seq_len + torch.arange(query_length, device=device)
        ).view(1, query_length, 1)
        visible = kv_positions <= query_positions
        mask = torch.full(
            (1, query_length, total_kv_len),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        mask = mask.masked_fill(visible, 0)
        return mask.unsqueeze(1).expand(batch_size, -1, -1, -1)

    @staticmethod
    def _build_mm_token_type_ids(
        input_ids: torch.Tensor,
        *,
        image_token_id: int,
        video_token_id: int,
    ) -> torch.Tensor:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == image_token_id] = 1
        mm_token_type_ids[input_ids == video_token_id] = 2
        return mm_token_type_ids

    @staticmethod
    def _inject_token_features(
        *,
        inputs_embeds: torch.Tensor,
        token_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.numel() == 0:
            return inputs_embeds

        batch_size, seq_length = token_mask.shape
        token_counts = token_mask.to(torch.long).sum(dim=-1)
        total_tokens = int(token_counts.sum().item())
        if total_tokens == 0 or total_tokens != features.shape[0]:
            return inputs_embeds

        start_offsets = torch.cumsum(token_counts, dim=0) - token_counts
        slot_ids = token_mask.to(torch.long).cumsum(dim=-1)
        slot_ids = slot_ids + start_offsets.unsqueeze(-1)
        slot_ids = torch.where(
            token_mask,
            slot_ids,
            torch.zeros_like(slot_ids),
        )

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
            batch_size,
            seq_length,
            inputs_embeds.shape[-1],
        )
        token_mask = token_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        return inputs_embeds * (1 - token_mask) + gathered * token_mask

    @staticmethod
    def _build_cached_position_ids(
        *,
        rope_deltas: torch.Tensor,
        past_seq_len: int,
        seq_length: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if rope_deltas.ndim == 1:
            rope_deltas = rope_deltas.unsqueeze(-1)

        base_positions = torch.arange(
            seq_length, device=device, dtype=torch.long
        ).view(1, 1, -1)
        base_positions = base_positions.repeat(3, batch_size, 1)
        delta = (past_seq_len + rope_deltas).view(1, batch_size, 1)
        return (base_positions + delta).to(dtype=torch.long)

    def _build_state_spec(
        self,
        *,
        config_helper,
        inputs_dtype: torch.dtype,
    ) -> IOSpec:
        hidden_size = config_helper.decoder_conf.hidden_size
        vision_conf = config_helper.conf.vision_config
        image_grid = torch.tensor(
            [self.SAMPLE_IMAGE_GRID_THW], dtype=torch.long
        )
        num_image_tokens = int(
            (image_grid.prod(-1) // (vision_conf.spatial_merge_size**2)).item()
        )
        image_embeddings = torch.randn(
            (num_image_tokens, hidden_size), dtype=inputs_dtype
        )

        return IOSpec(
            inputs=(
                image_embeddings,
                torch.zeros((0, hidden_size), dtype=inputs_dtype),
                image_grid,
                torch.zeros((0, 3), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
            ),
            input_names=self.STATE_INPUT_NAMES,
            output_names=self.STATE_OUTPUT_NAMES,
            dynamic_axes={
                "in_image_embeddings": {0: "IMG_STATE"},
                "in_video_embeddings": {0: "VID_STATE"},
                "in_image_grid_thw": {0: "IMG_GRID"},
                "in_video_grid_thw": {0: "VID_GRID"},
            },
        )

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
        vision_conf = config_helper.conf.vision_config
        image_grid = torch.tensor(
            [self.SAMPLE_IMAGE_GRID_THW], dtype=torch.long
        )
        num_image_tokens = int(
            (image_grid.prod(-1) // (vision_conf.spatial_merge_size**2)).item()
        )
        effective_seq_len = self._ensure_seq_length(
            n_input_tokens, num_image_tokens, 0
        )

        base_spec = super().build_input_spec(
            tokenizer=tokenizer,
            config_helper=config_helper,
            inputs_dtype=inputs_dtype,
            sample_text=sample_text,
            n_input_tokens=effective_seq_len,
            n_past_input_tokens=n_past_input_tokens,
            real_kv_cache=real_kv_cache,
        )
        state_spec = self._build_state_spec(
            config_helper=config_helper,
            inputs_dtype=inputs_dtype,
        )

        input_ids = base_spec.inputs[0]
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
        input_ids[:, 0] = vision_start_token_id
        for idx in range(num_image_tokens):
            position = 1 + idx
            if position < effective_seq_len:
                input_ids[:, position] = image_token_id

        return IOSpec(
            inputs=base_spec.inputs + state_spec.inputs,
            input_names=base_spec.input_names + state_spec.input_names,
            output_names=base_spec.output_names + state_spec.output_names,
            dynamic_axes={**base_spec.dynamic_axes, **state_spec.dynamic_axes},
        )

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        hf_model = wrapper.model

        input_ids = inputs[0]
        cache_tensors = inputs[1 : 1 + wrapper.num_kv_tensors]
        (
            image_embeddings,
            video_embeddings,
            image_grid_thw,
            video_grid_thw,
            rope_deltas_state,
        ) = inputs[1 + wrapper.num_kv_tensors :]
        past_key_values = build_past_kv_dyn_cache(cache_tensors)
        self._last_state_outputs = (
            image_embeddings,
            video_embeddings,
            image_grid_thw,
            video_grid_thw,
        )

        inputs_embeds = hf_model.get_input_embeddings()(input_ids)
        image_token_id = hf_model.config.image_token_id
        video_token_id = hf_model.config.video_token_id
        mm_token_type_ids = self._build_mm_token_type_ids(
            input_ids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )

        inputs_embeds = self._inject_token_features(
            inputs_embeds=inputs_embeds,
            token_mask=input_ids == image_token_id,
            features=image_embeddings,
        )
        inputs_embeds = self._inject_token_features(
            inputs_embeds=inputs_embeds,
            token_mask=input_ids == video_token_id,
            features=video_embeddings,
        )

        past_seq_len = cache_tensors[0].shape[-2] if cache_tensors else 0
        rope_attention_mask = torch.ones(
            (input_ids.shape[0], past_seq_len + input_ids.shape[1]),
            dtype=torch.long,
            device=input_ids.device,
        )
        attention_mask = self._build_causal_attention_mask(
            batch_size=input_ids.shape[0],
            query_length=input_ids.shape[1],
            past_seq_len=past_seq_len,
            dtype=inputs_embeds.dtype,
            device=input_ids.device,
        )
        image_grid_arg = image_grid_thw if image_grid_thw.numel() else None
        video_grid_arg = video_grid_thw if video_grid_thw.numel() else None

        if past_seq_len == 0 or rope_deltas_state.numel() == 0:
            position_ids, rope_deltas_current = hf_model.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_arg,
                video_grid_thw=video_grid_arg,
                attention_mask=rope_attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            position_ids = position_ids.to(device=input_ids.device)
            rope_deltas_current = rope_deltas_current.to(
                device=input_ids.device, dtype=torch.long
            )
        else:
            rope_deltas_current = rope_deltas_state.to(
                device=input_ids.device, dtype=torch.long
            )
            position_ids = self._build_cached_position_ids(
                rope_deltas=rope_deltas_current,
                past_seq_len=past_seq_len,
                seq_length=input_ids.shape[1],
                batch_size=input_ids.shape[0],
                device=input_ids.device,
            )

        self._prev_rope_deltas = getattr(hf_model.model, "rope_deltas", None)
        self._last_rope_deltas = rope_deltas_current.detach().clone()
        hf_model.model.rope_deltas = rope_deltas_current

        return StateContext(
            model_inputs={
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
                "position_ids": position_ids,
            },
            state={
                "image_embeddings": image_embeddings,
                "video_embeddings": video_embeddings,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas_state": rope_deltas_state,
            },
        )

    def call_model(
        self,
        *,
        model,
        state_context: StateContext,
        wrapper,
    ) -> T.Any:
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
        outputs = super().build_forward_outputs(
            model=model,
            model_outputs=model_outputs,
            state_context=state_context,
            num_logits_to_keep=num_logits_to_keep,
        )
        rope_deltas = getattr(model_outputs, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = self._last_rope_deltas
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = self._prev_rope_deltas
        self._prev_rope_deltas = None
        if rope_deltas is None and self._last_rope_deltas is not None:
            rope_deltas = self._last_rope_deltas

        return outputs + [
            state_context.state["image_embeddings"],
            state_context.state["video_embeddings"],
            state_context.state["image_grid_thw"],
            state_context.state["video_grid_thw"],
            rope_deltas,
        ]
