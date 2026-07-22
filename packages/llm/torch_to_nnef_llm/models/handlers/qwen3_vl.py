import inspect
import typing as T

import torch

from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache

from .base import EmbeddingContract, EncoderHandler, IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_encoder_handler, register_handler


class _IntSeqlenRotary(torch.nn.Module):
    """Wrap the vision rotary embedding to take a python-int ``seqlen``.

    Qwen's vision tower bakes most grid-derived sizes via ``grid_thw.tolist()``,
    but ``rot_pos_emb`` feeds ``grid_thw[:, 1:].max()`` (a 0-d tensor) into the
    rotary ``arange``. With grid_thw baked constant that max is constant, yet as
    a tensor it lowers to a dynamic ``tract_core_range`` whose bounds mix TDim
    and i64 (tract rejects it). Coercing ``seqlen`` to ``int`` makes the arange
    fold to constant bounds, exactly like the tower's other (working) aranges.
    Shared by Qwen2.5-VL and Qwen3-VL vision encoders.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, seqlen):
        return self.inner(int(seqlen))


def bake_vision_rotary_seqlen(visual: torch.nn.Module) -> torch.nn.Module:
    """Idempotently coerce ``visual.rotary_pos_emb`` to a python-int seqlen."""
    if not isinstance(visual.rotary_pos_emb, _IntSeqlenRotary):
        visual.rotary_pos_emb = _IntSeqlenRotary(visual.rotary_pos_emb)
    return visual


def bake_vision_rot_pos_emb(
    visual: torch.nn.Module, grid_thw: torch.Tensor
) -> torch.nn.Module:
    """Precompute ``rot_pos_emb`` for a fixed grid and inject it as a constant.

    Qwen3-VL's ``rot_pos_emb`` builds position ids with ``torch.empty`` +
    in-place slice assignment + advanced indexing, a pattern that does not
    lower correctly to tract. For a baked (fixed) grid its output is a
    constant, so evaluate it once and return that constant during tracing.
    """
    if getattr(visual.rot_pos_emb, "_t2n_baked", False):
        return visual
    with torch.no_grad():
        const = visual.rot_pos_emb(grid_thw)

    def _baked(*_args, _const=const, **_kwargs):
        return _const

    _baked._t2n_baked = True
    visual.rot_pos_emb = _baked
    return visual


def _deepstack_indexes(config) -> T.List[int]:
    """Vision-encoder layer indexes whose features DeepStack re-injects."""
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return []
    return list(getattr(vision_config, "deepstack_visual_indexes", []))


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

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Qwen3VLForConditionalGeneration

    def prepare_model_for_export(self, model) -> None:
        # Qwen3-VL currently hits SDPA masking issues during torch.jit tracing.
        # Force eager attention in export mode to keep the graph traceable.
        model.config._attn_implementation = "eager"
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            lang_config = model.model.language_model.config
            lang_config._attn_implementation = "eager"

    def _ensure_seq_length(
        self,
        sequence_length: int,
        num_image_tokens: int,
        num_video_tokens: int,
    ) -> int:
        minimal = 1 + num_image_tokens + num_video_tokens + 1
        return max(sequence_length, minimal)

    @staticmethod
    def _get_rope_index_kwargs(
        hf_model, mm_token_type_ids: torch.Tensor
    ) -> T.Dict[str, torch.Tensor]:
        signature = inspect.signature(hf_model.model.get_rope_index)
        if "mm_token_type_ids" in signature.parameters:
            return {"mm_token_type_ids": mm_token_type_ids}
        return {}

    @classmethod
    def _split_inputs(
        cls, inputs: T.Tuple[torch.Tensor, ...]
    ) -> T.Tuple[T.Tuple[torch.Tensor, ...], T.Tuple[torch.Tensor, ...]]:
        state_input_count = len(cls.STATE_INPUT_NAMES)
        return (
            inputs[1:-state_input_count],
            inputs[-state_input_count:],
        )

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
        if total_tokens == 0:
            return inputs_embeds
        if total_tokens != features.shape[0]:
            raise ValueError(
                f"feature/slot count mismatch: got {features.shape[0]} "
                f"feature(s) for {total_tokens} placeholder slot(s) in "
                "input_ids"
            )

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
    def _static_deepstack_process(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Static-shape replacement for the model's ``_deepstack_process``.

        Upstream does ``hidden[mask, :] += visual_embeds`` via boolean advanced
        indexing, whose output shape is data-dependent: NNEF shape inference
        then infers 0 masked rows and the ``+ visual_embeds`` broadcast fails
        (``[0, H] + [N, H]``). We scatter the embeds to the masked positions
        with a fixed shape (same ``index_select`` trick as
        ``_inject_token_features``) and add -- numerically identical, but
        exportable.
        """
        if visual_embeds.numel() == 0:
            return hidden_states
        batch_size, seq_length = visual_pos_masks.shape
        token_counts = visual_pos_masks.to(torch.long).sum(dim=-1)
        start_offsets = torch.cumsum(token_counts, dim=0) - token_counts
        slot_ids = visual_pos_masks.to(torch.long).cumsum(dim=-1)
        slot_ids = slot_ids + start_offsets.unsqueeze(-1)
        slot_ids = torch.where(
            visual_pos_masks, slot_ids, torch.zeros_like(slot_ids)
        )
        zero_feature = torch.zeros(
            (1, visual_embeds.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        feature_bank = torch.cat(
            [
                zero_feature,
                visual_embeds.to(hidden_states.device, hidden_states.dtype),
            ],
            dim=0,
        )
        gathered = feature_bank.index_select(0, slot_ids.reshape(-1)).view(
            batch_size, seq_length, hidden_states.shape[-1]
        )
        return hidden_states + gathered

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
        # Reset any random token that lands on a special id to a filler that is
        # itself not a special id, so the number of placeholder slots matches
        # exactly the embeddings we provide. A hard-coded reset value collides
        # when a special id equals it (e.g. image_token_id == 1), leaving stray
        # placeholders and a feature/slot count mismatch.
        specials = {vision_start_token_id, image_token_id, video_token_id}
        if vocab_size > len(specials):
            filler = next(t for t in range(vocab_size) if t not in specials)
            for special in specials:
                input_ids[input_ids == special] = filler
        input_ids[:, 0] = vision_start_token_id
        for idx in range(num_image_tokens):
            position = 1 + idx
            if position < effective_seq_len:
                input_ids[:, position] = image_token_id

        # DeepStack: one extra image-embedding input per vision-encoder index,
        # re-injected at the first N decoder layers. Without these the decoder
        # silently drops DeepStack (the landed single-splice gap), since with
        # pixel_values=None the model computes no deepstack_visual_embeds.
        hidden_size = config_helper.decoder_conf.hidden_size
        n_deepstack = len(_deepstack_indexes(config_helper.conf))
        deepstack_inputs = tuple(
            torch.randn((num_image_tokens, hidden_size), dtype=inputs_dtype)
            for _ in range(n_deepstack)
        )
        deepstack_input_names = [
            f"in_image_deepstack_{i}" for i in range(n_deepstack)
        ]
        deepstack_output_names = [
            f"out_image_deepstack_{i}" for i in range(n_deepstack)
        ]
        deepstack_axes = {
            name: {0: "IMG_DEEP"} for name in deepstack_input_names
        }

        return IOSpec(
            inputs=base_spec.inputs + state_spec.inputs + deepstack_inputs,
            input_names=base_spec.input_names
            + state_spec.input_names
            + deepstack_input_names,
            output_names=base_spec.output_names
            + state_spec.output_names
            + deepstack_output_names,
            dynamic_axes={
                **base_spec.dynamic_axes,
                **state_spec.dynamic_axes,
                **deepstack_axes,
            },
        )

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        hf_model = wrapper.model

        n_deepstack = len(_deepstack_indexes(hf_model.config))
        if n_deepstack:
            deepstack_embeds = list(inputs[-n_deepstack:])
            inputs = inputs[:-n_deepstack]
        else:
            deepstack_embeds = []

        input_ids = inputs[0]
        cache_tensors, state_inputs = self._split_inputs(inputs)
        (
            image_embeddings,
            video_embeddings,
            image_grid_thw,
            video_grid_thw,
            rope_deltas_state,
        ) = state_inputs
        past_key_values = build_past_kv_dyn_cache(cache_tensors)

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
                **self._get_rope_index_kwargs(hf_model, mm_token_type_ids),
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

        prev_rope_deltas = getattr(hf_model.model, "rope_deltas", None)
        last_rope_deltas = rope_deltas_current.detach().clone()
        hf_model.model.rope_deltas = rope_deltas_current

        # DeepStack: the top model builds visual_pos_masks + deepstack embeds
        # only when pixel_values is set. With pixel_values=None we inject them
        # into the language_model call ourselves via a forward-pre-hook, which
        # the text model already consumes (adds each at its first N layers).
        deepstack_handle = None
        deepstack_lm = None
        if deepstack_embeds:
            visual_pos_masks = input_ids == hf_model.config.image_token_id
            deepstack_lm = hf_model.model.language_model
            # Swap the model's data-dependent `hidden[mask, :] += embeds`
            # deepstack step for a static-shape scatter-add so the graph
            # exports (see `_static_deepstack_process`). Instance attribute so
            # it is called unbound; removed again in `build_forward_outputs`.
            deepstack_lm._deepstack_process = (
                lambda hidden_states, visual_pos_masks, visual_embeds: (
                    self._static_deepstack_process(
                        hidden_states, visual_pos_masks, visual_embeds
                    )
                )
            )

            def _inject_deepstack(module, args, kwargs):
                kwargs["visual_pos_masks"] = visual_pos_masks
                kwargs["deepstack_visual_embeds"] = deepstack_embeds
                return (args, kwargs)

            deepstack_handle = deepstack_lm.register_forward_pre_hook(
                _inject_deepstack, with_kwargs=True
            )

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
                "prev_rope_deltas": prev_rope_deltas,
                "last_rope_deltas": last_rope_deltas,
                "deepstack_embeds": deepstack_embeds,
                "deepstack_handle": deepstack_handle,
                "deepstack_lm": deepstack_lm,
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
            rope_deltas = state_context.state["last_rope_deltas"]
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = state_context.state["prev_rope_deltas"]

        deepstack_handle = state_context.state.get("deepstack_handle")
        if deepstack_handle is not None:
            deepstack_handle.remove()
        deepstack_lm = state_context.state.get("deepstack_lm")
        if deepstack_lm is not None and (
            "_deepstack_process" in vars(deepstack_lm)
        ):
            # remove the instance override so the class method resurfaces
            del deepstack_lm._deepstack_process

        return (
            outputs
            + [
                state_context.state["image_embeddings"],
                state_context.state["video_embeddings"],
                state_context.state["image_grid_thw"],
                state_context.state["video_grid_thw"],
                rope_deltas,
            ]
            + list(state_context.state.get("deepstack_embeds", []))
        )


class Qwen3VLVisionEncoder(torch.nn.Module):
    """Qwen3-VL vision tower with grid_thw baked constant.

    Emits the merged image embeddings plus one DeepStack feature tensor per
    ``deepstack_visual_indexes`` entry (each ``[num_tokens, out_hidden]``), all
    consumed by the decoder handler's DeepStack injection.
    """

    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        bake_vision_rot_pos_emb(visual, grid_thw)
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values: torch.Tensor):
        out = self.visual(pixel_values, grid_thw=self.grid_thw)
        return (out.pooler_output, *out.deepstack_features)


@register_encoder_handler
class Qwen3VLVisionEncoderHandler(EncoderHandler):
    """Encoder handler for the Qwen3-VL vision tower (main + DeepStack)."""

    MODALITY = "vision"
    ARCH_NAMES = ("qwen3_vl",)
    SAMPLE_GRID_THW = (1, 8, 8)

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        # bake the rotary arange length to a python int (see qwen2_5_vl)
        visual = bake_vision_rotary_seqlen(hf_model.model.visual)
        grid = torch.tensor([self.SAMPLE_GRID_THW], dtype=torch.long)
        return Qwen3VLVisionEncoder(visual, grid)

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
        n_deepstack = len(_deepstack_indexes(config_helper.conf))
        pixel_values = torch.randn((num_patches, patch_dim), dtype=inputs_dtype)
        output_names = ["out_image_embeddings"] + [
            f"out_image_deepstack_{i}" for i in range(n_deepstack)
        ]
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=output_names,
            # grid_thw baked constant -> fixed patch count (see qwen2_5_vl).
            dynamic_axes={},
        )

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        return StateContext(model_inputs={"pixel_values": inputs[0]}, state={})

    def build_forward_outputs(
        self, *, model_outputs, state_context
    ) -> T.List[torch.Tensor]:
        return list(model_outputs)

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        # DeepStack injects each collected feature stream into the FIRST
        # `n_deepstack` DECODER layers: HF adds `deepstack_visual_embeds[i]` at
        # decoder layer `i` (modeling: `layer_idx in range(len(embeds))`), so
        # the injection layers are `range(n_deepstack)`. (config
        # `deepstack_visual_indexes` = [8,16,24] are the VISION-tower blocks the
        # streams are COLLECTED from, not decoder injection layers.)
        n_deepstack = len(_deepstack_indexes(config_helper.conf))
        return [
            EmbeddingContract(
                modality="image",
                hidden_size=config_helper.conf.vision_config.out_hidden_size,
                placeholder_token_id_attr="image_token_id",
                # matches the decoder graph's ``in_image_embeddings`` symbol
                # (the input this contract feeds); the encoder tower itself is
                # fixed-shape (baked grid).
                dynamic_axis="IMG_STATE",
                injection_layers=tuple(range(n_deepstack)),
                deepstack_dynamic_axis="IMG_DEEP",
            )
        ]
