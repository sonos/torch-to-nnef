import inspect
import typing as T

import torch

from torch_to_nnef.exceptions import T2NErrorConsistency
from torch_to_nnef_llm.models.base import build_past_kv_dyn_cache

from .base import (
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    StateContext,
    deepstack_input_name,
    deepstack_output_name,
    reset_special_ids_to_filler,
    resolve_submodule,
    scatter_features_by_mask,
)
from .default import DefaultArchitectureHandler
from .registry import register_encoder_handler, register_handler


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
        # Qwen3-VL hits SDPA masking issues during torch.jit tracing, so force
        # eager to keep the decoder graph traceable. This runs during the
        # decoder dump, after the exporter's global fp16 SDPA routing, so the
        # qwen3 decoder stays eager in both f32 and f16 (only the vision tower,
        # whose encoder handler does not force eager, takes the fp16 SDPA path).
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
        return scatter_features_by_mask(
            inputs_embeds=inputs_embeds,
            token_mask=token_mask,
            features=features,
        )

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
        (``[0, H] + [N, H]``). ``scatter_features_by_mask(additive=True)``
        scatters the embeds to the masked positions with a fixed shape and adds
        -- numerically identical, but exportable.
        """
        return scatter_features_by_mask(
            inputs_embeds=hidden_states,
            token_mask=visual_pos_masks,
            features=visual_embeds,
            additive=True,
        )

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
        reset_special_ids_to_filler(
            input_ids,
            {vision_start_token_id, image_token_id, video_token_id},
            vocab_size,
        )
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
            deepstack_input_name("image", i) for i in range(n_deepstack)
        ]
        deepstack_output_names = [
            deepstack_output_name("image", i) for i in range(n_deepstack)
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
            # exports (see `_static_deepstack_process`). Assigned as an instance
            # attribute of a staticmethod: the model calls it unbound with
            # (hidden_states, visual_pos_masks, visual_embeds). Removed again in
            # build_forward_outputs.
            deepstack_lm._deepstack_process = self._static_deepstack_process

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

    def cleanup(self, *, state_context, wrapper) -> None:
        # Undo the model mutations build_forward_inputs made to fake a decode
        # step: the rope_deltas attribute, the DeepStack forward-pre-hook and
        # the `_deepstack_process` instance override. Runs in a finally so a
        # forward that raises mid-trace does not leak them into later traces.
        state = state_context.state
        model = wrapper.model
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = state.get("prev_rope_deltas")
        deepstack_handle = state.get("deepstack_handle")
        if deepstack_handle is not None:
            deepstack_handle.remove()
        deepstack_lm = state.get("deepstack_lm")
        if deepstack_lm is not None and "_deepstack_process" in vars(
            deepstack_lm
        ):
            # remove the instance override so the class method resurfaces
            del deepstack_lm._deepstack_process


class Qwen3VLVisionEncoder(torch.nn.Module):
    """Dynamic-resolution Qwen3-VL vision tower (single frame).

    Input ``pixel_values`` is the processor's patch tensor reshaped to the 2D
    merge-block grid ``[MH, MW, merge, merge, patch_dim]`` (``MH = h // merge``,
    ``MW = w // merge`` both dynamic); flattening dims 0..3 restores the
    processor's merge-block-major patch order, so the reshape is host-side only.
    Both position tables (the 2D-RoPE frequencies and the interpolated learned
    ``pos_embed``) are rebuilt from ``arange(shape)`` rather than
    ``grid_thw.tolist()``, so the grid axes stay symbolic in NNEF: the tower
    exports once and runs at any resolution, no baking. The attention is the
    tower's full attention (``cu_seqlens`` spans the whole single image),
    reimplemented inline so nothing depends on the data-valued ``grid_thw``.
    Emits merged embeddings + one DeepStack stream per collected block.

    Mirrors ``Qwen3VLVisionModel.forward`` in transformers
    ``modeling_qwen3_vl``; the no-download DeepStack chain-parity test
    (``test_dummy_deepstack_chain_parity``) compares this against HF's native
    forward, so a transformers bump that changes the tower is caught.
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.merge = visual.spatial_merge_size
        self.num_grid = visual.num_grid_per_side
        self.deep_idx = list(visual.deepstack_visual_indexes)
        # `_pos_embed` divides by (h - 1) with h = mh * merge (mh >= 1); a
        # merge >= 2 keeps h >= 2 so the denominator is >= 1. Qwen3-VL always
        # uses spatial_merge_size == 2; guard the assumption explicitly (a bare
        # assert would be stripped under `python -O`, silently re-enabling the
        # NaN on a degenerate merge == 1 config).
        if self.merge < 2:
            raise T2NErrorConsistency(
                f"spatial_merge_size must be >= 2, got {self.merge}"
            )

    def _rot_pos_emb(self, mh, mw):
        merge = self.merge
        h, w = mh * merge, mw * merge
        dev = self.visual.pos_embed.weight.device
        block_rows = torch.arange(mh, device=dev)
        block_cols = torch.arange(mw, device=dev)
        intra = torch.arange(merge, device=dev)
        shape = (mh, mw, merge, merge)
        row_idx = (
            (
                block_rows[:, None, None, None] * merge
                + intra[None, None, :, None]
            )
            .expand(shape)
            .reshape(-1)
        )
        col_idx = (
            (
                block_cols[None, :, None, None] * merge
                + intra[None, None, None, :]
            )
            .expand(shape)
            .reshape(-1)
        )
        inv = self.visual.rotary_pos_emb.inv_freq
        row_freqs = torch.outer(torch.arange(h, device=dev).to(inv.dtype), inv)
        col_freqs = torch.outer(torch.arange(w, device=dev).to(inv.dtype), inv)
        return torch.cat(
            [
                row_freqs.index_select(0, row_idx),
                col_freqs.index_select(0, col_idx),
            ],
            dim=-1,
        )

    def _pos_embed(self, mh, mw):
        merge = self.merge
        h, w = mh * merge, mw * merge
        grid = self.num_grid
        emb = self.visual.pos_embed.weight
        dev = emb.device
        # linspace(0, grid - 1, n) as arange(n) * (grid - 1) / (n - 1); valid
        # for n >= 2, which holds since h, w = mh|mw * merge and merge >= 2
        # (asserted in __init__).
        hs = torch.arange(h, device=dev).to(emb.dtype) * ((grid - 1) / (h - 1))
        ws = torch.arange(w, device=dev).to(emb.dtype) * ((grid - 1) / (w - 1))
        hf, wf = hs.int(), ws.int()
        hc = (hf + 1).clamp(max=grid - 1)
        wc = (wf + 1).clamp(max=grid - 1)
        dh, dw = hs - hf.to(emb.dtype), ws - wf.to(emb.dtype)
        base_h, base_hc = hf * grid, hc * grid
        idx = [
            (base_h[:, None] + wf[None]).reshape(-1),
            (base_h[:, None] + wc[None]).reshape(-1),
            (base_hc[:, None] + wf[None]).reshape(-1),
            (base_hc[:, None] + wc[None]).reshape(-1),
        ]
        wgt = [
            ((1 - dh)[:, None] * (1 - dw)[None]).reshape(-1),
            ((1 - dh)[:, None] * dw[None]).reshape(-1),
            (dh[:, None] * (1 - dw)[None]).reshape(-1),
            (dh[:, None] * dw[None]).reshape(-1),
        ]
        pe = sum(
            emb.index_select(0, idx[i]) * wgt[i][:, None] for i in range(4)
        )  # [h * w, hidden] in row-major
        hidden = pe.shape[-1]
        # row-major -> merge-block-major, matching the patch order
        return (
            pe.view(mh, merge, mw, merge, hidden)
            .permute(0, 2, 1, 3, 4)
            .reshape(h * w, hidden)
        )

    def _attn(self, blk, x, cos, sin):
        attn = blk.attn
        seq = x.shape[0]
        qkv = (
            attn.qkv(x).reshape(seq, 3, attn.num_heads, -1).permute(1, 0, 2, 3)
        )
        q, k, v = qkv.unbind(0)
        cos2, sin2 = cos.unsqueeze(1), sin.unsqueeze(1)

        def rope(t_):
            half = t_.shape[-1] // 2
            rot = torch.cat((-t_[..., half:], t_[..., :half]), dim=-1)
            return t_ * cos2 + rot * sin2

        q = rope(q.float()).transpose(0, 1).unsqueeze(0)
        k = rope(k.float()).transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0).float()
        w = torch.softmax(
            torch.matmul(q, k.transpose(2, 3)) * attn.scaling, dim=-1
        )
        o = (
            torch.matmul(w, v)
            .to(x.dtype)
            .squeeze(0)
            .transpose(0, 1)
            .reshape(seq, -1)
        )
        return attn.proj(o)

    def forward(self, pixel_values: torch.Tensor):
        merge = self.merge
        mh, mw = pixel_values.shape[0], pixel_values.shape[1]
        pd = pixel_values.shape[-1]
        flat = pixel_values.reshape(mh * mw * merge * merge, pd)
        hidden = self.visual.patch_embed(flat)
        hidden = hidden + self._pos_embed(mh, mw)
        rot = self._rot_pos_emb(mh, mw)
        emb = torch.cat((rot, rot), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        deep = []
        for i, blk in enumerate(self.visual.blocks):
            hidden = hidden + self._attn(blk, blk.norm1(hidden), cos, sin)
            hidden = hidden + blk.mlp(blk.norm2(hidden))
            if i in self.deep_idx:
                merger = self.visual.deepstack_merger_list[
                    self.deep_idx.index(i)
                ]
                deep.append(merger(hidden))
        pooled = self.visual.merger(hidden)
        return (pooled, *deep)


@register_encoder_handler
class Qwen3VLVisionEncoderHandler(EncoderHandler):
    """Encoder handler for the Qwen3-VL vision tower (main + DeepStack)."""

    MODALITY = "vision"
    ARCH_NAMES = ("qwen3_vl",)
    MODEL_INPUT_NAME = "pixel_values"
    #: Sample grid (t, h, w) in patch units; h, w are multiples of merge.
    SAMPLE_GRID_THW = (1, 8, 8)

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        return Qwen3VLVisionEncoder(resolve_submodule(hf_model, "model.visual"))

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        vision_conf = config_helper.conf.vision_config
        merge = vision_conf.spatial_merge_size
        _, h, w = self.SAMPLE_GRID_THW
        mh, mw = h // merge, w // merge
        patch_dim = (
            vision_conf.in_channels
            * vision_conf.temporal_patch_size
            * vision_conf.patch_size
            * vision_conf.patch_size
        )
        n_deepstack = len(_deepstack_indexes(config_helper.conf))
        # 2D merge-block grid: flattening dims 0..3 is the processor's patch
        # order, so the host reshapes flat patches to [MH, MW, merge, merge, .].
        pixel_values = torch.randn(
            (mh, mw, merge, merge, patch_dim), dtype=inputs_dtype
        )
        output_names = ["out_image_embeddings"] + [
            deepstack_output_name("image", i) for i in range(n_deepstack)
        ]
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=output_names,
            # dynamic resolution: the two grid axes are symbolic in NNEF.
            dynamic_axes={"pixel_values": {0: "IMG_H", 1: "IMG_W"}},
        )

    def build_forward_outputs(
        self, *, model_outputs, state_context
    ) -> T.List[torch.Tensor]:
        # multi-output: main embeddings + one DeepStack stream per index.
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
                # (the input this contract feeds); the encoder emits a dynamic
                # number of tokens (its grid axes are symbolic).
                dynamic_axis="IMG_STATE",
                injection_layers=tuple(range(n_deepstack)),
                deepstack_dynamic_axis="IMG_DEEP",
            )
        ]

    def manifest_input_contract(self, config_helper):
        merge = config_helper.conf.vision_config.spatial_merge_size
        return {
            "name": "pixel_values",
            "layout": ["MH", "MW", merge, merge, "patch_dim"],
            "host_prep": (
                "Reshape the processor's merge-block-major patches to "
                f"[MH, MW, {merge}, {merge}, patch_dim] (MH = h // {merge}, "
                f"MW = w // {merge}); pure reshape, no padding."
            ),
        }
