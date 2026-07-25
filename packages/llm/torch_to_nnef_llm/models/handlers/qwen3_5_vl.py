"""Handlers for the Qwen3.5 dense VLM family (e.g. Hcompany/Holo-3.1).

Model type ``qwen3_5`` (distinct from ``qwen3_5_moe``).

Two graphs, arch-keyed for the whole Qwen3.5-VL family:

* :class:`Qwen35VisionEncoderHandler` exports the vision tower (no DeepStack,
  Conv3d patch embed, dynamic resolution).
* :class:`Qwen35ArchitectureHandler` exports the hybrid decoder: a
  gated-delta-net (GDN) linear-attention layer for every ``linear_attention``
  entry in ``config.layer_types`` and a standard attention layer for every
  ``full_attention`` entry. The GDN layers carry a streaming conv state plus a
  matrix recurrent state; the full-attention layers carry a KV cache. A single
  dynamic-sequence graph serves both prefill (fresh zero states) and decode
  (carried states), because the GDN recurrence is emitted as the
  ``t2n_extra::gated_delta_scan`` op which runs for any sequence length.
"""

import typing as T

import torch
import torch.nn.functional as F

from torch_to_nnef.exceptions import T2NErrorConsistency

from .base import (
    ArchitectureHandler,
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    StateContext,
    reset_special_ids_to_filler,
    resolve_submodule,
    scatter_features_by_mask,
)
from .registry import register_encoder_handler, register_handler


def _gated_delta_scan_defined() -> bool:
    try:
        _ = torch.ops.t2n_extra.gated_delta_scan
    except (AttributeError, RuntimeError):
        return False
    return True


if not _gated_delta_scan_defined():
    # Torch-side definition of the GDN scan the decoder handler emits during
    # tracing (the T2N NNEF-emission handler lives in
    # ``torch_to_nnef.op.extras.scan_ops``). Registered idempotently so import
    # of this handler is safe even if a test already defined the op in-process.
    @torch.library.custom_op(
        "t2n_extra::gated_delta_scan",
        mutates_args=(),
        schema=(
            "(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor s0) "
            "-> (Tensor, Tensor)"
        ),
    )
    def _gated_delta_scan_op(q, k, v, g, beta, s0):
        """Pure-torch reference: the gated-delta recurrence over T (axis 2)."""
        state = s0
        outs = []
        for step in range(q.shape[2]):
            q_t, k_t, v_t = q[:, :, step], k[:, :, step], v[:, :, step]
            state = state * g[:, :, step].exp()[..., None, None]
            kv = (state * k_t.unsqueeze(-1)).sum(-2)
            delta = (v_t - kv) * beta[:, :, step][..., None]
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((state * q_t.unsqueeze(-1)).sum(-2))
        return torch.stack(outs, dim=2), state

    @_gated_delta_scan_op.register_fake
    def _gated_delta_scan_meta(q, k, v, g, beta, s0):
        batch, heads, seq, _ = q.shape
        return (
            q.new_empty((batch, heads, seq, v.shape[-1])),
            s0.new_empty(s0.shape),
        )


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Qwen35VisionEncoder(torch.nn.Module):
    """Dynamic-resolution Qwen3.5 vision tower (single image, no DeepStack).

    Input ``pixel_values`` is the processor's patch tensor reshaped to the 2D
    merge-block grid ``[MH, MW, merge, merge, patch_dim]`` (``MH = h // merge``,
    ``MW = w // merge`` both dynamic); flattening dims 0..3 restores the
    processor's merge-block-major patch order, so the reshape is host-side only.
    Both position tables (the 2D-RoPE frequencies and the interpolated learned
    ``pos_embed``) are rebuilt from ``arange(shape)`` rather than
    ``grid_thw.tolist()``, so the grid axes stay symbolic in NNEF: the tower
    exports once and runs at any resolution. The attention spans the whole
    single image, reimplemented inline so nothing depends on ``grid_thw``.

    Mirrors ``Qwen3_5VisionModel.forward`` in transformers ``modeling_qwen3_5``;
    the no-download chain-parity test compares this against HF's native forward,
    so a transformers bump that changes the tower is caught. Structurally it is
    a de-DeepStacked ``Qwen3VLVisionEncoder`` (the merge-block-major reshape
    algebra is identical).
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.merge = visual.spatial_merge_size
        self.num_grid = visual.num_grid_per_side
        # ``_pos_embed`` divides by (h - 1) with h = mh * merge; merge >= 2
        # keeps h >= 2 so the denominator is >= 1. A bare assert would be
        # stripped under ``python -O``, re-enabling the NaN on merge == 1.
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
        # for n >= 2, which holds since h, w = mh|mw * merge and merge >= 2.
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
            return t_ * cos2 + _rotate_half(t_) * sin2

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
        for blk in self.visual.blocks:
            hidden = hidden + self._attn(blk, blk.norm1(hidden), cos, sin)
            hidden = hidden + blk.mlp(blk.norm2(hidden))
        return self.visual.merger(hidden)


@register_encoder_handler
class Qwen35VisionEncoderHandler(EncoderHandler):
    """Encoder handler for the Qwen3.5 vision tower."""

    MODALITY = "vision"
    ARCH_NAMES = ("qwen3_5",)
    MODEL_INPUT_NAME = "pixel_values"
    #: Sample grid (t, h, w) in patch units; h, w are multiples of merge.
    SAMPLE_GRID_THW = (1, 8, 8)

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        return Qwen35VisionEncoder(resolve_submodule(hf_model, "model.visual"))

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
        # 2D merge-block grid: flattening dims 0..3 is the processor's patch
        # order, so the host reshapes flat patches to [MH, MW, merge, merge, .].
        pixel_values = torch.randn(
            (mh, mw, merge, merge, patch_dim), dtype=inputs_dtype
        )
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            # dynamic resolution: the two grid axes are symbolic in NNEF.
            dynamic_axes={"pixel_values": {0: "IMG_H", 1: "IMG_W"}},
        )

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        return [
            EmbeddingContract(
                modality="image",
                hidden_size=config_helper.conf.vision_config.out_hidden_size,
                placeholder_token_id_attr="image_token_id",
                # matches the decoder graph's ``in_image_embeddings`` symbol
                # (the input this contract feeds); the encoder emits a dynamic
                # number of tokens (its grid axes are symbolic).
                dynamic_axis="IMG_STATE",
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


class _HybridGDNForward:
    """Reusable gated-delta-net + full-attention compute for one decode graph.

    Reimplements ``Qwen3_5TextModel.forward`` threading EXPLICIT hybrid state
    tensors instead of the transformers ``Cache`` object: every
    ``linear_attention`` layer reads/writes ``(conv_state, rec_state)`` and
    every ``full_attention`` layer reads/writes ``(key, value)``. Prefill and
    decode share one path because the recurrence is emitted as
    ``t2n_extra::gated_delta_scan`` (runs for any T) with ``s0`` the incoming
    recurrent state, and the short conv prepends the incoming ``conv_state``
    (zeros on a fresh prefill == a left-pad). Validated against HF's native
    forward: prefill 1.3e-7, decode-step 1.2e-7.
    """

    @staticmethod
    def gdn(gdn, hidden, conv_state_in, rec_state_in):
        batch, seq, _ = hidden.shape
        conv_k = gdn.conv_kernel_size
        qkv = gdn.in_proj_qkv(hidden).transpose(1, 2)  # [B, conv_dim, T]
        z = gdn.in_proj_z(hidden).reshape(batch, seq, -1, gdn.head_v_dim)
        b = gdn.in_proj_b(hidden)
        a = gdn.in_proj_a(hidden)
        conv_dim = qkv.shape[1]
        # prepend incoming conv_state (width conv_k - 1); a padding=0 conv then
        # yields EXACTLY length T (no `[:T]` slice, which tract cannot unify
        # with the dynamic T at the downstream reshape).
        padded = torch.cat([conv_state_in, qkv], dim=-1)
        conv = F.silu(
            F.conv1d(
                padded, gdn.conv1d.weight, gdn.conv1d.bias, groups=conv_dim
            )
        )
        conv_state_out = padded[:, :, -(conv_k - 1) :]
        mixed = conv.transpose(1, 2)
        query, key, value = torch.split(
            mixed, [gdn.key_dim, gdn.key_dim, gdn.value_dim], -1
        )
        query = query.reshape(batch, seq, gdn.num_k_heads, gdn.head_k_dim)
        key = key.reshape(batch, seq, gdn.num_k_heads, gdn.head_k_dim)
        value = value.reshape(batch, seq, gdn.num_v_heads, gdn.head_v_dim)
        beta = b.sigmoid()
        g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
        rep = gdn.num_v_heads // gdn.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        scale = 1.0 / (gdn.head_k_dim**0.5)
        q_p = (_l2norm(query) * scale).transpose(1, 2)
        k_p = _l2norm(key).transpose(1, 2)
        v_p = value.transpose(1, 2)
        g_p = g.transpose(1, 2)
        beta_p = beta.transpose(1, 2)
        y, rec_state_out = torch.ops.t2n_extra.gated_delta_scan(
            q_p, k_p, v_p, g_p, beta_p, rec_state_in
        )
        core = y.transpose(1, 2).reshape(-1, gdn.head_v_dim)
        core = gdn.norm(core, z.reshape(-1, gdn.head_v_dim)).reshape(
            batch, seq, -1
        )
        return gdn.out_proj(core), conv_state_out, rec_state_out

    @staticmethod
    def attn(self_attn, hidden, cos, sin, key_in, value_in, mask):
        batch, seq, _ = hidden.shape
        head_dim = self_attn.head_dim
        query, gate = torch.chunk(
            self_attn.q_proj(hidden).view(batch, seq, -1, head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch, seq, -1)
        query = self_attn.q_norm(
            query.reshape(batch, seq, -1, head_dim)
        ).transpose(1, 2)
        key = self_attn.k_norm(
            self_attn.k_proj(hidden).view(batch, seq, -1, head_dim)
        ).transpose(1, 2)
        value = (
            self_attn.v_proj(hidden)
            .view(batch, seq, -1, head_dim)
            .transpose(1, 2)
        )
        cos2, sin2 = cos.unsqueeze(1), sin.unsqueeze(1)
        rot = cos.shape[-1]  # partial rotary: only the first `rot` head dims
        query = torch.cat(
            [
                query[..., :rot] * cos2 + _rotate_half(query[..., :rot]) * sin2,
                query[..., rot:],
            ],
            dim=-1,
        )
        key = torch.cat(
            [
                key[..., :rot] * cos2 + _rotate_half(key[..., :rot]) * sin2,
                key[..., rot:],
            ],
            dim=-1,
        )
        key = torch.cat([key_in, key], dim=2)
        value = torch.cat([value_in, value], dim=2)
        rep = self_attn.num_key_value_groups
        keys = key.repeat_interleave(rep, dim=1)
        values = value.repeat_interleave(rep, dim=1)
        weights = torch.matmul(query, keys.transpose(2, 3)) * self_attn.scaling
        weights = weights + mask
        weights = torch.softmax(weights.float(), dim=-1).to(query.dtype)
        out = (
            torch.matmul(weights, values)
            .transpose(1, 2)
            .reshape(batch, seq, -1)
        )
        out = out * torch.sigmoid(gate)
        return self_attn.o_proj(out), key, value


@register_handler
class Qwen35ArchitectureHandler(ArchitectureHandler):
    """Hybrid decoder handler for the Qwen3.5 dense VLM (``qwen3_5``).

    Owns the whole decode forward (``with_dyn_cache = False``): the
    transformers ``Cache`` object cannot represent the mixed GDN conv/recurrent
    + attention KV state through tracing, so the handler reimplements the
    forward and threads each layer's state as explicit graph inputs/outputs.
    Image embeddings from the vision encoder graph splice in at the image
    placeholder token, exactly like the other VLM decoder handlers.
    """

    ARCH_NAMES = ("qwen3_5",)
    with_dyn_cache = False
    SAMPLE_IMAGE_GRID_THW = (1, 4, 4)

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Qwen3_5ForConditionalGeneration

    def prepare_model_for_export(self, model) -> None:
        model.config._attn_implementation = "eager"
        lang = model.model.language_model
        lang.config._attn_implementation = "eager"

    # --- hybrid state layout (single source of truth) -------------------
    @staticmethod
    def _layer_types(text_conf) -> T.List[str]:
        return list(text_conf.layer_types)

    @classmethod
    def _state_names(cls, text_conf) -> T.Tuple[T.List[str], T.List[str]]:
        in_names: T.List[str] = []
        out_names: T.List[str] = []
        for idx, ltype in enumerate(cls._layer_types(text_conf)):
            if ltype == "linear_attention":
                in_names += [f"in_conv_state_{idx}", f"in_rec_state_{idx}"]
                out_names += [f"out_conv_state_{idx}", f"out_rec_state_{idx}"]
            else:
                in_names += [f"cache_key_{idx}", f"cache_value_{idx}"]
                out_names += [f"out_cache_key_{idx}", f"out_cache_value_{idx}"]
        return in_names, out_names

    @staticmethod
    def _gdn_dims(text_conf):
        n_k = text_conf.linear_num_key_heads
        n_v = text_conf.linear_num_value_heads
        h_k = text_conf.linear_key_head_dim
        h_v = text_conf.linear_value_head_dim
        conv_k = text_conf.linear_conv_kernel_dim
        conv_dim = h_k * n_k * 2 + h_v * n_v
        return n_k, n_v, h_k, h_v, conv_k, conv_dim

    def _build_state_inputs(
        self, text_conf, n_past: int, inputs_dtype: torch.dtype
    ) -> T.Tuple[T.List[torch.Tensor], T.Dict[str, T.Dict[int, str]]]:
        _, n_v, h_k, h_v, conv_k, conv_dim = self._gdn_dims(text_conf)
        head_dim = getattr(
            text_conf,
            "head_dim",
            text_conf.hidden_size // text_conf.num_attention_heads,
        )
        n_kv = text_conf.num_key_value_heads
        state_inputs: T.List[torch.Tensor] = []
        dynamic_axes: T.Dict[str, T.Dict[int, str]] = {}
        in_names, _ = self._state_names(text_conf)
        name_it = iter(in_names)
        for ltype in self._layer_types(text_conf):
            if ltype == "linear_attention":
                state_inputs.append(
                    torch.zeros((1, conv_dim, conv_k - 1), dtype=inputs_dtype)
                )
                state_inputs.append(
                    torch.zeros((1, n_v, h_k, h_v), dtype=inputs_dtype)
                )
                next(name_it)
                next(name_it)
            else:
                key_name = next(name_it)
                val_name = next(name_it)
                state_inputs.append(
                    torch.zeros((1, n_kv, n_past, head_dim), dtype=inputs_dtype)
                )
                state_inputs.append(
                    torch.zeros((1, n_kv, n_past, head_dim), dtype=inputs_dtype)
                )
                # the attention KV cache grows along the past axis at decode.
                dynamic_axes[key_name] = {2: "P"}
                dynamic_axes[val_name] = {2: "P"}
        return state_inputs, dynamic_axes

    # --- I/O spec -------------------------------------------------------
    def _num_image_tokens(self, config_helper) -> int:
        vision_conf = config_helper.conf.vision_config
        grid = torch.tensor([self.SAMPLE_IMAGE_GRID_THW], dtype=torch.long)
        return int(
            (grid.prod(-1) // (vision_conf.spatial_merge_size**2)).item()
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
        del real_kv_cache
        text_conf = config_helper.decoder_conf
        hidden_size = text_conf.hidden_size
        num_image_tokens = self._num_image_tokens(config_helper)
        # room for: vision_start + image tokens + at least one text token.
        effective_seq = max(n_input_tokens, num_image_tokens + 2)

        test_input = tokenizer(sample_text, return_tensors="pt")
        input_ids = test_input.input_ids[:, :effective_seq].clone()
        vocab_size = text_conf.vocab_size
        conf = config_helper.conf
        image_token_id = conf.image_token_id
        video_token_id = conf.video_token_id
        vision_start_token_id = getattr(
            conf, "vision_start_token_id", image_token_id - 1
        )
        input_ids.random_(0, vocab_size)
        reset_special_ids_to_filler(
            input_ids,
            {vision_start_token_id, image_token_id, video_token_id},
            vocab_size,
        )
        input_ids[:, 0] = vision_start_token_id
        for idx in range(num_image_tokens):
            if 1 + idx < effective_seq:
                input_ids[:, 1 + idx] = image_token_id

        state_inputs, state_axes = self._build_state_inputs(
            text_conf, n_past_input_tokens, inputs_dtype
        )
        in_state_names, out_state_names = self._state_names(text_conf)

        image_embeddings = torch.randn(
            (num_image_tokens, hidden_size), dtype=inputs_dtype
        )
        image_grid = torch.tensor(
            [self.SAMPLE_IMAGE_GRID_THW], dtype=torch.long
        )
        rope_deltas = torch.zeros((1, 1), dtype=torch.long)

        inputs = (
            input_ids,
            *state_inputs,
            image_embeddings,
            image_grid,
            rope_deltas,
        )
        input_names = (
            ["input_ids"]
            + in_state_names
            + ["in_image_embeddings", "in_image_grid_thw", "in_rope_deltas"]
        )
        output_names = (
            ["outputs"]
            + out_state_names
            + ["out_image_embeddings", "out_image_grid_thw", "out_rope_deltas"]
        )
        dynamic_axes = {
            "input_ids": {1: "S"},
            "in_image_embeddings": {0: "IMG_STATE"},
            "in_image_grid_thw": {0: "IMG_GRID"},
            **state_axes,
        }
        return IOSpec(
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # --- forward inputs -------------------------------------------------
    @staticmethod
    def _build_mm_token_type_ids(input_ids, *, image_token_id, video_token_id):
        mm = torch.zeros_like(input_ids, dtype=torch.int)
        mm[input_ids == image_token_id] = 1
        mm[input_ids == video_token_id] = 2
        return mm

    @staticmethod
    def _causal_mask(seq, past, dtype, device):
        total = past + seq
        k_pos = torch.arange(total, device=device).view(1, 1, total)
        q_pos = (past + torch.arange(seq, device=device)).view(1, seq, 1)
        visible = k_pos <= q_pos
        mask = torch.full(
            (1, seq, total), torch.finfo(dtype).min, dtype=dtype, device=device
        )
        return mask.masked_fill(visible, 0).unsqueeze(1)

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        hf_model = wrapper.model
        text_conf = hf_model.config.text_config
        n_layers = len(self._layer_types(text_conf))
        input_ids = inputs[0]
        state_inputs = list(inputs[1 : 1 + 2 * n_layers])
        image_embeddings, image_grid_thw, rope_deltas_state = inputs[
            1 + 2 * n_layers :
        ]

        embeds = hf_model.get_input_embeddings()(input_ids)
        image_token_id = hf_model.config.image_token_id
        video_token_id = hf_model.config.video_token_id
        embeds = scatter_features_by_mask(
            inputs_embeds=embeds,
            token_mask=input_ids == image_token_id,
            features=image_embeddings,
        )

        # past length is the attention KV cache depth; find the first
        # full_attention layer's key input. Pure-GDN configs have no KV -> 0.
        past_len = 0
        for idx, ltype in enumerate(self._layer_types(text_conf)):
            if ltype == "full_attention":
                past_len = state_inputs[2 * idx].shape[2]
                break

        mm_token_type_ids = self._build_mm_token_type_ids(
            input_ids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        rope_attn_mask = torch.ones(
            (input_ids.shape[0], past_len + input_ids.shape[1]),
            dtype=torch.long,
            device=input_ids.device,
        )
        image_grid_arg = image_grid_thw if image_grid_thw.numel() else None
        if past_len == 0 or rope_deltas_state.numel() == 0:
            position_ids, rope_deltas_current = hf_model.model.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_arg,
                video_grid_thw=None,
                attention_mask=rope_attn_mask,
            )
            position_ids = position_ids.to(device=input_ids.device)
            rope_deltas_current = rope_deltas_current.to(
                device=input_ids.device, dtype=torch.long
            )
        else:
            rope_deltas_current = rope_deltas_state.to(
                device=input_ids.device, dtype=torch.long
            )
            base = (
                torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                )
                .view(1, 1, -1)
                .repeat(3, input_ids.shape[0], 1)
            )
            delta = (past_len + rope_deltas_current).view(
                1, input_ids.shape[0], 1
            )
            position_ids = base + delta

        mask = self._causal_mask(
            input_ids.shape[1], past_len, embeds.dtype, input_ids.device
        )
        return StateContext(
            model_inputs={},
            state={
                "embeds": embeds,
                "position_ids": position_ids,
                "mask": mask,
                "state_inputs": state_inputs,
                "image_embeddings": image_embeddings,
                "image_grid_thw": image_grid_thw,
                "rope_deltas": rope_deltas_current,
            },
        )

    # --- run the hybrid forward ----------------------------------------
    def call_model(self, *, model, state_context, wrapper) -> T.Any:
        del wrapper
        st = state_context.state
        lang = model.model.language_model
        text_conf = model.config.text_config
        hidden = st["embeds"]
        cos, sin = lang.rotary_emb(hidden, st["position_ids"])
        state_in = list(st["state_inputs"])
        new_states: T.List[torch.Tensor] = []
        cursor = 0
        for idx, layer in enumerate(lang.layers):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            if self._layer_types(text_conf)[idx] == "linear_attention":
                conv_in = state_in[cursor]
                rec_in = state_in[cursor + 1]
                mix, conv_out, rec_out = _HybridGDNForward.gdn(
                    layer.linear_attn, normed, conv_in, rec_in
                )
                new_states += [conv_out, rec_out]
            else:
                key_in = state_in[cursor]
                value_in = state_in[cursor + 1]
                mix, key_out, value_out = _HybridGDNForward.attn(
                    layer.self_attn,
                    normed,
                    cos,
                    sin,
                    key_in,
                    value_in,
                    st["mask"],
                )
                new_states += [key_out, value_out]
            cursor += 2
            hidden = residual + mix
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
        hidden = lang.norm(hidden)
        logits = model.lm_head(hidden)
        return {"logits": logits, "new_states": new_states}

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs: T.Any,
        state_context,
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        del model
        logits = model_outputs["logits"]
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:]
        state = state_context.state
        return (
            [logits]
            + list(model_outputs["new_states"])
            + [
                state["image_embeddings"],
                state["image_grid_thw"],
                state["rope_deltas"],
            ]
        )
