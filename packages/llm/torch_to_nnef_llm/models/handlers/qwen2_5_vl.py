"""Joint-export handlers for Qwen2.5-VL vision-language models.

- :class:`Qwen25VLVisionEncoderHandler`: the vision tower (conv3d patch embed,
  2D-RoPE, window + full attention, patch merger), exported at dynamic
  resolution. The window partition is encoded in the input shape
  (``[NH, W, NW, W, merge, merge, patch_dim]``, window counts NH/NW dynamic),
  so the data-dependent ``window_index`` / ``cu_seqlens`` reduce to structural
  reshapes and every position table is rebuilt from ``arange(shape)`` -- no
  baked ``grid_thw``. Host contract: feed a whole number of merger-windows
  (zero-pad, discard padded output tokens).
- :class:`Qwen25VLArchitectureHandler`: the decoder graph. Reuses the landed
  Qwen3-VL decoder handler (image-embedding injection + mRoPE + rope_deltas);
  only the loaded model class differs (no DeepStack in Qwen2.5-VL).

Both handlers are validated in PyTorch (encoder output bit-exact vs
``get_image_features``, decoder wrapper self-consistent) and export to tract in
``f32``. Only the ``f16`` vision path is currently gated: not by t2n, but by a
tract ``-O`` optimizer bug on the ``einsum(acc=f32)`` accumulation pattern,
fixed in tract main.
"""

import typing as T

import torch

from .base import (
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    resolve_submodule,
)
from .qwen3_vl import Qwen3VLArchitectureHandler
from .registry import register_encoder_handler, register_handler


class Qwen25VLVisionEncoder(torch.nn.Module):
    """Dynamic-resolution Qwen2.5-VL vision tower (single frame, windowed).

    Input ``pixel_values`` is the window-structured grid
    ``[NH, W, NW, W, merge, merge, patch_dim]`` where ``W`` is the merger-window
    size (``window_size // merge // patch_size``, static) and ``NH``/``NW`` are
    the window counts (dynamic). Encoding the window partition in the shape
    keeps every reshape a product of ``{NH, NW, W, merge}`` with the dynamic
    axes appearing linearly, so no ``symbolic // W`` folding is ever needed (the
    reason ``grid_thw`` was previously baked). The host feeds
    ``pixel_values.reshape(NH, W, NW, W, merge, merge, .)`` from the processor's
    merge-block-major patches, zero-padded up to a whole number of windows;
    padded output tokens are discarded host-side.

    Windowed attention becomes batched full attention within each window;
    ``fullatt_block_indexes`` blocks attend globally. Positions (2D-RoPE) are
    rebuilt from ``arange(shape)``; the final merger output is un-permuted back
    to the processor's row-major merge-block order.

    Mirrors ``Qwen2_5_VisionTransformerPretrainedModel.forward`` in transformers
    ``modeling_qwen2_5_vl``; the no-download chain-parity test
    (``test_dummy_chain_parity``) compares this against HF's native forward, so
    a transformers bump that changes the tower is caught.
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.merge = visual.spatial_merge_size
        self.win = (
            visual.window_size // visual.spatial_merge_size // visual.patch_size
        )
        self.fullatt = set(visual.fullatt_block_indexes)

    def _rot_pos_emb(self, nh, nw):
        win, merge = self.win, self.merge
        dev = self.visual.rotary_pos_emb.inv_freq.device
        h, w = nh * win * merge, nw * win * merge
        nh_r = torch.arange(nh, device=dev)[:, None, None, None, None, None]
        a_r = torch.arange(win, device=dev)[None, :, None, None, None, None]
        nw_r = torch.arange(nw, device=dev)[None, None, :, None, None, None]
        b_r = torch.arange(win, device=dev)[None, None, None, :, None, None]
        pr = torch.arange(merge, device=dev)[None, None, None, None, :, None]
        pc = torch.arange(merge, device=dev)[None, None, None, None, None, :]
        shape = (nh, win, nw, win, merge, merge)
        row = ((nh_r * win + a_r) * merge + pr).expand(shape).reshape(-1)
        col = ((nw_r * win + b_r) * merge + pc).expand(shape).reshape(-1)
        inv = self.visual.rotary_pos_emb.inv_freq
        row_freqs = torch.outer(torch.arange(h, device=dev).to(inv.dtype), inv)
        col_freqs = torch.outer(torch.arange(w, device=dev).to(inv.dtype), inv)
        return torch.cat(
            [row_freqs.index_select(0, row), col_freqs.index_select(0, col)],
            dim=-1,
        )

    def _to_window_major(self, x, nh, nw, last):
        # [seq, *last] row-major [NH,W,NW,W,munit] -> window-major windows
        win, munit = self.win, self.merge * self.merge
        x = x.reshape(nh, win, nw, win, munit, *last)
        perm = (0, 2, 1, 3, 4) + tuple(5 + i for i in range(len(last)))
        return x.permute(*perm).reshape(nh * nw * win * win * munit, *last)

    def _attn(self, blk, x, cos, sin, windowed, nh, nw):
        attn = blk.attn
        qkv = attn.qkv(x).reshape(-1, 3, attn.num_heads, attn.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        cos2, sin2 = cos.unsqueeze(1), sin.unsqueeze(1)

        def rope(t_):
            t_ = t_.float()
            half = t_.shape[-1] // 2
            rot = torch.cat((-t_[..., half:], t_[..., :half]), dim=-1)
            return t_ * cos2 + rot * sin2

        q, k, v = rope(q), rope(k), v.float()
        heads, hd = attn.num_heads, attn.head_dim
        if windowed:
            wt = self.win * self.win * self.merge * self.merge
            q = q.reshape(nh * nw, wt, heads, hd).permute(0, 2, 1, 3)
            k = k.reshape(nh * nw, wt, heads, hd).permute(0, 2, 1, 3)
            v = v.reshape(nh * nw, wt, heads, hd).permute(0, 2, 1, 3)
            o = self._sdpa(q, k, v, attn.scaling)
            o = o.permute(0, 2, 1, 3).reshape(-1, heads * hd)
        else:
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            o = self._sdpa(q, k, v, attn.scaling)
            o = o.squeeze(0).transpose(0, 1).reshape(-1, heads * hd)
        return attn.proj(o.to(x.dtype))

    @staticmethod
    def _sdpa(q, k, v, scaling):
        w = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * scaling, dim=-1
        )
        return torch.matmul(w, v)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        win, merge = self.win, self.merge
        munit = merge * merge
        nh, nw = pixel_values.shape[0], pixel_values.shape[2]
        pd = pixel_values.shape[-1]
        flat = pixel_values.reshape(nh * win * nw * win * munit, pd)
        hidden = self.visual.patch_embed(flat)
        c = hidden.shape[-1]
        rot = self._rot_pos_emb(nh, nw)
        hidden = self._to_window_major(hidden, nh, nw, (c,))
        rot = self._to_window_major(rot, nh, nw, (rot.shape[-1],))
        emb = torch.cat((rot, rot), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        for i, blk in enumerate(self.visual.blocks):
            windowed = i not in self.fullatt
            hidden = hidden + self._attn(
                blk, blk.norm1(hidden), cos, sin, windowed, nh, nw
            )
            hidden = hidden + blk.mlp(blk.norm2(hidden))
        pooled = self.visual.merger(hidden)  # window-major merger-blocks
        out = pooled.shape[-1]
        return (
            pooled.reshape(nh, nw, win, win, out)
            .permute(0, 2, 1, 3, 4)
            .reshape(nh * win * nw * win, out)
        )


@register_encoder_handler
class Qwen25VLVisionEncoderHandler(EncoderHandler):
    """Encoder handler for the Qwen2.5-VL vision tower."""

    MODALITY = "vision"
    ARCH_NAMES = ("qwen2_5_vl",)
    MODEL_INPUT_NAME = "pixel_values"
    #: Sample size as a (NH, NW) window count; the actual grid is derived so it
    #: is always a whole number of merger-windows (the encoder's host contract).
    SAMPLE_WINDOWS = (2, 2)

    @staticmethod
    def merger_window(vision_conf) -> int:
        return (
            vision_conf.window_size
            // vision_conf.spatial_merge_size
            // vision_conf.patch_size
        )

    def get_encoder_module(self, hf_model) -> torch.nn.Module:
        return Qwen25VLVisionEncoder(
            resolve_submodule(hf_model, "model.visual")
        )

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        vision_conf = config_helper.conf.vision_config
        merge = vision_conf.spatial_merge_size
        win = self.merger_window(vision_conf)
        nh, nw = self.SAMPLE_WINDOWS
        patch_dim = (
            vision_conf.in_channels
            * vision_conf.temporal_patch_size
            * vision_conf.patch_size
            * vision_conf.patch_size
        )
        # window-structured grid: flattening dims 0..5 is the processor's patch
        # order; NH/NW (window counts) are the dynamic axes.
        pixel_values = torch.randn(
            (nh, win, nw, win, merge, merge, patch_dim), dtype=inputs_dtype
        )
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            # dynamic resolution: the window-count axes are symbolic in NNEF.
            dynamic_axes={"pixel_values": {0: "WIN_H", 2: "WIN_W"}},
        )

    def contracts(self, config_helper) -> T.List[EmbeddingContract]:
        return [
            EmbeddingContract(
                modality="image",
                hidden_size=config_helper.conf.vision_config.out_hidden_size,
                placeholder_token_id_attr="image_token_id",
                # matches the (shared Qwen3-VL) decoder graph's
                # ``in_image_embeddings`` symbol; the encoder emits a dynamic
                # number of tokens (its window-count axes are symbolic).
                dynamic_axis="IMG_STATE",
            )
        ]

    def manifest_input_contract(self, config_helper):
        vc = config_helper.conf.vision_config
        merge = vc.spatial_merge_size
        win = self.merger_window(vc)
        return {
            "name": "pixel_values",
            "layout": ["WIN_H", win, "WIN_W", win, merge, merge, "patch_dim"],
            "window_size": win,
            "requires_window_multiple": True,
            "host_prep": (
                "Reshape the processor's merge-block-major patches to "
                f"[NH, {win}, NW, {win}, {merge}, {merge}, patch_dim]. The LLM "
                f"grid (h // {merge}, w // {merge}) MUST first be zero-padded "
                f"to a whole number of {win}-wide merger-windows; output "
                "tokens beyond the true (unpadded) count must be discarded. "
                "Feeding a non-window-aligned grid produces wrong results."
            ),
        }


@register_handler
class Qwen25VLArchitectureHandler(Qwen3VLArchitectureHandler):
    """Decoder handler for Qwen2.5-VL (Qwen3-VL logic minus DeepStack)."""

    ARCH_NAMES = ("qwen2_5_vl",)

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Qwen2_5_VLForConditionalGeneration
