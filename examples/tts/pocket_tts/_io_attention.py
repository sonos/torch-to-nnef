"""KV-cache-as-IO building blocks for Pocket-TTS export.

Pocket-TTS' ``StreamingMultiheadAttention`` keeps the KV cache as an
in-place-mutated state dict: ``cache[0, :, offset:offset+T_new] = k``. That
shape doesn't trace into a static NNEF graph, so for export we re-express the
same attention with the cache passed as explicit ``past_k`` / ``past_v``
inputs and the post-step cache as outputs -- the exact same pattern the LLM
example uses for stateful transformers.

The wrappers in this module are built so that *we can copy weights from the
real PyTorch Pocket-TTS modules into them and get bit-exact output*: the
inner Linear projections, RoPE, and attention math are identical to the
streaming forward, only the cache plumbing changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pocket_tts.modules.mimi_transformer import (
    StreamingTransformer,
    StreamingTransformerLayer,
)
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from torch import nn


def apply_rope_with_offset(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for an arbitrary ``offset`` scalar tensor.

    Mirrors ``pocket_tts.modules.rope.apply_rope`` but takes ``offset`` as a
    scalar int64 tensor (so tract sees it as a graph input rather than a
    constant baked at trace time). Q and K are expected as ``(B, T, H, D)``.
    """
    _, t, h, d = q.shape
    _, _, hk, _ = k.shape
    ds = torch.arange(d // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / d))
    ts = torch.arange(t, device=q.device, dtype=torch.float32) + offset.to(
        torch.float32
    )
    ts = ts.view(-1, 1, 1)
    q_pairs = q.view(-1, t, h, d // 2, 2)
    k_pairs = k.view(-1, t, hk, d // 2, 2)
    qr = q_pairs[..., 0].float()
    qi = q_pairs[..., 1].float()
    kr = k_pairs[..., 0].float()
    ki = k_pairs[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1).view(-1, t, h, d)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1).view(-1, t, hk, d)
    return qo, ko


class IOSelfAttention(nn.Module):
    """Causal self-attention with past K/V as explicit IO and updated K/V out."""

    def __init__(self, attn: StreamingMultiheadAttention):
        super().__init__()
        self.in_proj = attn.in_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.dim_per_head
        self.context = attn.context
        self.max_period = attn.rope.max_period

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, S_new, D); past_k/past_v: (B, T_past, H, Dh); offset: () int64.
        b, s_new, _ = x.shape
        qkv = self.in_proj(x).view(
            b, s_new, 3, self.num_heads, self.head_dim
        )
        q, k_new, v_new = torch.unbind(qkv, dim=2)
        q, k_new = apply_rope_with_offset(q, k_new, offset, self.max_period)
        k_all = torch.cat([past_k, k_new], dim=1)
        v_all = torch.cat([past_v, v_new], dim=1)
        # Build the same causal mask the streaming forward uses, but as a
        # function of the explicit offset. q_pos[i] = offset + i,
        # k_pos[j] = j, mask[i, j] = (q_pos[i] >= k_pos[j]) and
        # (q_pos[i] - k_pos[j] < context) when context is set.
        t_total = k_all.shape[1]
        q_pos = torch.arange(s_new, device=x.device) + offset
        k_pos = torch.arange(t_total, device=x.device)
        delta = q_pos.view(-1, 1) - k_pos.view(1, -1)
        mask = delta >= 0
        if self.context is not None:
            mask = mask & (delta < self.context)
        # SDPA wants (B, H, S, D)
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_all.transpose(1, 2),
            v_all.transpose(1, 2),
            attn_mask=mask[None, None],
            dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(
            b, s_new, self.num_heads * self.head_dim
        )
        return self.out_proj(attn_out), k_all, v_all


class IOTransformerLayer(nn.Module):
    """Wraps a Pocket-TTS ``StreamingTransformerLayer`` with KV-as-IO attention."""

    def __init__(self, layer: StreamingTransformerLayer):
        super().__init__()
        self.self_attn = IOSelfAttention(layer.self_attn)
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.layer_scale_1 = layer.layer_scale_1
        self.layer_scale_2 = layer.layer_scale_2

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self-attention residual.
        attn_out, new_k, new_v = self.self_attn(
            self.norm1(x), past_k, past_v, offset
        )
        x = x + self.layer_scale_1(attn_out)
        # FF residual.
        ff_in = self.norm2(x)
        ff_out = self.linear2(F.gelu(self.linear1(ff_in)))
        x = x + self.layer_scale_2(ff_out)
        return x, new_k, new_v


class IOTransformer(nn.Module):
    """Stack of IO transformer layers with stacked KV cache (n_layers, 2, ...)."""

    def __init__(self, transformer: StreamingTransformer):
        super().__init__()
        self.layers = nn.ModuleList(
            [IOTransformerLayer(layer) for layer in transformer.layers]
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # past_kv: (n_layers, 2, B, T_past, H, Dh). Returns (out, new_kv).
        new_layers: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            past_k_l = past_kv[i, 0]
            past_v_l = past_kv[i, 1]
            x, new_k, new_v = layer(x, past_k_l, past_v_l, offset)
            new_layers.append(torch.stack([new_k, new_v], dim=0))
        new_kv = torch.stack(new_layers, dim=0)
        return x, new_kv
