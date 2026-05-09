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
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from torch import nn


def make_rope_freqs(head_dim: int, max_period: float) -> torch.Tensor:
    """Precomputed frequency band, same as ``pocket_tts.modules.rope``."""
    ds = torch.arange(head_dim // 2, dtype=torch.float32)
    return torch.exp(ds * (-math.log(max_period) * 2 / head_dim))


def apply_rope_at_positions(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE rotation using absolute ``positions`` supplied by the caller.

    Mirrors ``pocket_tts.modules.rope.apply_rope`` semantically but takes:
    - ``positions`` -- ``(T,) int64`` absolute time indices for each query and
      key step (the Rust runtime computes
      ``[offset, offset+1, ..., offset+T-1]`` and feeds it as a graph input);
    - ``freqs`` -- the precomputed ``(D//2,)`` frequency band (a module buffer),

    which keeps every shape-dependent ``arange`` out of the forward pass.
    """
    _, t, h, d = q.shape
    _, _, hk, _ = k.shape
    ts = positions.to(torch.float32).view(-1, 1, 1)
    q_pairs = q.view(-1, t, h, d // 2, 2)
    k_pairs = k.view(-1, t, hk, d // 2, 2)
    qr = q_pairs[..., 0].float()
    qi = q_pairs[..., 1].float()
    kr = k_pairs[..., 0].float()
    ki = k_pairs[..., 1].float()
    # Compute the angle once and reuse for cos/sin -- tract / t2n
    # otherwise emit the broadcast-align unsqueeze under the same NNEF
    # tensor name twice (two separate ``aligned`` definitions) which
    # trips a "Clashing resolution" check at runtime.
    angle = freqs * ts  # (T, 1, D//2)
    # Pre-align ``rotr`` / ``roti`` to ``q_pairs[..., 0]`` rank
    # (B, T, H, D//2) once each, so that the four multiplies below need no
    # further rank-broadcast and t2n doesn't emit duplicate aligned names.
    rotr = torch.cos(angle).unsqueeze(0)  # (1, T, 1, D//2)
    roti = torch.sin(angle).unsqueeze(0)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1).view(-1, t, h, d)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1).view(-1, t, hk, d)
    return qo, ko


class IOSelfAttention(nn.Module):
    """Causal self-attention with past K/V as explicit IO and updated K/V out.

    Position indices for RoPE and the causal mask are caller-supplied
    (``q_positions`` for the new tokens, ``k_positions`` for the full K/V
    slice including past) so the graph contains no shape-dependent
    ``arange`` -- tract's ``Range`` op chokes on TDim/F32 mixing in a
    static export.
    """

    def __init__(self, attn: StreamingMultiheadAttention):
        super().__init__()
        self.in_proj = attn.in_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.dim_per_head
        self.context = attn.context
        self.register_buffer(
            "rope_freqs",
            make_rope_freqs(self.head_dim, attn.rope.max_period),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s_new, _ = x.shape
        qkv = self.in_proj(x).view(b, s_new, 3, self.num_heads, self.head_dim)
        q, k_new, v_new = torch.unbind(qkv, dim=2)
        q, k_new = apply_rope_at_positions(
            q, k_new, q_positions, self.rope_freqs
        )
        k_all = torch.cat([past_k, k_new], dim=1)
        v_all = torch.cat([past_v, v_new], dim=1)
        # Causal mask: q_pos[i] >= k_pos[j], plus a sliding-window cap of
        # ``context`` when present. Tract's SDPA op treats ``attn_mask`` as an
        # additive float bias on the scores, so we materialise the mask as
        # ``0.0`` (keep) / ``-inf`` (drop) instead of bool to match.
        delta = q_positions.view(-1, 1) - k_positions.view(1, -1)
        keep = delta >= 0
        if self.context is not None:
            keep = keep & (delta < self.context)
        attn_mask = torch.where(
            keep,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), float("-inf"), dtype=q.dtype, device=q.device),
        )
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_all.transpose(1, 2),
            v_all.transpose(1, 2),
            attn_mask=attn_mask[None, None],
            dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(
            b, s_new, self.num_heads * self.head_dim
        )
        return self.out_proj(attn_out), k_all, v_all


class IOTransformerLayer(nn.Module):
    """Pocket-TTS ``StreamingTransformerLayer`` with KV-as-IO attention."""

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
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self-attention residual.
        attn_out, new_k, new_v = self.self_attn(
            self.norm1(x), past_k, past_v, q_positions, k_positions
        )
        x = x + self.layer_scale_1(attn_out)
        # FF residual.
        ff_in = self.norm2(x)
        ff_out = self.linear2(F.gelu(self.linear1(ff_in)))
        x = x + self.layer_scale_2(ff_out)
        return x, new_k, new_v


class IOTransformer(nn.Module):
    """Stack of IO transformer layers; KV cache shape is (n_layers, 2, ...)."""

    def __init__(self, transformer: StreamingTransformer):
        super().__init__()
        self.layers = nn.ModuleList(
            [IOTransformerLayer(layer) for layer in transformer.layers]
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # past_kv: (n_layers, 2, B, T_past, H, Dh). Returns (out, new_kv).
        new_layers: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            past_k_l = past_kv[i, 0]
            past_v_l = past_kv[i, 1]
            x, new_k, new_v = layer(
                x, past_k_l, past_v_l, q_positions, k_positions
            )
            new_layers.append(torch.stack([new_k, new_v], dim=0))
        new_kv = torch.stack(new_layers, dim=0)
        return x, new_kv
