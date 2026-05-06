"""Exportable FlowLM init / step wrappers.

Two NNEF graphs come out of this module:

* ``FlowLMInit`` -- one-shot per utterance: token IDs + voice KV prefix in,
  transformer hidden state at the BOS audio position + EOS logit + populated
  KV cache out. The text conditioner (``LUTConditioner.embed``) is folded into
  the front of this graph so the Rust runtime only has to do SentencePiece
  tokenization.
* ``FlowLMStep`` -- one autoregressive step: one new audio latent + current
  KV cache in, transformer hidden state + EOS logit + updated KV cache out.

Both wrappers reuse the trained weights of an upstream ``FlowLMModel``
(``input_linear``, ``transformer.layers``, ``out_norm``, ``out_eos``,
``bos_emb``) without any rewriting, so the only difference vs. the streaming
reference is *how* the KV cache is plumbed -- in-place on the streaming side,
explicit IO here.
"""

from __future__ import annotations

import torch
from pocket_tts.models.flow_lm import FlowLMModel
from torch import nn

from examples.tts.pocket_tts._io_attention import IOTransformer


class FlowLMInit(nn.Module):
    """First FlowLM call: token IDs + voice KV → transformer_out + eos + KV."""

    def __init__(self, flow_lm: FlowLMModel):
        super().__init__()
        self.embed = flow_lm.conditioner.embed
        self.input_linear = flow_lm.input_linear
        self.io_transformer = IOTransformer(flow_lm.transformer)
        self.out_norm = flow_lm.out_norm
        self.out_eos = flow_lm.out_eos
        self.register_buffer("bos_emb", flow_lm.bos_emb.detach().clone())

    def forward(
        self,
        token_ids: torch.Tensor,
        past_kv: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # token_ids:   (B, T_text) int64
        # past_kv:     (n_layers, 2, B, T_past, H, D) float32 (= voice prefix)
        # q_positions: (T_text + 1,) int64 = caller-supplied absolute positions
        #              for the joint [text..., BOS] query stream
        # k_positions: (T_past + T_text + 1,) int64 = positions for the full
        #              K cache after this step (used for the causal mask)
        text_emb = self.embed(token_ids)  # (B, T_text, dim)
        b = text_emb.shape[0]
        bos = self.bos_emb.view(1, 1, -1).expand(b, 1, -1)  # (B, 1, ldim)
        bos_proj = self.input_linear(bos)  # (B, 1, dim)
        seq_in = torch.cat([text_emb, bos_proj], dim=1)
        out, new_kv = self.io_transformer(
            seq_in, past_kv, q_positions, k_positions
        )
        out = self.out_norm(out)
        # Hidden state at the BOS audio position (last in the joint sequence).
        out_last = out[:, -1, :]
        eos_logit = self.out_eos(out_last)  # (B, 1)
        return out_last, eos_logit, new_kv


class FlowLMStep(nn.Module):
    """Subsequent FlowLM call: one audio latent + KV → out + eos + KV."""

    def __init__(self, flow_lm: FlowLMModel):
        super().__init__()
        self.input_linear = flow_lm.input_linear
        self.io_transformer = IOTransformer(flow_lm.transformer)
        self.out_norm = flow_lm.out_norm
        self.out_eos = flow_lm.out_eos

    def forward(
        self,
        audio_latent: torch.Tensor,
        past_kv: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # audio_latent: (B, ldim)
        # past_kv:      (n_layers, 2, B, T_past, H, D)
        # q_positions:  (1,) int64    -- absolute position of this audio step
        # k_positions:  (T_past + 1,) -- positions for the full K cache
        x = audio_latent.unsqueeze(1)  # (B, 1, ldim)
        x = self.input_linear(x)  # (B, 1, dim)
        out, new_kv = self.io_transformer(x, past_kv, q_positions, k_positions)
        out = self.out_norm(out)
        out_last = out[:, -1, :]
        eos_logit = self.out_eos(out_last)
        return out_last, eos_logit, new_kv
