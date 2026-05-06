"""Helpers for the ``mini_pocket_tts_*`` zoo entries.

Lives outside ``test_model_zoo.py`` so the streaming-conv / KV-cache adapter
classes don't bloat that file's cyclomatic complexity. See the matching
``examples/tts/pocket_tts/{decoder,flow_lm}.py`` exporters for the
production-shaped equivalents and longer rationales.
"""

import copy

import torch
from pocket_tts.conditioners import text as conditioners_text
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.seanet import SEANetDecoder
from pocket_tts.modules.stateful_module import StatefulModule
from torch import nn

from examples.tts.pocket_tts._flow_lm_export import FlowLMInit, FlowLMStep


class _StatelessConv1d(nn.Module):
    def __init__(self, streaming):
        super().__init__()
        self.conv = streaming.conv
        self.left_pad = streaming._effective_kernel_size - streaming._stride

    def forward(self, x):
        if self.left_pad > 0:
            x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class _StatelessConvTranspose1d(nn.Module):
    def __init__(self, streaming):
        super().__init__()
        self.convtr = streaming.convtr
        self.tail = streaming._kernel_size - streaming._stride

    def forward(self, x):
        y = self.convtr(x)
        return y[..., : -self.tail] if self.tail > 0 else y


def _patch_streaming(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, StreamingConv1d):
            setattr(module, name, _StatelessConv1d(child))
        elif isinstance(child, StreamingConvTranspose1d):
            setattr(module, name, _StatelessConvTranspose1d(child))
        else:
            _patch_streaming(child)


class MiniPocketTTSDecoder(nn.Module):
    """Tiny SEANet decoder mirroring Pocket-TTS' Mimi config at small scale."""

    def __init__(self):
        super().__init__()
        streaming = SEANetDecoder(
            channels=1,
            dimension=64,
            n_filters=8,
            n_residual_layers=1,
            ratios=[4, 5, 8],
            pad_mode="constant",
        ).eval()
        for name, mod in streaming.named_modules():
            if isinstance(mod, StatefulModule):
                mod._module_absolute_name = name
        stateless = copy.deepcopy(streaming)
        _patch_streaming(stateless)
        self.dec = stateless

    def forward(self, latent):
        return self.dec(latent, model_state={})


def _build_lut_conditioner_without_tokenizer(
    n_bins: int, dim: int
) -> LUTConditioner:
    """Build a ``LUTConditioner`` whose ``__init__`` skips SentencePiece."""

    class _StubTokenizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def vocab_size(self):
            return n_bins

    real = conditioners_text.SentencePieceTokenizer
    conditioners_text.SentencePieceTokenizer = _StubTokenizer
    try:
        cond = LUTConditioner(
            n_bins=n_bins, tokenizer_path="", dim=dim, output_dim=dim
        )
    finally:
        conditioners_text.SentencePieceTokenizer = real
    cond.tokenizer = None
    return cond


def _build_mini_flow_lm() -> FlowLMModel:
    n_bins, d_model = 100, 16
    num_layers, num_heads = 2, 2
    ldim = 8
    context = 32
    conditioner = _build_lut_conditioner_without_tokenizer(
        n_bins=n_bins, dim=d_model
    )
    transformer = StreamingTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=32,
        context=context,
    )
    flow_net = SimpleMLPAdaLN(
        in_channels=ldim,
        model_channels=16,
        out_channels=ldim,
        cond_channels=d_model,
        num_res_blocks=2,
        num_time_conds=2,
    )
    flow_lm = FlowLMModel(
        conditioner=conditioner,
        flow_net=flow_net,
        transformer=transformer,
        dim=d_model,
        ldim=ldim,
        insert_bos_before_voice=False,
    ).eval()
    for name, mod in flow_lm.transformer.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = name
    return flow_lm


class MiniPocketTTSFlowLMInit(nn.Module):
    """First FlowLM call: tokens + voice KV → transformer_out + eos + KV."""

    def __init__(self):
        super().__init__()
        self.wrapped = FlowLMInit(_build_mini_flow_lm())

    def forward(self, token_ids, past_kv, q_positions, k_positions):
        return self.wrapped(token_ids, past_kv, q_positions, k_positions)


class MiniPocketTTSFlowLMStep(nn.Module):
    """Subsequent FlowLM call: one audio latent + KV → out + eos + KV."""

    def __init__(self):
        super().__init__()
        self.wrapped = FlowLMStep(_build_mini_flow_lm())

    def forward(self, audio_latent, past_kv, q_positions, k_positions):
        return self.wrapped(audio_latent, past_kv, q_positions, k_positions)


def mini_flow_lm_init_inputs():
    return (
        torch.randint(0, 100, (1, 4), dtype=torch.long),
        torch.zeros(2, 2, 1, 0, 2, 8),
        torch.arange(5, dtype=torch.long),
        torch.arange(5, dtype=torch.long),
    )


def mini_flow_lm_step_inputs():
    return (
        torch.randn(1, 8),
        torch.randn(2, 2, 1, 4, 2, 8),
        torch.tensor([4], dtype=torch.long),
        torch.arange(5, dtype=torch.long),
    )
