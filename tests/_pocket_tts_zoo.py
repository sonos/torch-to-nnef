"""Helper for the ``mini_pocket_tts_decoder`` zoo entry.

Lives outside ``test_model_zoo.py`` so the streaming-conv adapter classes
don't bloat that file's cyclomatic complexity. See
``examples/tts/pocket_tts/decoder.py`` for the production-shaped equivalent
and a longer rationale.
"""

import copy

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.seanet import SEANetDecoder
from pocket_tts.modules.stateful_module import StatefulModule
from torch import nn


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
