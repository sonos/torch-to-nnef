"""Export Kyutai's Pocket-TTS Mimi decoder via torch-to-nnef.

Target: ``kyutai/pocket-tts`` (https://github.com/kyutai-labs/pocket-tts).

The shipped Pocket-TTS architecture is a ``FlowLM`` (text + voice prompt ->
continuous latents) followed by a ``Mimi`` neural codec decoder
(latents -> 24 kHz waveform). At inference the decoder is the always-on
hot path (one call per audio chunk), and it is the natural first slice to
get running on tract.

The catch: every Conv1d / ConvTranspose1d in Mimi is a ``StreamingConv1d`` /
``StreamingConvTranspose1d`` whose forward expects a ``model_state`` dict and
mutates it **in place** with running KV-cache-style buffers. Tracing through
those mutations does not produce a clean static graph. So the export does
not target the streaming path directly -- instead it builds a stateless
mirror that replaces each streaming leaf with a static-graph equivalent
(``F.pad`` for the left context, end-slice for the right partial), reuses
the trained weights as-is, and produces bit-exact output for the bulk
non-streaming case.

Run:
    # Mini, fast (random weights, ~50k params, no HF download):
    python decoder.py --mini --skip-io-check
    # Mini with check_io against tract:
    python decoder.py --mini

The ``--mini`` flag mirrors Pocket-TTS' real Mimi config at a tiny scale so
this script can run anywhere without authenticating against the gated
checkpoint. Producing audio from a *real* Pocket-TTS checkpoint would also
require exporting the FlowLM front-end, which is out of scope for this
example (the FlowLM is autoregressive and stateful in a way that would
need a bigger adapter; see TODO in the README).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from pocket_tts.modules.conv import (
    StreamingConv1d,
    StreamingConvTranspose1d,
)
from pocket_tts.modules.seanet import SEANetDecoder
from pocket_tts.modules.stateful_module import StatefulModule
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef


class StatelessConv1d(nn.Module):
    """Static-graph mirror of ``StreamingConv1d`` for bulk decoding.

    The streaming forward prepends an in-place ``previous`` buffer of length
    ``effective_kernel - stride`` to the input before running the underlying
    ``nn.Conv1d`` with ``padding=0``. For a fresh streaming session that
    buffer is all zeros, which is the same as left-padding the full input
    with zeros -- so an ``F.pad`` followed by the same ``Conv1d`` produces
    identical output and traces cleanly.
    """

    def __init__(self, streaming: StreamingConv1d):
        super().__init__()
        self.conv = streaming.conv
        self.left_pad = streaming._effective_kernel_size - streaming._stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class StatelessConvTranspose1d(nn.Module):
    """Static-graph mirror of ``StreamingConvTranspose1d`` for bulk decoding.

    The streaming forward stores the trailing ``kernel - stride`` samples of
    each call and adds them to the head of the next call. For a fresh
    streaming session the stored partial is all zeros and we emit only the
    samples that wouldn't be overwritten by the next chunk -- equivalent to
    truncating the convtranspose output by ``kernel - stride`` on the right.
    """

    def __init__(self, streaming: StreamingConvTranspose1d):
        super().__init__()
        self.convtr = streaming.convtr
        self.tail = streaming._kernel_size - streaming._stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        if self.tail > 0:
            y = y[..., : -self.tail]
        return y


def replace_streaming_with_stateless(module: nn.Module) -> None:
    """Recursively swap streaming conv leaves with their stateless mirrors."""
    for name, child in list(module.named_children()):
        if isinstance(child, StreamingConv1d):
            setattr(module, name, StatelessConv1d(child))
        elif isinstance(child, StreamingConvTranspose1d):
            setattr(module, name, StatelessConvTranspose1d(child))
        else:
            replace_streaming_with_stateless(child)


class StatelessSEANetDecoder(nn.Module):
    """Wrap a (now-patched) ``SEANetDecoder`` into a single-arg forward.

    Once the streaming leaves are replaced, the streaming-class isinstance
    checks inside ``SEANetDecoder.forward`` / ``SEANetResnetBlock.forward``
    no longer match, so the decoder falls through to the plain ``layer(x)``
    branch. The ``model_state={}`` is just an unused placeholder kept around
    by ``SEANetResnetBlock``'s call signature.
    """

    def __init__(self, dec: SEANetDecoder):
        super().__init__()
        self.dec = dec

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.dec(latent, model_state={})


def build_mini_decoder() -> SEANetDecoder:
    """Tiny SEANet decoder mirroring Pocket-TTS' Mimi config at small scale.

    Real Mimi: ``dimension=512``, ``n_filters=64``, ``ratios=[8, 6, 5, 4]``,
    ``n_residual_layers=1`` (24 kHz audio, hop length 960). Here we shrink
    every dim to keep the structure (initial conv, alternating ELU /
    transposed-conv / residual block, final conv) while staying ~50k params.
    """
    # ``dimension`` is the channel count entering the SEANet decoder. In real
    # Mimi a ``decoder_transformer`` projects FlowLM's ``ldim`` latents up to
    # this dim; we don't export that transformer in the mini path, so set
    # ``dimension == ldim`` (=8 in ``flow_lm.py:build_mini_flow_lm``) so the
    # autoregressive latent stream feeds straight into the decoder.
    dec = SEANetDecoder(
        channels=1,
        dimension=8,
        n_filters=8,
        n_residual_layers=1,
        ratios=[4, 5, 8],
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        pad_mode="constant",
        compress=2,
    ).eval()
    # ``StatefulModule.get_state`` looks up state by ``_module_absolute_name``;
    # set it here so a downstream sanity check against the streaming forward
    # works (the stateless export does not read the state).
    for name, mod in dec.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = name
    return dec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./pocket_tts_decoder.nnef.tgz")
    )
    parser.add_argument(
        "--latent-frames",
        type=int,
        default=8,
        help="Number of latent frames to decode (real Mimi runs at 12.5 Hz).",
    )
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (only validates the graph emit).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        default=True,
        help="Tiny random-weights config (default; only mode supported "
        "today). Real Mimi decoder export would need extra adapter work to "
        "load Pocket-TTS' safetensors and rebuild the matching structure.",
    )
    args = parser.parse_args()

    streaming = build_mini_decoder()
    stateless = copy.deepcopy(streaming)
    replace_streaming_with_stateless(stateless)
    model = StatelessSEANetDecoder(stateless).eval()

    latent_dim = streaming.dimension
    latent = torch.randn(1, latent_dim, args.latent_frames, dtype=torch.float32)
    upsample = 1
    for r in streaming.ratios:
        upsample *= r
    print(
        f"latent_dim={latent_dim} frames={args.latent_frames} "
        f"upsample={upsample} -> audio_samples={args.latent_frames * upsample}"
    )

    with torch.no_grad():
        out = model(latent)
    print(f"PyTorch output shape: {tuple(out.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=model,
        args=(latent,),
        file_path_export=args.out,
        inference_target=TractNNEF(version=tract_version, check_io=check_io),
        input_names=["latent"],
        output_names=["audio"],
        debug_bundle_path=Path("./debug_pocket_tts_decoder.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
