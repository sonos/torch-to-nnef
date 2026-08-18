"""Export Pocket-TTS' Mimi audio-decode chain as a single NNEF graph.

The autoregressive front-end (FlowLM + flow_net) hands off raw latents in
``(B, ldim, T)`` shape; ``MimiModel`` then runs a four-step chain to turn
those into a 24 kHz waveform. Production code path
(``TTSModel.generate_audio_stream``):

    raw_latent * emb_std + emb_mean   -> (B, ldim, T)
    quantizer(...)                    -> (B, dim, T)         (1x1 Conv1d)
    upsample(...)                     -> (B, dim, T*S)       (depthwise ConvTr)
    decoder_transformer(...)          -> (B, dim, T*S)       (Linear+SDPA xN)
    decoder(...)                      -> (B, channels, T_au) (SEANet stack)

This script wraps all four into one stateless module so the Rust runtime
only needs to call a single ``mimi_decode.nnef.tgz`` graph after the
autoregressive loop -- no Python in the inference path.

Run:
    # Mini, fast (random weights):
    python mimi_decode.py --mini --skip-io-check
    # Real Pocket-TTS checkpoint:
    python mimi_decode.py --full --skip-io-check
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from pocket_tts import TTSModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from torch import nn

from examples.tts.pocket_tts._io_attention import (
    apply_rope_at_positions,
    make_rope_freqs,
)
from examples.tts.pocket_tts.decoder import replace_streaming_with_stateless
from torch_to_nnef import TractNNEF, export_model_to_nnef


class BulkSelfAttention(nn.Module):
    """Stateless mirror of ``StreamingMultiheadAttention`` for bulk decode.

    Reuses the trained ``in_proj`` / ``out_proj`` / RoPE settings of the
    streaming module but evaluates attention over the full sequence
    without touching the streaming KV cache. Causal mask + RoPE positions
    are derived from ``q.shape[1]`` directly -- which becomes a tract
    symbol when the parent graph declares ``T_LATENT`` dynamic.

    Forward accepts ``*_args`` so it's a drop-in replacement for
    ``StreamingMultiheadAttention.forward(x, model_state)``: the layer's
    ``_sa_block`` keeps doing the residual add unchanged. Inlining the
    attention math here also dodges a beartype check on
    ``rope_offset(batch_size: int)`` upstream which fails under tracing
    because ``b = projected.shape[0]`` is a SymInt-like tensor.
    """

    def __init__(self, attn: StreamingMultiheadAttention):
        super().__init__()
        self.in_proj = attn.in_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.dim_per_head
        # Per-layer attention ``context``: decoder_transformer layers ship
        # with sliding-window attention; we keep whatever value the loaded
        # checkpoint configured.
        self.context = attn.context
        self.register_buffer(
            "rope_freqs",
            make_rope_freqs(self.head_dim, attn.rope.max_period),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.in_proj(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)
        positions = torch.arange(t, device=x.device, dtype=torch.long)
        q, k = apply_rope_at_positions(q, k, positions, self.rope_freqs)
        # Causal mask with optional sliding-window cap.
        delta = positions.view(-1, 1) - positions.view(1, -1)
        keep = delta >= 0
        if self.context is not None:
            keep = keep & (delta < self.context)
        attn_mask = torch.where(
            keep,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), float("-inf"), dtype=q.dtype, device=q.device),
        )
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask[None, None],
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MimiDecodePath(nn.Module):
    """Latent-to-waveform bulk decode through Mimi, fully stateless.

    Wraps the four production submodules (denormalisation buffers,
    quantizer Conv1d, upsample depthwise ConvTr, decoder_transformer
    bulk-attention path, SEANet decoder), with every ``StreamingConv1d``
    / ``StreamingConvTranspose1d`` swapped out for static-graph mirrors
    (see ``decoder.py``). All attention runs in the standalone
    (``model_state=None``) path so there's no in-place KV-cache mutation
    to trace.
    """

    def __init__(
        self,
        mimi: MimiModel,
        emb_std: torch.Tensor,
        emb_mean: torch.Tensor,
        convtr_trim: bool = True,
    ):
        super().__init__()
        # ``emb_std`` / ``emb_mean`` are FlowLM's per-ldim training-EMA
        # statistics. We bake their value at export time as buffers; if a
        # checkpoint update changes them, re-export.
        # Everything we touch needs a deepcopy because we patch streaming
        # convs in place; the loaded TTSModel may still be used by other
        # callers (e.g. zoo tests in the same process).
        mimi = copy.deepcopy(mimi)
        for name, mod in mimi.named_modules():
            if isinstance(mod, StatefulModule):
                mod._module_absolute_name = name
        # ``convtr_trim`` is the only knob ``--for-pulse`` flips: drop
        # the manual post-convtr tail-trim so tract-pulse owns the
        # overlap-add. Conv left-pad stays enabled either way -- tract
        # pulsifies plain ``Pad`` ops natively, and the SEANet residual
        # branches need the pad in place to keep their shapes aligned.
        replace_streaming_with_stateless(mimi.upsample, convtr_trim=convtr_trim)
        replace_streaming_with_stateless(
            mimi.decoder_transformer, convtr_trim=convtr_trim
        )
        replace_streaming_with_stateless(mimi.decoder, convtr_trim=convtr_trim)
        # Swap streaming attention for the bulk variant. The layer's
        # ``_sa_block`` keeps running unchanged: it calls
        # ``self.self_attn(x, model_state)`` and adds the residual; our
        # ``BulkSelfAttention.forward(x, *_args, **_kwargs)`` absorbs the
        # ignored ``model_state`` arg.
        for layer in mimi.decoder_transformer.transformer.layers:
            layer.self_attn = BulkSelfAttention(layer.self_attn)
        self.quantizer = mimi.quantizer
        self.upsample = mimi.upsample
        self.decoder_transformer = mimi.decoder_transformer
        self.decoder = mimi.decoder
        self.register_buffer("emb_std", emb_std.detach().clone())
        self.register_buffer("emb_mean", emb_mean.detach().clone())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: ``(B, ldim, T)``. Production hands FlowLM's per-step
        # ``(B, ldim)`` outputs stacked into ``(B, T, ldim)`` and then
        # ``transpose(-1, -2)`` before the quantizer (see
        # ``tts_model.py:_decode_latents_worker``); the Rust CLI builds
        # the stack channel-first so the input here is already
        # post-transpose.
        emb_std = self.emb_std.view(1, -1, 1)
        emb_mean = self.emb_mean.view(1, -1, 1)
        x = latent * emb_std + emb_mean
        x = self.quantizer(x)  # (B, dim, T)
        # ``model_state=None`` falls into the stateless path inside the
        # streaming attention; the conv leaves are already patched.
        x = self.upsample(x, None)  # (B, dim, T*S)
        (x,) = self.decoder_transformer(
            x, None
        )  # (B, T*S, dim) -> (B, dim, T*S)
        return self.decoder(x, None)  # (B, channels, T_audio)


_MINI_NOT_SUPPORTED = (
    "mini mode for the full mimi decode is intentionally unsupported; "
    "use ``decoder.py --mini`` (SEANet only) for the mini smoke path."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./mimi_decode.nnef.tgz")
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
        help="(Not supported -- mini path uses ``decoder.py`` for the "
        "SEANet leaf only; the full Mimi chain only makes sense at real "
        "dims.)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load the real Pocket-TTS checkpoint and export the full "
        "latent->audio Mimi path as one NNEF graph.",
    )
    parser.add_argument(
        "--latent-frames",
        type=int,
        default=50,
        help="Number of FlowLM-emitted latent frames used as the *example* "
        "shape during tracing. The exported graph declares ``T_LATENT`` as "
        "a dynamic axis, so the Rust runtime can call it with any positive "
        "frame count -- this value only seeds tract's shape inference.",
    )
    parser.add_argument(
        "--for-pulse",
        action="store_true",
        help="Skip the manual post-convtr tail trim so tract's pulse "
        "machinery handles the overlap-add. Required for ``tract --pulse N``.",
    )
    args = parser.parse_args()

    if args.mini:
        print(_MINI_NOT_SUPPORTED)
        raise SystemExit(2)

    print("loading real Pocket-TTS for Mimi audio decode")
    tts = TTSModel.load_model()
    model = MimiDecodePath(
        tts.mimi,
        tts.flow_lm.emb_std,
        tts.flow_lm.emb_mean,
        convtr_trim=not args.for_pulse,
    ).eval()
    ldim = tts.flow_lm.ldim
    print(
        f"  ldim={ldim} mimi.dim={tts.mimi.dimension} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
        f"latent_frames={args.latent_frames}"
    )

    latent = torch.randn(1, ldim, args.latent_frames, dtype=torch.float32)
    with torch.no_grad():
        out = model(latent)
    print(f"PyTorch output shape: {tuple(out.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    # Declare ``T_LATENT`` dynamic so the same graph runs at any latent
    # frame count -- the autoregressive loop can stop on EOS instead of
    # being padded to the trace shape. T at the input is symbolic;
    # downstream T*upsample (after ``upsample``) propagates as a derived
    # symbol through tract.
    target = TractNNEF(
        version=tract_version,
        check_io=not args.skip_io_check,
        reify_sdpa_operator=False,
        dynamic_axes={"latent": {2: "T_LATENT"}},
    )
    # Hand tract symbolic constraints it can use during pulse-mode
    # simplification. ``T_LATENT >= 1`` rules out the singleton-broadcast
    # branch when dim expressions differ across paths; an upper bound
    # keeps the search space bounded.
    # t2n's ``custom_extensions`` already prepends the NNEF ``extension``
    # keyword to each entry, so we pass just ``tract_assert <expr>``.
    custom_extensions = [
        "tract_assert T_LATENT >= 1",
        "tract_assert T_LATENT <= 1024",
    ]
    print(f"Exporting mimi_decode to {args.out} (tract {tract_version})")
    export_model_to_nnef(
        model=model,
        args=(latent,),
        file_path_export=args.out,
        inference_target=target,
        input_names=["latent"],
        output_names=["audio"],
        custom_extensions=custom_extensions,
        debug_bundle_path=Path("./debug_pocket_tts_mimi_decode.tgz"),
    )
    print("done")


if __name__ == "__main__":
    main()
