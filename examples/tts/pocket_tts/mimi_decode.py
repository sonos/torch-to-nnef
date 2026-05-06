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
from pocket_tts import TTSModel
import torch.nn.functional as F
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.mimi_transformer import StreamingTransformerLayer
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef

from examples.tts.pocket_tts._io_attention import (
    apply_rope_at_positions,
    make_rope_freqs,
)
from examples.tts.pocket_tts.decoder import replace_streaming_with_stateless


class BulkSelfAttention(nn.Module):
    """Stateless mirror of ``StreamingMultiheadAttention`` for bulk decode.

    Reuses the trained ``in_proj`` / ``out_proj`` / RoPE settings of the
    streaming module but evaluates attention over the full sequence
    without touching the streaming KV cache. Causal mask + RoPE positions
    are derived from ``q.shape[1]`` directly. This is what
    ``StreamingMultiheadAttention.forward(x, model_state=None)`` does
    semantically; we re-implement it inline to dodge a beartype check on
    ``rope_offset(batch_size: int)`` that fails under tracing because
    ``b = projected.shape[0]`` is a SymInt-like tensor.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.in_proj(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)
        positions = torch.arange(t, device=x.device, dtype=torch.long)
        q, k = apply_rope_at_positions(q, k, positions, self.rope_freqs)
        # Causal mask with optional sliding-window cap (decoder_transformer
        # ships with ``context=250``).
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
        out = out.transpose(1, 2).reshape(
            b, t, self.num_heads * self.head_dim
        )
        return self.out_proj(out)


def _patch_attention_to_bulk(layer: StreamingTransformerLayer) -> None:
    layer.self_attn = BulkSelfAttention(layer.self_attn)
    # ``StreamingTransformerLayer._sa_block`` calls
    # ``self.self_attn(x, model_state)`` -- our bulk attn ignores any extra
    # args via the Module's keyword/positional flexibility, but to keep
    # signatures clean we monkey-patch ``_sa_block`` to a single-arg call.
    orig_sa_block = layer._sa_block

    def _bulk_sa_block(self, x, _model_state):  # noqa: ARG001
        return self.layer_scale_1(self.self_attn(self.norm1(x)))

    layer._sa_block = _bulk_sa_block.__get__(layer, type(layer))
    _ = orig_sa_block  # silence linter


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

    def __init__(self, mimi: MimiModel, emb_std: torch.Tensor, emb_mean: torch.Tensor):
        super().__init__()
        # Everything we touch needs a deepcopy because we patch streaming
        # convs in place; the loaded TTSModel may still be used by other
        # callers (e.g. zoo tests in the same process).
        mimi = copy.deepcopy(mimi)
        for name, mod in mimi.named_modules():
            if isinstance(mod, StatefulModule):
                mod._module_absolute_name = name
        replace_streaming_with_stateless(mimi.upsample)
        replace_streaming_with_stateless(mimi.decoder_transformer)
        replace_streaming_with_stateless(mimi.decoder)
        for layer in mimi.decoder_transformer.transformer.layers:
            _patch_attention_to_bulk(layer)
        self.quantizer = mimi.quantizer
        self.upsample = mimi.upsample
        self.decoder_transformer = mimi.decoder_transformer
        self.decoder = mimi.decoder
        self.register_buffer("emb_std", emb_std.detach().clone())
        self.register_buffer("emb_mean", emb_mean.detach().clone())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, ldim, T) -- the Rust CLI dumps this layout.
        emb_std = self.emb_std.view(1, -1, 1)
        emb_mean = self.emb_mean.view(1, -1, 1)
        x = latent * emb_std + emb_mean
        x = self.quantizer(x)  # (B, dim, T)
        # ``model_state=None`` falls into the stateless path inside the
        # streaming attention; the conv leaves are already patched.
        x = self.upsample(x, None)  # (B, dim, T*S)
        (x,) = self.decoder_transformer(x, None)  # (B, T*S, dim) -> (B, dim, T*S)
        return self.decoder(x, None)  # (B, channels, T_audio)


def build_mini_path() -> MimiDecodePath:
    """Tiny decode chain mirroring the production structure.

    Real Mimi: ``dimension=512, ratios=[6,5,4]`` plus a 32-dim quantizer
    output_proj and a 6.3M-param decoder_transformer. Skipping the
    decoder_transformer + upsample makes the mini path significantly
    smaller; we cover them in ``--full``. The mini config keeps the
    quantizer 1x1 conv + SEANet decoder and degenerate upsample so the
    CLI pipeline can still smoke-test end to end.
    """
    raise NotImplementedError(
        "mini mode for the full mimi decode is intentionally unsupported -- "
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
    args = parser.parse_args()

    if args.mini:
        build_mini_path()  # raises NotImplementedError with guidance.

    print("loading real Pocket-TTS for Mimi audio decode")
    tts = TTSModel.load_model()
    model = MimiDecodePath(
        tts.mimi, tts.flow_lm.emb_std, tts.flow_lm.emb_mean
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

    tract_version = args.tract_version or "0.23.0-dev.5"
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
    print(f"Exporting mimi_decode to {args.out} (tract {tract_version})")
    export_model_to_nnef(
        model=model,
        args=(latent,),
        file_path_export=args.out,
        inference_target=target,
        input_names=["latent"],
        output_names=["audio"],
        debug_bundle_path=Path("./debug_pocket_tts_mimi_decode.tgz"),
    )
    print("done")


if __name__ == "__main__":
    main()
