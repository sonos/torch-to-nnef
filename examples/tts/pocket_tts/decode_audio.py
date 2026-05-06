"""Decode FlowLM-emitted audio latents into a WAV via Pocket-TTS' Mimi.

Hybrid step: the Rust CLI runs the autoregressive 89M-param FlowLM through
tract and dumps the per-frame audio latents to ``latents.npz``; this script
loads them, runs them through the real ``MimiModel.decode_from_latent`` (the
20M-param SEANet decoder + transformer + upsample chain), and writes a
24 kHz WAV. Once the Mimi decode chain is exported to NNEF this step
goes away.

Run:
    python decode_audio.py --latents cli/latents.npz --out cli/out.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import StatefulModule, init_states
from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latents",
        type=Path,
        required=True,
        help="``.npz`` written by the Rust CLI's ``--dump-latents``.",
    )
    parser.add_argument("--out", type=Path, default=Path("out.wav"))
    args = parser.parse_args()

    print("loading Pocket-TTS Mimi (audio decoder)")
    model = TTSModel.load_model()
    for name, mod in model.mimi.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = name

    npz = np.load(args.latents)
    # Rust CLI dumps as ``(B, ldim, T)``. Pocket-TTS' production handover
    # to Mimi does: denormalise via ``flow_lm.emb_std / emb_mean``,
    # transpose ``(B, ldim, T) -> (B, T, ldim)``, run through the
    # quantizer (which projects ``ldim -> mimi.dimension``), then decode.
    raw = torch.from_numpy(npz["latents"]).to(torch.float32)
    print(f"loaded latents: shape={tuple(raw.shape)} dtype={raw.dtype}")
    # raw is already (B, ldim, T). Pocket-TTS' production code path takes
    # FlowLM's per-step (B, ldim) latents stacked into (B, T, ldim) and
    # transposes to (B, ldim, T) before the quantizer; we already have
    # the post-transpose layout, so skip the transpose. emb_std / emb_mean
    # are (ldim,); broadcast on dim 1 by reshaping to (1, ldim, 1).
    emb_std = model.flow_lm.emb_std.view(1, -1, 1)
    emb_mean = model.flow_lm.emb_mean.view(1, -1, 1)
    mimi_in = raw * emb_std + emb_mean  # (B, ldim, T)
    quantized = model.mimi.quantizer(mimi_in)
    print(
        f"projected via quantizer: shape={tuple(quantized.shape)} "
        f"(matches mimi.decoder.dimension={model.mimi.dimension})"
    )

    # decoder_transformer's KV cache has to accommodate the POST-upsample
    # length, not the raw latent length: ``upsample`` blows time up by
    # ``encoder_frame_rate / frame_rate`` (16x for real Pocket-TTS).
    upsample_factor = int(
        round(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
    )
    state = init_states(
        model.mimi,
        batch_size=quantized.shape[0],
        sequence_length=quantized.shape[2] * upsample_factor,
    )
    with torch.no_grad():
        audio = model.mimi.decode_from_latent(quantized, state)
    audio_np = audio.squeeze(0).squeeze(0).numpy()
    sr = model.sample_rate
    print(
        f"decoded audio: {audio_np.shape[0]} samples "
        f"({audio_np.shape[0] / sr:.2f}s @ {sr} Hz)"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.out, model.sample_rate, audio_np.astype(np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
