"""Bake a Pocket-TTS voice prompt into an NNEF ``.dat`` tensor.

The exported ``flow_lm_init`` graph takes the per-layer KV cache prefix as
an explicit ``past_kv`` input. This script generates that prefix once from
an audio prompt and serialises it to ``voice.dat`` via t2n's
``write_nnef_tensor`` -- the same writer t2n already uses for graph weights,
so the Rust runtime can load it with NNEF's standard tensor reader.

Modes:

* ``--mini`` (default): synthesise a deterministic random tensor with the
  right ``(n_layers, 2, B, T_voice, H, D)`` shape for the mini config the
  example exports use. Bundled into ``voices/alba.dat`` as a smoke-test
  asset -- doesn't produce real speech with random weights, just lets the
  Rust CLI demo wire end-to-end.
* ``--from-audio AUDIO``: load the real Pocket-TTS checkpoint (gated HF
  download), run ``model.get_state_for_audio_prompt(AUDIO)``, harvest the
  flow_lm self-attention caches, stack and save. This is the production
  path.

Run (mini, deterministic):
    python bake_voice.py --mini --out voices/alba.dat
Run (production):
    python bake_voice.py --from-audio path/to/voice.wav --out voices/custom.dat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pocket_tts import TTSModel

from torch_to_nnef.nnef_io.writer import write_nnef_tensor


def harvest_voice_state(
    model, audio_conditioning, truncate: bool = False
) -> torch.Tensor:
    """Run ``get_state_for_audio_prompt`` and stack flow_lm KV caches.

    Returns a single tensor of shape ``(n_layers, 2, B, T_voice, H, D)``
    where ``2`` is ``[K, V]`` -- the layout the exported ``flow_lm_init`` /
    ``flow_lm_step`` graphs expect for ``past_kv``.
    """
    state = model.get_state_for_audio_prompt(
        audio_conditioning, truncate=truncate
    )
    layers = []
    for module_name, sub_state in state.items():
        if not module_name.startswith("transformer.layers."):
            continue
        if "self_attn" not in module_name:
            continue
        cache = sub_state["cache"]  # (2, B, T_max, H, D), NaN beyond offset
        offset = int(sub_state["offset"].view(-1)[0].item())
        layers.append(cache[:, :, :offset].clone())
    if not layers:
        raise RuntimeError(
            "no per-layer KV caches harvested from voice state; "
            "check whether the model exposes them via "
            "transformer.layers.<i>.self_attn"
        )
    return torch.stack(layers, dim=0)  # (n_layers, 2, B, T_voice, H, D)


def bake_mini(out_path: Path, seed: int = 0) -> torch.Tensor:
    """Deterministic placeholder voice tensor matching the mini export shape.

    Mirrors the dimensions in ``flow_lm.py:build_mini_flow_lm``
    (``num_layers=2``, ``num_heads=2``, ``head_dim=8``) at a representative
    ``T_voice=4``. Fixed-seed random so the bundled asset is reproducible.
    """
    n_layers, batch, n_heads, head_dim, t_voice = 2, 1, 2, 8, 4
    g = torch.Generator().manual_seed(seed)
    voice = torch.randn(
        n_layers,
        2,
        batch,
        t_voice,
        n_heads,
        head_dim,
        generator=g,
        dtype=torch.float32,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_nnef_tensor(voice.numpy(), str(out_path), quantized=False)
    print(f"baked mini voice -> {out_path} shape={tuple(voice.shape)}")
    return voice


def bake_from_audio(
    audio_conditioning: Path | str,
    out_path: Path,
    truncate: bool = True,
) -> torch.Tensor:
    """Production path: real Pocket-TTS checkpoint + audio prompt -> voice.dat.

    ``TTSModel.load_model`` triggers a gated HF download on first call.
    Accepts either a local path or a ``hf://`` URL the way Pocket-TTS does.
    ``truncate`` matches Pocket-TTS' own default: the voice prompt is
    capped at 30 s of audio so a long input doesn't blow up the trace
    shape (``T_voice``) the ``flow_lm_init`` graph is exported at.
    """
    model = TTSModel.load_model()
    voice = harvest_voice_state(model, audio_conditioning, truncate=truncate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_nnef_tensor(voice.numpy(), str(out_path), quantized=False)
    print(
        f"baked {audio_conditioning} -> {out_path} shape={tuple(voice.shape)} "
        f"(truncate={truncate})"
    )
    return voice


# Default voice-prompt URL: Kyutai's "alba" voice from the public catalog.
DEFAULT_VOICE_HF_URL = "alba"

# Six bundled voices for the demo, chosen so all hit Pocket-TTS' 30 s
# truncation cap and therefore share ``T_voice`` (126 frames at the
# encoder's frame rate). That's required because ``flow_lm_init.nnef.tgz``
# is exported at a single static ``T_VOICE`` -- multiple voices can share
# the same graph only if their KV-prefix lengths agree.
# 3 female / 3 male across 4 source datasets (alba-mackenna,
# voice-donations, expresso, EARS, VCTK) for variety.
BUNDLED_VOICES = ["alba", "marius", "cosette", "jean", "mary", "charles"]


def bake_bundled(
    out_dir: Path, voices: list[str], truncate: bool = True
) -> None:
    """Bake the bundled-demo voice catalogue, one ``.dat`` per voice.

    Loads the Pocket-TTS checkpoint once and reuses it across voices so
    the HF download cost is amortised. Asserts that all baked voices
    share the same ``T_voice`` -- mismatched lengths would silently
    break the static ``flow_lm_init`` trace shape.
    """
    model = TTSModel.load_model()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_voices: dict[str, int] = {}
    for name in voices:
        voice = harvest_voice_state(model, name, truncate=truncate)
        t_voices[name] = voice.shape[3]
        out_path = out_dir / f"{name}.dat"
        write_nnef_tensor(voice.numpy(), str(out_path), quantized=False)
        print(f"baked {name:<10s} -> {out_path} T_voice={voice.shape[3]}")
    seen = set(t_voices.values())
    if len(seen) > 1:
        raise RuntimeError(
            f"baked voices have inconsistent T_voice: {t_voices}; "
            "the static ``flow_lm_init`` graph only handles one length"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("voices/alba.dat"),
        help="Path to write the NNEF ``.dat`` tensor (single-voice modes).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("voices"),
        help="Directory to write per-voice ``.dat`` files in ``--bundled`` mode.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Bake a deterministic placeholder tensor matching the mini "
        "export shape (default; only mode that runs without HF auth).",
    )
    parser.add_argument(
        "--from-audio",
        type=str,
        default=None,
        help="Audio file (local path or ``hf://`` URL or one of the Pocket-TTS "
        "predefined voice names like ``alba``) to derive the voice from. "
        "Production path; needs the real Pocket-TTS checkpoint.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Bake from the default voice prompt (Kyutai's ``alba``) via "
        "``Pocket-TTS``'s built-in voice catalogue. Equivalent to "
        "``--from-audio alba``.",
    )
    parser.add_argument(
        "--bundled",
        action="store_true",
        help="Bake the demo voice catalogue (alba, marius, cosette, jean, "
        "mary, charles) into ``--out-dir``, one ``.dat`` per voice. All "
        "share ``T_voice`` so the same exported ``flow_lm_init`` graph "
        "can switch between them at runtime.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for ``--mini`` mode (so the bundled asset is reproducible).",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Disable Pocket-TTS' default 30 s voice-prompt truncation. "
        "Long inputs increase ``T_voice`` and grow the ``flow_lm_init`` "
        "trace shape proportionally.",
    )
    args = parser.parse_args()

    truncate = not args.no_truncate
    if args.bundled:
        bake_bundled(args.out_dir, BUNDLED_VOICES, truncate=truncate)
        return
    audio = args.from_audio
    if audio is None and args.full:
        audio = DEFAULT_VOICE_HF_URL
    if audio is not None:
        bake_from_audio(audio, args.out, truncate=truncate)
    else:
        bake_mini(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
