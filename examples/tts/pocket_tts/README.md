# Pocket-TTS

Target repo: [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts).

End-to-end Pocket-TTS through tract: a single Rust binary takes text + a
voice prompt and writes a 24 kHz WAV. No Python in the inference path.
Runs at RTFx ≈2.5× on CPU and ≈3.3× on Metal GPU for the canonical
"hello I am a text to speech voice" prompt.

Pocket-TTS architecture: a `FlowLM` autoregressive transformer (text +
voice prompt → continuous audio latents) followed by a `Mimi` neural
codec decoder (latents → 24 kHz waveform). This example exports four
NNEF graphs and threads them together in Rust.

## Status

Working end-to-end demo. Known follow-ups (none blocking):

- `flow_lm_init` traces at static `(T_TEXT, T_VOICE)` — different text
  length needs a re-export. Tract symbol relations would lift this.
- Bulk-mode Mimi decode (full utterance in one call), not the chunked
  pulse-mode streaming Mimi was designed for.
- Three small wrappers around `pocket_tts` (`BulkSelfAttention`,
  `replace_streaming_with_stateless`, a SentencePiece stub for the mini
  conditioner) that should land upstream as `bulk_decode=True` /
  `tokenizer=None` kwargs.

## Quick start

```bash
./run.sh             # mini path: random weights, output is noise
MODE=full ./run.sh   # real ~110M-param Pocket-TTS, real audio
```

`MODE=full` triggers an HF download on first run (the gated Pocket-TTS
checkpoint), exports the four graphs, builds the Rust CLI, and writes
`cli/out.wav`. Sample output on a M-series CPU:

```
EOS at frame 33 (logit -3.496 > -4)
generated 33 audio latents
decoded 63360 samples (2.64 s @ 24000 Hz) in 1.05 s wall time -- RTFx 2.53
```

Add `--gpu` to route the four graphs through tract's Metal runtime
(macOS only):

```
decoded 63360 samples (2.64 s @ 24000 Hz) in 0.95 s wall time -- RTFx 2.79 [Metal GPU]
```

## Layout

| Script           | Exports                                                       |
| ---------------- | ------------------------------------------------------------- |
| `flow_net.py`    | LSD denoiser (`flow_net.nnef.tgz`)                            |
| `flow_lm.py`     | Autoregressive transformer (`flow_lm_init`, `flow_lm_step`)   |
| `decoder.py`     | Mini-only SEANet decoder (`decoder.nnef.tgz`)                 |
| `mimi_decode.py` | Full Mimi decode chain (`mimi_decode.nnef.tgz`)               |
| `bake_voice.py`  | Voice prompt KV prefix (`voices/alba.dat`)                    |
| `extract_tokenizer.py` | SentencePiece model from the Pocket-TTS checkpoint      |

The full Mimi decode graph wraps four submodules into a single stateless
NNEF graph: latent denormalisation, the quantizer 1×1 conv (ldim → mimi
dim), the depthwise transposed-conv upsample, the decoder transformer,
and the SEANet decoder. Streaming convs and the streaming KV-cache
attention are mirrored with stateless wrappers (see `decoder.py` and the
`BulkSelfAttention` class in `mimi_decode.py`); weights are reused as-is.

The graph declares `T_LATENT` as a dynamic axis, so the same exported
artifact runs at any frame count -- the autoregressive loop terminates
on real EOS (default threshold `-4.0`, matching Pocket-TTS) and feeds
however many latents it produced into the decoder. Audio length tracks
the utterance: same prompt across noise seeds 0/1/2/7 gives
33/29/28/27 audio frames (2.64 / 2.32 / 2.24 / 2.16 s).

This is still a **bulk** decode (full utterance in one call), not the
chunked pulse-mode streaming Mimi was designed for; that's the next
step.

## CLI flags

The Rust binary lives in [`cli/`](cli/). Demo-relevant flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--text "..."` | -- | Requires `--tokenizer`. |
| `--tokenizer tokenizer.model` | -- | SentencePiece model from `extract_tokenizer.py`. |
| `--voice voices/alba.dat` | bundled mini | Output of `bake_voice.py`. |
| `--ldim` | 8 | 32 for the real checkpoint. |
| `--max-frames` | 32 | Safety cap; loop terminates on EOS first. |
| `--lsd-steps` | 1 | Pocket-TTS default. |
| `--temp` | 0.7 | Initial-noise std is `sqrt(temp)`. |
| `--noise-clamp` | -- | Optional truncated-normal bound. |
| `--eos-threshold` | -4.0 | Pocket-TTS default. |
| `--seed` | 0 | Reproducible noise. |
| `--gpu` | off | Metal GPU runtime (macOS only). |

Stdout reports RTFx (audio_seconds / wall_seconds) on every run.
