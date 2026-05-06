# Pocket-TTS

Target repo: [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts).

End-to-end Pocket-TTS through tract: a single Rust binary takes text + a
voice prompt and writes a 24 kHz WAV. No Python in the inference path.

Faster than realtime: RTFx ≈2.5× CPU / ≈3.3× Metal GPU on an Apple M4
Pro for the canonical "hello I am a text to speech voice" prompt.

For reference, Kyutai's own PyTorch streaming reference clocks ≈6× on a
base M4 CPU (their published number) and ≈10× on the same M4 Pro we
measured. The gap is mostly structural — see *Status* below.

Pocket-TTS architecture: a `FlowLM` autoregressive transformer (text +
voice prompt → continuous audio latents) followed by a `Mimi` neural
codec decoder (latents → 24 kHz waveform). This example exports four
NNEF graphs and threads them together in Rust.

## Status

Working end-to-end demo. Known follow-ups (none blocking):

- **Bulk-mode Mimi decode**, not the chunked pulse-mode streaming Mimi
  was designed for. Kyutai's reference overlaps Mimi decode with the
  FlowLM autoregressive loop (concurrent), so total wall time ≈
  max(loop, decode); ours is sum(loop, decode). Pulse-mode through
  tract is the path to closing the RTFx gap.
- **No quantization**. Pocket-TTS supports torchao 8-bit upstream; our
  exports are float32 throughout.
- **`past_kv.clone()` per step** in the autoregressive loop: ~8.6 MB
  redundant alloc per step at full dims. Could be amortised by a
  ring-buffer or by exposing the cache as a runtime-managed tensor.
- `flow_lm_init` traces at static `(T_TEXT, T_VOICE)` — different text
  length needs a re-export. Tract symbol relations would lift this.
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
checkpoint), exports the four NNEF graphs, bakes 6 bundled voices
(see *Voices* below), builds the Rust CLI, and writes `cli/out.wav`.
Sample output on a M-series CPU:

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

## Voices

`MODE=full` bakes six voices into `cli/voices/<name>.dat`:

| Name      | Source dataset      | Approx. character |
| --------- | ------------------- | ----------------- |
| `alba`    | alba-mackenna       | warm female       |
| `cosette` | expresso emotional  | expressive female |
| `mary`    | VCTK p333           | calm female       |
| `marius`  | voice donations     | male              |
| `jean`    | EARS p010 freeform  | male              |
| `charles` | VCTK p254           | calm male         |

All six hit Pocket-TTS' 30 s `truncate=True` cap and therefore share
`T_voice = 126`, so the same exported `flow_lm_init` graph switches
between them with no re-export. The Rust CLI selects via:

```bash
./cli/target/release/pocket-tts-tract \
    --voice-name marius \
    --voices-dir cli/voices \
    --tokenizer cli/tokenizer.model \
    --text "hello I am a text to speech voice" \
    --ldim 32 --max-frames 256 --out marius.wav
```

Same prompt across the catalogue produces distinct audio (different
speakers naturally hit EOS at different frame counts):

```
alba    -> 33 frames (2.64 s)
cosette -> 24 frames (1.92 s)
charles -> 25 frames (2.00 s)
marius  -> 19 frames (1.52 s)
```

## CLI flags

The Rust binary lives in [`cli/`](cli/). Demo-relevant flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--text "..."` | -- | Requires `--tokenizer`. |
| `--tokenizer tokenizer.model` | -- | SentencePiece model from `extract_tokenizer.py`. |
| `--voice voices/alba.dat` | bundled mini | Explicit path to a baked `.dat`. |
| `--voice-name <name>` | -- | Pick a bundled voice (alba/marius/cosette/jean/mary/charles). |
| `--voices-dir <dir>` | `voices` | Where `--voice-name` looks up `<name>.dat`. |
| `--ldim` | 8 | 32 for the real checkpoint. |
| `--max-frames` | 32 | Safety cap; loop terminates on EOS first. |
| `--lsd-steps` | 1 | Pocket-TTS default. |
| `--temp` | 0.7 | Initial-noise std is `sqrt(temp)`. |
| `--noise-clamp` | -- | Optional truncated-normal bound. |
| `--eos-threshold` | -4.0 | Pocket-TTS default. |
| `--seed` | 0 | Reproducible noise. |
| `--gpu` | off | Metal GPU runtime (macOS only). |

Stdout reports RTFx (audio_seconds / wall_seconds) on every run.
