# Pocket-TTS

Target repo: [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts).

End-to-end Pocket-TTS through tract: a single Rust binary takes text + a
voice prompt and writes a 24 kHz WAV. No Python in the inference path.

Performance on Apple M4 Pro, pulse-mode + worker-thread streaming
(`--pulse 2`, default after the streaming work landed). Numbers are
apples-to-apples vs Kyutai's own `python -m pocket_tts generate` on
the same machine, both warm (model loaded, first call discarded):

| Implementation                  | "hello I am ..." (~2.2 s) | TIMIT phrase (~3.0 s) |
| ------------------------------- | ------------------------- | --------------------- |
| Kyutai PyTorch FP32             | 9.71×                     | 9.92×                 |
| **Our tract FP32**              | ~8.4×                     | 8.71×                 |
| **Our tract FP16** (`--fp16`)   | **9.94×**                 | **9.88×**             |

Tract FP16 is at parity with Kyutai's PyTorch FP32 reference on M4 Pro
(slight edge on the canonical phrase, slight deficit on TIMIT — within
noise).

GPU (Metal) is currently a small win: bulk-mode Mimi runs ~5× faster
on Metal but pulse-mode CPU + threading already overlaps mimi_decode
with the AR loop, so the wall is dominated by `flow_lm_step` (M=1
GEMVs hit `apple_amx_mmm_f16_64x1` / `_f32_32x1`).

Optional: `cargo build --release --features transformers-detect` runs
`tract-transformers`' SDPA / RoPE / KV-cache detection rewrites before
optimization. On the current tract version it's a slight CPU
pessimisation (`Sdpa::eval` rebuilds a sub-graph per call rather than
dispatching a fast kernel) and a marginal GPU speedup. Kept feature-
gated so the codepath is ready when tract ships a fast Sdpa CPU kernel.

Pocket-TTS architecture: a `FlowLM` autoregressive transformer (text +
voice prompt → continuous audio latents) followed by a `Mimi` neural
codec decoder (latents → 24 kHz waveform). This example exports four
NNEF graphs and threads them together in Rust.

## Bundle size

Single-voice deployable on disk (`MODE=full` after `./run.sh`):

| Component                       | Size  |
| ------------------------------- | ----- |
| `pocket-tts-tract` binary       | 23 MB |
| `tokenizer.model` (SentencePiece) | 60 KB |
| `flow_lm_init.nnef.tgz` (fp16)  | 153 MB |
| `flow_lm_step.nnef.tgz` (fp16)  | 145 MB |
| `flow_net.nnef.tgz` (fp32)      | 38 MB |
| `mimi_decode.nnef.tgz` (fp32)   | 40 MB |
| `voices/alba.dat`               | 6 MB  |
| **Total** (1 voice, fp16 FlowLM) | **~403 MB** |
| All 6 bundled voices            | +30 MB |

The two FlowLM graphs duplicate weights (separate NNEF archives); a
shared-asset packaging step would shave another ~150 MB. `flow_net`
and `mimi_decode` are still fp32 — fp16 export there is an open
follow-up (~40 MB more saved if validated).

## Status

Working end-to-end demo with pulse-mode streaming. Resolved since the
initial bulk-mode revision:

- ✅ **Pulse-mode Mimi decode** with worker-thread overlap (`--pulse 2`).
  Total wall ≈ max(AR loop, Mimi decode). Surfaced two tract pulse-mode
  bugs (LCM-merge of stream-axis dims, Deconv overlap-add bias double-add)
  that are tracked in tract PRs #2202 and #2204.
- ✅ **fp16 export** for FlowLM (`--fp16`) — halves the FlowLM disk
  footprint and ~13% per-step speedup. Surfaced an `aten::layer_norm`
  upcast gap in t2n (mirrored from `batch_norm`'s `force_norm_in_f32`
  pattern in this PR).

Known follow-ups (none blocking):

- **fp16 FlowLM ships with `force_attention_inner_in_f32=False`** because
  t2n's SDPA upcast produces structurally wrong outputs on Pocket-TTS
  (audible as a different language); pure f16 SDPA matches f32 within
  ~1%. Worth a separate t2n investigation.
- **Q4_0 export path** through tract's block-quant infrastructure
  is the preferred next step for further speedup. Tract has
  AMX-friendly dot-product kernels for 4-bit weights; would require
  a t2n quantization-aware export.
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
Sample output on M4 Pro, pulse-mode + threading + fp16 FlowLM:

```
EOS at frame 33 (logit -3.496 > -4)
generated 33 audio latents
decoded 63360 samples (2.64 s @ 24000 Hz) in 0.27 s wall time -- RTFx 9.94
  breakdown: init 16 ms / flow_lm_step 151 ms (32 steps, 4.7 ms/step) / flow_net 21 ms / mimi_decode 161 ms
```

`flow_lm_step` and `mimi_decode` run concurrently via a worker thread,
so the wall is `max(flow_path, mimi_path)`.

Add `--gpu` to route the four graphs through tract's Metal runtime
(macOS only).

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
| `--pulse N` | 0 (bulk) | Pulse size for streaming Mimi decode (`N=2` = 160 ms audio per call). Runs concurrently with the AR loop via a worker thread. |
| `--fp16` (export-side) | off | `flow_lm.py --fp16`: cast FlowLM weights to half precision; export sets `force_norm_in_f32=True` so LayerNorm stays in f32 for stability. ~13% per-step speedup, ~150 MB smaller on disk. |

Stdout reports RTFx (audio_seconds / wall_seconds) on every run.
