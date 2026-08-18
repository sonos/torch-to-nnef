# Pocket-TTS Rust CLI

Self-contained binary: takes a SentencePiece tokenizer + voice prompt +
text and writes a 24 kHz WAV. Every model call goes through tract; there
is no Python or external decoder process at runtime.

## Pipeline

```
text → SentencePiece → token IDs
   ↓
flow_lm_init (token IDs + voice.dat KV prefix → transformer hidden + KV)
   ↓ for each audio frame:
flow_net × lsd_steps (LSD denoising loop on Gaussian noise)
   ↓
flow_lm_step (next audio latent + KV → next hidden + KV)
   ↓ until EOS or --max-frames:
   collect latents
   ↓
mimi_decode (quantizer + upsample + decoder_transformer + SEANet)
   ↓ 24 kHz waveform
WAV
```

The CLI prefers `mimi_decode.nnef.tgz` (full Mimi chain) when present in
`--models` and falls back to `decoder.nnef.tgz` (SEANet only, mini mode).

## Run

The simplest path is the parent directory's `run.sh`: it exports the
graphs, builds the binary, and synthesises a fixed-text WAV.

```bash
cd ..
./run.sh             # mini, noise output
MODE=full ./run.sh   # real Pocket-TTS, real audio
```

Direct invocation (after the assets are exported in `cli/models/`,
`cli/voices/`, and `cli/tokenizer.model`):

```bash
./target/release/pocket-tts-tract \
    --models models \
    --voice-name alba \
    --voices-dir voices \
    --tokenizer tokenizer.model \
    --text "Hello, world." \
    --ldim 32 --max-frames 256 \
    --out hello.wav
```

`--voice-name` picks a bundled voice (alba/marius/cosette/jean/mary/charles)
and resolves to `<voices-dir>/<name>.dat`. Pass `--voice <path>` to use an
arbitrary baked `.dat` instead.

`--max-frames` is just a safety cap; the loop terminates on real EOS
(default threshold `-4.0`, matching Pocket-TTS' own CLI). The
`mimi_decode` graph declares `T_LATENT` as dynamic, so audio length
scales with the actual utterance.

Sample output:

```
EOS at frame 33 (logit -3.496 > -4)
generated 33 audio latents
decoded 63360 samples (2.64 s @ 24000 Hz) in 1.05 s wall time -- RTFx 2.53
wrote out.wav
```

## GPU (Metal, macOS only)

Pass `--gpu` to apply tract's `MetalTransform` to all four graphs and
wrap the generation phase in `with_metal_stream`:

```bash
./target/release/pocket-tts-tract --gpu --models models --voice voices/alba.dat \
    --tokenizer tokenizer.model --text "Hello, world." --ldim 32 \
    --max-frames 256 --out hello-gpu.wav
```

```
running through tract Metal GPU runtime
...
decoded 63360 samples (2.64 s @ 24000 Hz) in 0.95 s wall time -- RTFx 2.79 [Metal GPU]
```

GPU output matches CPU within float-quantization tolerance.

On non-macOS, `--gpu` is rejected at startup.

## Sampling controls

These mirror Pocket-TTS' own CLI defaults:

| Flag | Default | Effect |
| --- | --- | --- |
| `--lsd-steps N` | 1 | Number of LSD denoising steps per audio frame. |
| `--temp T` | 0.7 | Initial-noise std is `sqrt(T)`. |
| `--noise-clamp C` | unset | Optional symmetric truncated-normal bound. |
| `--eos-threshold X` | -4.0 | Loop terminates when raw EOS logit > X. |
| `--seed S` | 0 | Reproducible noise. |

## Mini-only knobs

For the random-weights mini path the conditioner has no real vocabulary,
so pass raw token IDs:

```bash
./target/release/pocket-tts-tract --tokens 1,2,3,4 --max-frames 8 --out out.wav
```

The mini decoder is traced at 8 latent frames, so match `--max-frames` to
that.
