# Pocket-TTS Rust CLI

End-to-end demo binary that takes a SentencePiece tokenizer + voice prompt
+ text and writes a 24 kHz WAV, with every model call going through tract.

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
decoder (mimi SEANet → 24 kHz waveform)
   ↓
WAV
```

## Prerequisites

Export the four NNEF graphs from this directory's Python scripts:

```bash
cd ../          # examples/tts/pocket_tts/
python flow_net.py   --mini --out cli/models/flow_net.nnef.tgz
python flow_lm.py    --mini --out-init cli/models/flow_lm_init.nnef.tgz \
                            --out-step cli/models/flow_lm_step.nnef.tgz
python decoder.py    --mini --out cli/models/decoder.nnef.tgz
python bake_voice.py --mini --out cli/voices/alba.dat
```

The bundled `cli/voices/alba.dat` works for the mini config; the production
path takes a real audio prompt (`bake_voice.py --from-audio path.wav`).

## Run

```bash
cd cli
# Mini demo: deterministic plumbing test, output is not coherent speech
cargo run --release -- --tokens 1,2,3,4 --out out.wav

# Real demo (needs the production tokenizer.model + production-checkpoint exports):
cargo run --release -- --text "Hello, world." --tokenizer tokenizer.model --out hello.wav
```

The mini demo runs entirely on the `--tokens` path because the random-weights
conditioner has no real vocabulary.

## What's mock vs. real

* The four NNEF graphs are real tract export artifacts (verified by
  `pytest tests/test_model_zoo.py -k mini_pocket_tts`, which spawns the
  tract CLI for `check_io`).
* The `--mini` weights are random, so the audio output is noise -- this binary
  is a *plumbing* demo for the export-runtime path, not a TTS demo.
* Swapping in real-checkpoint exports (TODO in the parent README) gives real
  audio without changing a line of Rust.

## Known issue (in-progress)

`flow_lm_init` currently fails through the `tract-nnef` Rust library with

```
Clashing resolution for expression. 24=24 != 16. (SessionState(SymbolValues { values: {} }))
```

The same graph passes `check_io` (which runs it through the tract CLI), and
the other three graphs (`decoder`, `flow_net`, `flow_lm_step`) run cleanly
through the Rust library. The mismatch is specific to the embed + concat
path at the front of `flow_lm_init`. Until that's resolved, the CLI panics
on the first call -- track via the Rust binary build/test in PR #76 follow-ups.
