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

The simplest path is the parent directory's `run.sh` -- it exports the
graphs, builds the binary, and synthesises a fixed-text WAV.

```bash
cd ..
./run.sh             # mini, noise output
MODE=full ./run.sh   # real Pocket-TTS, real audio
```

Direct invocation (after the assets are exported in `cli/models/`,
`cli/voices/alba.dat`, and `cli/tokenizer.model`):

```bash
./target/release/pocket-tts-tract \
    --models models \
    --voice voices/alba.dat \
    --tokenizer tokenizer.model \
    --text "Hello, world." \
    --ldim 32 --max-frames 50 --eos-threshold 1e9 \
    --out hello.wav
```

`--max-frames` is just a safety cap; the loop terminates on EOS (default
threshold `-4.0`, matching Pocket-TTS' own CLI default). The `mimi_decode`
graph declares `T_LATENT` as dynamic, so audio length scales with the
actual utterance.

## Mini-only knobs

For the random-weights mini path the conditioner has no real vocabulary,
so pass raw token IDs:

```bash
./target/release/pocket-tts-tract --tokens 1,2,3,4 --max-frames 8 --out out.wav
```

The mini decoder is traced at 8 latent frames -- match `--max-frames` to
that.
