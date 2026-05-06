# Pocket-TTS

Target repo: [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts).

A lightweight CPU-friendly TTS from Kyutai (~100M params, multi-language,
voice cloning). It's a `FlowLM` (text + voice prompt -> continuous audio
latents) feeding into a `Mimi` neural codec decoder (latents -> 24 kHz
waveform).

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
dim), the depthwise transposed-conv upsample, the decoder transformer, and
the SEANet decoder. Streaming convs and the streaming KV-cache attention
are mirrored with stateless wrappers (see `decoder.py` and the
`BulkSelfAttention` class in `mimi_decode.py`); weights are reused as-is.

The graph declares `T_LATENT` as a dynamic axis, so the same exported
artifact runs at any frame count -- the autoregressive loop terminates on
real EOS and feeds however many latents it produced into the decoder. This
is still a *bulk* decode (full utterance in one call), not the chunked
pulse-mode streaming Mimi was designed for; that is the next step.

## Run

`run.sh` does the whole pipeline (export + bake voice + build CLI + run).

```bash
./run.sh             # mini path: random weights, noise out
MODE=full ./run.sh   # real ~110M-param Pocket-TTS, real audio at 24 kHz
```

In `MODE=full` everything runs through tract: there is no Python in the
inference path. The shippable artefact is the Rust binary plus the
`cli/models/`, `cli/voices/`, and `cli/tokenizer.model` asset directory.
