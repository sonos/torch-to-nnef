# Pocket-TTS

Target repo: [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts).

A lightweight CPU-friendly TTS from Kyutai (~100M params, multi-language,
voice cloning). It's a `FlowLM` (text + voice prompt -> continuous audio
latents) feeding into a `Mimi` neural codec decoder (latents -> 24 kHz
waveform). The decoder is the always-on inference hot path.

`decoder.py` exports the **Mimi decoder** to NNEF. Mimi's convolutions are
streaming-stateful (in-place KV-cache-style buffers), which can't be traced
into a static graph as-is, so the script wraps each streaming conv with a
stateless mirror that reuses the trained weights and produces bit-exact
output for the bulk (non-streaming) decode case.

`--mini` is the only mode today: a tiny random-weights config that mirrors
the real Mimi structure for testing the export path without authenticating
against the gated checkpoint.

## TODO

- Load real Pocket-TTS Mimi weights (`safetensors` from `kyutai/pocket-tts`)
  and validate audio output against the reference Python decoder.
- Decoder transformer: `ProjectedTransformer` lives between the latent and
  the SEANet decoder; this script currently exports just the SEANet stack.
- FlowLM front-end: autoregressive, much trickier to slice into static
  graphs (sampling + stateful attention). Out of scope for this first pass.
