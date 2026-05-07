# FSMN-VAD in wasm

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-funasr%2Ffsmn--vad-yellow)](https://huggingface.co/funasr/fsmn-vad) [![arXiv](https://img.shields.io/badge/arXiv-2103.04450-b31b1b.svg)](https://arxiv.org/abs/2103.04450) [![demo](https://img.shields.io/badge/live-demo-brightgreen)](https://sonos.github.io/torch-to-nnef/latest/html/demo_vad.html)

Exports a [FunASR FSMN-VAD](https://huggingface.co/funasr/fsmn-vad) voice-activity detector to NNEF and compiles a tract-backed Rust crate to WebAssembly for in-browser real-time VAD on a microphone feed.

`py/fsmn_encoder.py` is a copy of FunASR's encoder stripped of the `funasr.register` dependency so the model can be imported standalone. This example also demonstrates dynamic axes at the time dimension (streaming).

For the JIT-only export path (passing a `.jit` artifact directly with no Python source), see the sibling [`silero_vad/`](../silero_vad/) example.

## Run

```bash
cd examples/vad
./run.sh
```

The `run.sh` script:
1. Sets up `.venv` + Rust toolchain via the bootstrap helpers
2. Runs `py/export.py` to produce the NNEF archive
3. Builds the Rust crate to wasm with `wasm-pack`
4. Drops the wasm + JS glue into `docs/html/` for the live demo

Live demo: [https://sonos.github.io/torch-to-nnef/latest/html/demo_vad.html](https://sonos.github.io/torch-to-nnef/latest/html/demo_vad.html).
