# Image classifier in wasm

[![arXiv](https://img.shields.io/badge/arXiv-1905.11946-b31b1b.svg)](https://arxiv.org/abs/1905.11946) [![demo](https://img.shields.io/badge/live-demo-brightgreen)](https://sonos.github.io/torch-to-nnef/latest/html/demo_image_classifier.html)

Companion to the [Getting started tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/). Exports `torchvision.models.efficientnet_b0` (EfficientNet-B0, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks") with a dynamic batch dimension, then compiles a tract-backed Rust crate to WebAssembly so the model runs in-browser.

## Run

```bash
cd examples/imageclass-wasm
./run.sh    # bootstraps uv + rust + wasm-pack, exports the NNEF, builds the wasm
```

The `run.sh` script:
1. Sets up `.venv` + Rust toolchain via the bootstrap helpers
2. Runs `export_with_batchable.py` to produce the NNEF archive
3. Builds the Rust crate to wasm with `wasm-pack`
4. Drops the wasm + JS glue into `docs/html/` for the live demo

Live demo: [https://sonos.github.io/torch-to-nnef/latest/html/demo_image_classifier.html](https://sonos.github.io/torch-to-nnef/latest/html/demo_image_classifier.html).
