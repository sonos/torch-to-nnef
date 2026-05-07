# Flux-Schnell

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FLUX.1--schnell-yellow)](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

12B-parameter DiT variant from Black Forest Labs. Tiny `--mini` config (random weights, full architecture, ~16k params) is validated end-to-end via `check_io` against tract; the full 12B export requires gated HF download and is not exercised in CI.

Key challenges for the full export: model size (may not fit a single NNEF tensor file), T5-XXL encoder (11B by itself), double-stream DiT blocks with separate image/text token streams.

```bash
python transformer.py --mini  # tiny config end-to-end check
```
