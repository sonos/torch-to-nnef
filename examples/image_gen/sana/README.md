# Sana

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Sana_1600M-yellow)](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) [![arXiv](https://img.shields.io/badge/arXiv-2410.10629-b31b1b.svg)](https://arxiv.org/abs/2410.10629)

NVIDIA Sana, an efficient text-to-image DiT with two architectural twists vs. the Flux / SD3 family already covered in this directory:

- **Linear attention** (ReLU-based) in self-attention blocks, no softmax.
- **Mix-FFN** (linear, depth-wise conv, linear) instead of plain FFN.
- **DC-AE** VAE for very high spatial compression (f32 / f64).

`transformer.py` exports the denoiser. `--mini` instantiates a tiny random-weights config (~16k params) that keeps the full architecture so the export path is exercised without a gated HF download.

```bash
python transformer.py --mini
```
