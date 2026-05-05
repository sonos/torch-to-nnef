# Sana

Target repo: `Efficient-Large-Model/Sana_1600M_1024px_diffusers`.

NVIDIA Sana — efficient text-to-image DiT with two architectural twists vs.
the Flux/SD3 family already covered in this directory:

- **Linear attention** (ReLU-based) in self-attention blocks, no softmax.
- **Mix-FFN** (linear → depth-wise conv → linear) instead of plain FFN.
- **DC-AE** VAE for very high spatial compression (f32 / f64).

`transformer.py` exports the denoiser. `--mini` instantiates a tiny random-
weights config (~16k params) that keeps the full architecture so this PR can
exercise the export path without a gated HF download.

Suggested order: transformer → DC-AE decoder → Gemma-2 / T5 text encoder.
