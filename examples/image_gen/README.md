# Image generation examples (exploratory)

Goal: export diffusion-based image generators to NNEF via torch-to-nnef and
check them end-to-end against tract.

Motivation: a colleague reported success running SD 1.5, SDXL, SD3 and
Flux-Schnell directly on tract (with a handful of tract-side fixes plus NNEF
`resize` support added recently). This directory explores doing the same via
torch-to-nnef rather than going through ONNX as an intermediate.

## Scope

Each subdir targets one model family and is split by component so the heaviest
tensors are not all loaded at once:

- `sd15/` -- Stable Diffusion 1.5: VAE decoder, UNet, CLIP text encoder
- `sdxl/` -- Stable Diffusion XL: two text encoders, larger UNet, VAE
- `sd3/` -- Stable Diffusion 3 (DiT architecture)
- `flux_schnell/` -- Flux-Schnell (DiT, 12B): tiny `--mini` config validated
  end-to-end via `check_io` against tract; full 12B checkpoint export needs
  a gated HF download.
- `sana/` -- NVIDIA Sana (DiT, 1.6B / 4.8B): linear-attention DiT with Mix-FFN
  and DC-AE VAE; tiny `--mini` config validated end-to-end via `check_io`.

## Status

Early exploration. First targets are the SD 1.5 VAE decoder and UNet, both
validated end-to-end against tract via `check_io`.

Expected gotchas: attention (flash / sdpa variants), grouped norm fusing,
timestep embedding, cross-attention text conditioning, FP16 weights, large
safetensors loading.
