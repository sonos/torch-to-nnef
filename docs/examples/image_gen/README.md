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

Flux-Schnell (DiT, 12B) is a planned follow-up: it currently exports cleanly
but fails on the tract side (SDPA + reshape deserializer panics on Flux's
RoPE'd shapes), so it is held out of this directory until the upstream tract
fixes land.

## Status

Early exploration. First targets are the SD 1.5 VAE decoder and UNet, both
validated end-to-end against tract via `check_io`.

Expected gotchas: attention (flash / sdpa variants), grouped norm fusing,
timestep embedding, cross-attention text conditioning, FP16 weights, large
safetensors loading.
