# Stable Diffusion 1.5

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-runwayml%2Fstable--diffusion--v1--5-yellow)](https://huggingface.co/runwayml/stable-diffusion-v1-5) [![arXiv](https://img.shields.io/badge/arXiv-2112.10752-b31b1b.svg)](https://arxiv.org/abs/2112.10752)

First image-generation target in this directory. Two scripts exporting the heaviest components separately so the full checkpoint never sits in memory at once:

- **`vae_decoder.py`**: VAE decoder (latent -> RGB upsample path). Validated end-to-end via `check_io` against tract.
- **`unet.py`**: UNet denoiser. Validated end-to-end via `check_io` against tract.

The CLIP text encoder is not exported here (small enough to be handled separately if needed).

## Run

```bash
cd examples/image_gen/sd15
pip install -r ../requirements.txt
python vae_decoder.py
python unet.py
```

Each script runs `export_model_to_nnef(check_io=True)` so a tract numerical mismatch fails the export loudly.
