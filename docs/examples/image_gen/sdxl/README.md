# SDXL (TODO)

Target repos:
- UNet + VAE: `stabilityai/stable-diffusion-xl-base-1.0`
- Two text encoders (CLIP-L + OpenCLIP G)

Export pattern mirrors `../sd15/vae_decoder.py`. Start with the VAE decoder,
then text encoders, then the larger UNet.
