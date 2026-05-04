# SD3 (TODO)

Target repo: `stabilityai/stable-diffusion-3-medium`.

Key difference vs SD 1.5 / SDXL: DiT (diffusion transformer) architecture
instead of UNet. Three text encoders (CLIP-L, CLIP-G, T5-XXL).

Start with VAE decoder to validate the upsample path, then the DiT transformer
(expect attention / rotary-pos-embedding gotchas).
