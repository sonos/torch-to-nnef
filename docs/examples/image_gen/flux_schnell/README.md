# Flux-Schnell (TODO)

Target repo: `black-forest-labs/FLUX.1-schnell`.

12B-parameter DiT variant. Key challenges for export: model size (may not fit
a single NNEF tensor file), T5-XXL encoder (11B by itself), double-stream DiT
blocks with separate image/text token streams.

Suggested order: VAE decoder -> T5 text encoder -> DiT (the hardest).
