#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

echo "[image_gen] Exporting all sub-models..."
# vae/flux/sana run the tract IO check (they pass; the earlier failure was a
# missing onnxscript dep, now in pyproject). unet keeps --skip-io-check: that's
# the example's own intended fast mode for the huge SD1.5 UNet. retry: SD1.5
# UNet/VAE weights come from HF (429s on shared CI IPs); flux/sana --mini use
# random weights (no download).
hf_pull "stable-diffusion-v1-5/stable-diffusion-v1-5"  # retry the HF download
python sd15/unet.py --skip-io-check
python sd15/vae_decoder.py
python flux_schnell/transformer.py --mini
python sana/transformer.py --mini
echo "[image_gen] Done."

