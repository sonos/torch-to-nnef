#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[image_gen] Quick runs:"
echo "- SD1.5 UNet (skip IO check): python sd15/unet.py --skip-io-check"
echo "- SD1.5 VAE decoder:        python sd15/vae_decoder.py"
echo "- Flux Schnell (mini):       python flux_schnell/transformer.py --mini"
echo "- Sana (mini):               python sana/transformer.py --mini"

echo "Running a minimal default: SD1.5 UNet (skip-io-check)"
python sd15/unet.py --skip-io-check

