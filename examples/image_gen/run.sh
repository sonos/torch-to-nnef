#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

echo "[image_gen] Exporting all sub-models..."
python sd15/unet.py --skip-io-check
python sd15/vae_decoder.py
python flux_schnell/transformer.py --mini
python sana/transformer.py --mini
echo "[image_gen] Done."

