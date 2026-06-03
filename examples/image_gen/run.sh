#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

echo "[image_gen] Exporting all sub-models..."
# --skip-io-check on every sub-model: these are large (SD1.5 UNet/VAE) or
# random-weight (flux/sana --mini) models where the tract IO check is slow and
# trips on f32 outliers; we validate the export *runs and emits NNEF*. retry:
# SD1.5 / Sana weights come from HF (429s on shared CI IPs).
retry python sd15/unet.py --skip-io-check
retry python sd15/vae_decoder.py --skip-io-check
python flux_schnell/transformer.py --mini --skip-io-check
python sana/transformer.py --mini --skip-io-check
echo "[image_gen] Done."

