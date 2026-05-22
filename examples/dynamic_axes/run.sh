#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Optional: bootstrap a local venv via uv (uncomment if desired)
# ../bootstrap-uv.sh

echo "[dynamic_axes] Running example exports..."
python export_albert_fixed.py
python export_with_batchable.py
python cnn_deepspeech_stream.py
echo "[dynamic_axes] Done. See generated .nnef.tgz artifacts in this folder."

