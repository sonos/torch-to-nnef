#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Optional: bootstrap a local venv via uv (uncomment if desired)
# ../bootstrap-uv.sh

# Sample input image used by export_with_batchable.py.
wget -nc https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg

echo "[dynamic_axes] Running example exports..."
python export_albert_fixed.py
python export_with_batchable.py
python cnn_deepspeech_stream.py
echo "[dynamic_axes] Done. See generated .nnef.tgz artifacts in this folder."

