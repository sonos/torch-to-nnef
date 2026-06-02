#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Sample input image used by export.py (IO verification) and run.py.
wget -nc https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg

echo "[getting_started_py] Exporting ViT_B_16 to NNEF..."
python export.py
echo "[getting_started_py] Export complete: vit_b_16.nnef.tgz"
echo "To run inference, place Grace_Hopper.jpg here and run: python run.py"

