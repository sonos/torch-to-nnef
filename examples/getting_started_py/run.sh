#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[getting_started_py] Exporting ViT_B_16 to NNEF..."
python export.py
echo "[getting_started_py] Export complete: vit_b_16.nnef.tgz"
echo "To run inference, place Grace_Hopper.jpg here and run: python run.py"

