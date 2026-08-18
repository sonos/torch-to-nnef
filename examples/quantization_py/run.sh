#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

echo "[quantization_py] Exporting toy CNN int8 example..."
python export_toy_cnn_8bit.py
echo "[quantization_py] Done. See generated .nnef.tgz."
