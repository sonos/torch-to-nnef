#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

echo "[t2n_extra_custom_op] Exporting custom-op example (my_relu)..."
python export.py
echo "[t2n_extra_custom_op] Done. See my_relu.nnef.tgz."
