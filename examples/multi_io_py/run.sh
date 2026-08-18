#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

echo "[multi_io_py] Exporting ALBERT multi-IO example..."
hf_pull "albert-base-v2"   # retry the HF download, then export once
python export_albert.py
echo "[multi_io_py] Done. See albert.nnef.tgz."
