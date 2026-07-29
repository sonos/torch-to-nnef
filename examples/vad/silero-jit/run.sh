#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../../bootstrap-uv.sh
source .venv/bin/activate

echo "[vad/silero-jit] Exporting silero_vad.jit to NNEF..."
python export.py
echo "[vad/silero-jit] Done. See generated .nnef.tgz."
