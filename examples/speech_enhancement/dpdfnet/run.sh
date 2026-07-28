#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../../bootstrap-uv.sh
source .venv/bin/activate

# Clone CEVA's DPDFNet repo + download the checkpoint (idempotent).
bash bootstrap.sh

echo "[speech_enhancement/dpdfnet] Exporting DPDFNet 2..."
python export.py
echo "[speech_enhancement/dpdfnet] Done. See generated .nnef.tgz."
