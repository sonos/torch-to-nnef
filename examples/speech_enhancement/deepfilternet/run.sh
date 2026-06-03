#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../../bootstrap-uv.sh
source .venv/bin/activate

# Clone the upstream model fork + download the DFN3 checkpoint (idempotent).
bash bootstrap.sh

echo "[speech_enhancement/deepfilternet] Exporting DeepFilterNet 3..."
python export.py
echo "[speech_enhancement/deepfilternet] Done. See generated .nnef.tgz."

