#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[vad/silero-jit] Exporting silero_vad.jit to NNEF..."
python export.py
echo "[vad/silero-jit] Done. See generated .nnef.tgz."

