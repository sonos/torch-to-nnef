#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[speech_enhancement/deepfilternet] Exporting DeepFilterNet 3..."
python export.py
echo "[speech_enhancement/deepfilternet] Done. See generated .nnef.tgz."

