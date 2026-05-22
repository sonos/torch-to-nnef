#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[speech_enhancement/dpdfnet] Exporting DPDFNet 2..."
python export.py
echo "[speech_enhancement/dpdfnet] Done. See generated .nnef.tgz."

