#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../../bootstrap-uv.sh
source .venv/bin/activate

# Clone the upstream model fork + download the DFN3 checkpoint (idempotent).
bash bootstrap.sh

echo "[speech_enhancement/deepfilternet] Exporting DeepFilterNet 3 (NNEF)..."
python export.py
echo "[speech_enhancement/deepfilternet] Exporting STFT variant (native irfft)..."
python export_stft_variant.py
echo "[speech_enhancement/deepfilternet] Exporting ONNX reference baseline..."
python export_onnx_baseline.py
echo "[speech_enhancement/deepfilternet] Done. See generated .nnef.tgz / .onnx artifacts."

