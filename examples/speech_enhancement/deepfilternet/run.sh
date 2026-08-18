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
# Note: export_onnx_baseline.py (grazder's reference ONNX export) is intentionally
# NOT run here -- torch's legacy TorchScript ONNX exporter SIGFPEs on the DFN3 GRU
# on Linux. It is a reference baseline, not a torch-to-nnef export; run manually.
echo "[speech_enhancement/deepfilternet] Done. See generated .nnef.tgz artifacts."
