#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set up an isolated venv with pinned deps (idempotent).
source ../bootstrap-uv.sh
source .venv/bin/activate

# Sample input image for export_with_batchable.py. Fetch the real one when
# possible, else synthesize a placeholder: the export only needs a valid JPEG
# (content is not asserted), and Wikimedia 429s the shared CI IP pool.
if [ ! -f Grace_Hopper.jpg ]; then
  wget -q --tries=3 --timeout=20 -O Grace_Hopper.jpg \
    --user-agent="torch-to-nnef-example/1.0 (+https://github.com/sonos/torch-to-nnef)" \
    https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg \
  || python -c "from PIL import Image; Image.new('RGB', (640, 480), (127, 127, 127)).save('Grace_Hopper.jpg')"
fi

echo "[dynamic_axes] Running example exports..."
# retry: ALBERT weights are fetched from HF (429s on shared CI IPs).
retry python export_albert_fixed.py
retry python export_with_batchable.py
python cnn_deepspeech_stream.py
echo "[dynamic_axes] Done. See generated .nnef.tgz artifacts in this folder."

