#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Sync the locked uv environment (creates .venv).
source ../bootstrap-uv.sh
source .venv/bin/activate

# Sample input image for export.py / run.py. Fetch the real one when possible,
# else synthesize a placeholder: the export only needs a valid JPEG (content is
# not asserted), and Wikimedia 429s the shared CI IP pool.
if [ ! -f Grace_Hopper.jpg ]; then
  wget -q --tries=3 --timeout=20 -O Grace_Hopper.jpg \
    --user-agent="torch-to-nnef-example/1.0 (+https://github.com/sonos/torch-to-nnef)" \
    https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg \
  || python -c "from PIL import Image; Image.new('RGB', (640, 480), (127, 127, 127)).save('Grace_Hopper.jpg')"
fi

echo "[getting_started_py] Exporting ViT_B_16 to NNEF..."
python export.py
echo "[getting_started_py] Export complete: vit_b_16.nnef.tgz"
echo "To run inference, place Grace_Hopper.jpg here and run: python run.py"
