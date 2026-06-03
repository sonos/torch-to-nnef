#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

# Sample input image: real if reachable, else a synthetic placeholder
# (export only needs a valid JPEG; Wikimedia 429s the shared CI IP pool).
if [ ! -f Grace_Hopper.jpg ]; then
  wget -q --tries=3 --timeout=20 -O Grace_Hopper.jpg \
    --user-agent="torch-to-nnef-example/1.0 (+https://github.com/sonos/torch-to-nnef)" \
    https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg \
  || python -c "from PIL import Image; Image.new('RGB', (640, 480), (127, 127, 127)).save('Grace_Hopper.jpg')"
fi
python ./export_with_batchable.py
wasm-pack build --target web --out-dir ../../docs/html
rm ../../docs/html/.gitignore ../../docs/html/*.ts
find ../../docs/html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
