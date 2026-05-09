#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

wget -nc https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg
python ./export_with_batchable.py
wasm-pack build --target web --out-dir ../../docs/html
rm ../../docs/html/.gitignore ../../docs/html/*.ts
find ../../docs/html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
