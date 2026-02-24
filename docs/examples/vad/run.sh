#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

python ./export.py

wasm-pack build --target web --out-dir ../../html

rm ../../html/.gitignore ../../html/*.ts
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
