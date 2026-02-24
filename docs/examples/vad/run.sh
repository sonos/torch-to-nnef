#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

TRACT_VERSION="0.23.0-dev.2"
python -c "from torch_to_nnef.inference_target.tract import TractNNEF; TractNNEF('$TRACT_VERSION'); print('TractNNEF $TRACT_VERSION is available')"
TRACT_PATH=$HOME"/.cache/svc/tract/"$TRACT_VERSION"/tract"

# ROOT_DIR="$(pwd)"
# BIN_NAME="tract"
# BIN_PATH="$ROOT_DIR/bin/$BIN_NAME"
#
# if [ ! -x "$BIN_PATH" ]; then
#     echo "Installing $BIN_NAME..."
#     cargo install \
#         --git https://github.com/sonos/tract.git \
#         --rev 4194300 \
#         --root "$ROOT_DIR" \
#         --locked tract
# else
#     echo "$BIN_NAME already installed."
# fi
# TRACT_PATH=$BIN_PATH

python ./export.py -o . --tract-path $TRACT_PATH

wasm-pack build --target web --out-dir ../../html

rm ../../html/.gitignore ../../html/*.ts
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
