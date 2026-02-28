#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

TRACT_VERSION="0.23.0-dev.2"
python -c "from torch_to_nnef.inference_target.tract import TractNNEF; TractNNEF('$TRACT_VERSION'); print('TractNNEF $TRACT_VERSION is available')"
TRACT_PATH=$HOME"/.cache/svc/tract/"$TRACT_VERSION"/tract"

# Silence unexpected_cfg warnings from downstream crates referring to
# `#[cfg(feature = "inventory-registry")]` used inside macros.
# This registers the cfg value without enabling the feature.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--check-cfg=cfg(feature,values(\"inventory-registry\"))"

rm -rf ./model
t2n_export_nemo -s "vad_multilingual_marblenet" \
    -e "./model" \
    --tract-specific-path $TRACT_PATH \
    --collapse-batch-dim
(
    cd ./model
    $TRACT_PATH ./encoder.nnef.tgz \
        --nnef-tract-core \
        --nnef-tract-pulse \
        --pulse AUDIO_SIGNAL__TIME=4 \
        dump \
        --nnef ./encoder.pulsed.nnef.tgz
)

RUST_BACKTRACE=full wasm-pack build --target web --out-dir ../../html -- --features "log-vad"

rm ../../html/.gitignore ../../html/*.ts
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
