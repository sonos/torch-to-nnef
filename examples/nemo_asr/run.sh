#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate
mkdir -p assets
(
    cd assets
    wget -qN https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
    wget -qN https://raw.githubusercontent.com/mozilla/DeepSpeech/master/data/smoke_test/LDC93S1.wav -O data_smoke_test_LDC93S1.wav
)
rm -rf assets/model
TRACT_VERSION="0.23.0-dev.2"
python -c "from torch_to_nnef.inference_target.tract import TractNNEF; TractNNEF('$TRACT_VERSION'); print('TractNNEF $TRACT_VERSION is available')"
# retry (parakeet is fetched from HF; 429s on shared CI IPs), cleaning the
# output dir each attempt since t2n refuses to export into an existing dir.
# tolerance=close: the 600M-param conformer has ~24 f32-noise outliers out of
# 201M values (~2e-4); "approximate" allows 0 outliers, "close" tolerates them.
retry bash -c "rm -rf assets/model && t2n_export_nemo -s nvidia/parakeet-tdt-0.6b-v3 -e assets/model --tract-specific-path $HOME/.cache/svc/tract/$TRACT_VERSION/tract --tract-check-io-tolerance close"
cd ./src/nemo_asr/ && cargo test --release -- --nocapture && cd ../../
