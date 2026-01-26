#!/bin/bash

set -e

python3 -m venv .venv
source .venv/bin/activate
mkdir -p assets
(
    cd assets
    wget -qnc https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
    wget -qnc https://raw.githubusercontent.com/mozilla/DeepSpeech/master/data/smoke_test/LDC93S1.wav
)
rm -rf assets/model
pip install -e ../../../[nemo-tract]
export CUDA_VISIBLE_DEVICES=""
# t2n_export_nemo -s nvidia/parakeet-tdt-0.6b-v3 -e assets/model # --tract-specific-path $HOME/SONOS/src/tract/target/release/tract
t2n_export_nemo -s nvidia/parakeet-tdt-0.6b-v3 -e assets/model --tract-specific-path $HOME/dev/sonos/tract/target/release/tract
cd ./src/nemo_asr/ && cargo test --release -- --nocapture && cd ../../
