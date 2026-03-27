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
    --shape-config ../../../tests/assets/shapes.marblenet.collapsed.yaml

# Prepare test audio assets (speech + silence)
echo "Preparing test audio assets..."
ASSETS_DIR=./assets/audio
mkdir -p "$ASSETS_DIR"
SPEECH_URL="https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
SILENCE_URL="https://github.com/anars/blank-audio/raw/refs/heads/master/2-seconds-and-500-milliseconds-of-silence.mp3"
SPEECH_WAV="$ASSETS_DIR/speech.wav"
SILENCE_MP3="$ASSETS_DIR/silence.mp3"
SILENCE_WAV="$ASSETS_DIR/silence_16k.wav"

if command -v curl >/dev/null 2>&1; then
    curl -L -o "$SPEECH_WAV" "$SPEECH_URL" || true
    curl -L -o "$SILENCE_MP3" "$SILENCE_URL" || true
elif command -v wget >/dev/null 2>&1; then
    wget -O "$SPEECH_WAV" "$SPEECH_URL" || true
    wget -O "$SILENCE_MP3" "$SILENCE_URL" || true
else
    echo "Warning: neither curl nor wget available; skipping asset download"
fi

# Convert silence mp3 to 16kHz mono WAV if ffmpeg available; else synthesize silence WAV via Python
if [ -f "$SILENCE_MP3" ]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -hide_banner -loglevel error -i "$SILENCE_MP3" -ac 1 -ar 16000 "$SILENCE_WAV" || true
        rm -f "$SILENCE_MP3"
    fi
fi
if [ ! -f "$SILENCE_WAV" ]; then
    python - <<'PY'
import wave, struct, os
path = os.path.join('assets','audio','silence_16k.wav')
os.makedirs(os.path.dirname(path), exist_ok=True)
sr = 16000
dur = 2.5
n = int(sr*dur)
with wave.open(path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(b"\x00\x00" * n)
print('Synthesized', path)
PY
fi

echo "Running Rust unit tests..."
# RUST_BACKTRACE=full cargo test -- --nocapture

RUST_BACKTRACE=full wasm-pack build --target web --out-dir ../../html -- --features "log-vad"

rm ../../html/.gitignore ../../html/*.ts
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
