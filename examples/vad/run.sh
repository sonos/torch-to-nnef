#!/bin/bash
set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

# Silence unexpected_cfg warnings from downstream crates referring to
# `#[cfg(feature = "inventory-registry")]` used inside macros.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--check-cfg=cfg(feature,values(\"inventory-registry\"))"

# Export preprocessor + encoder NNEF graphs from the HF funasr/fsmn-vad weights.
rm -rf ./model
python ./py/export.py --out-dir ./model

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

if [ -f "$SILENCE_MP3" ]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -hide_banner -loglevel error -i "$SILENCE_MP3" -ac 1 -ar 16000 "$SILENCE_WAV" || true
        rm -f "$SILENCE_MP3"
    fi
fi
if [ ! -f "$SILENCE_WAV" ]; then
    python - <<'PY'
import wave, os
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

RUST_BACKTRACE=full wasm-pack build --target web --out-dir ../../html -- --features "log-vad"

rm -f ../../html/.gitignore ../../html/*.ts 2>/dev/null || true
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete 2>/dev/null || true
