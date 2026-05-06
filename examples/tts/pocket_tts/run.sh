#!/usr/bin/env bash
#
# End-to-end Pocket-TTS demo: export every NNEF graph, bake the voice
# prompt, build the Rust CLI, and synthesise a WAV for the fixed text
# "hello I am a text to speech voice".
#
# Usage:
#     ./run.sh                # default --mini path (random weights, noise out)
#     MODE=full ./run.sh      # real Pocket-TTS weights (HF download)
#     MODE=full GPU=1 ./run.sh  # full + tract Metal runtime (macOS only)
#
# Modes:
#   * ``--mini`` (default) -- random-weights config (~50k params per stage)
#     with synthesised tokens. Pipeline-correct WAV but acoustically noise.
#   * ``MODE=full``        -- real ~89M-param FlowLM + real ~20M-param Mimi
#     audio decoder (quantizer + upsample + decoder_transformer + SEANet),
#     all four graphs running through tract. No Python in the inference
#     path: the produced binary + ``cli/models/`` + ``cli/voices/`` +
#     ``cli/tokenizer.model`` is a self-contained shippable.
#
# The CLI prints RTFx (audio_seconds / wall_seconds) on every run.
set -euo pipefail

cd "$(dirname "$0")"
EXAMPLE_DIR="$(pwd)"

# 1. Bootstrap toolchains -----------------------------------------------------
source ../../bootstrap-uv.sh
source ../../bootstrap-rust.sh
# shellcheck disable=SC1091
source .venv/bin/activate

# Quiet a stray `inventory-registry` cfg warning from a transitive crate.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--check-cfg=cfg(feature,values(\"inventory-registry\"))"

# 2. Clean prior artefacts ---------------------------------------------------
rm -f cli/out.wav cli/models/decoder.nnef.tgz cli/models/mimi_decode.nnef.tgz
mkdir -p cli/models cli/voices

TRACT_VERSION="${TRACT_VERSION:-0.23.0-dev.5}"
MODE="${MODE:-mini}"
GPU="${GPU:-0}"
case "$MODE" in
    mini) WEIGHTS_FLAG=--mini ;;
    full) WEIGHTS_FLAG=--full ;;
    *) echo "MODE must be 'mini' or 'full', got: $MODE" >&2; exit 1 ;;
esac
COMMON_EXPORT_FLAGS=("$WEIGHTS_FLAG" --skip-io-check --tract-version "$TRACT_VERSION")
GPU_CLI_FLAGS=()
if [ "$GPU" = 1 ]; then
    GPU_CLI_FLAGS=(--gpu)
fi

# 3. Export the four NNEF graphs + bake the voice prompt ---------------------
TEXT="hello I am a text to speech voice"

# In --full mode we need to know the real text-token count + voice-prefix
# length up front so flow_lm_init can be traced at those exact sizes (the
# real model ships symbols tract can't easily relate, so we go static).
if [ "$MODE" = full ]; then
    echo "==> extracting tokenizer + measuring shapes"
    python extract_tokenizer.py --out cli/tokenizer.model
    python bake_voice.py --full --out cli/voices/alba.dat
    SHAPE_DATA="$(python -c '
import sentencepiece as sp
import nnef
sp_proc = sp.SentencePieceProcessor("cli/tokenizer.model")
ids = sp_proc.encode("'"$TEXT"'")
with open("cli/voices/alba.dat", "rb") as f:
    voice = nnef.read_tensor(f)
print(f"{len(ids)},{voice.shape[3]}")
')"
    T_TEXT="${SHAPE_DATA%,*}"
    T_VOICE="${SHAPE_DATA#*,}"
    echo "    T_TEXT=$T_TEXT  T_VOICE=$T_VOICE"
else
    T_TEXT=4
    T_VOICE=4
fi

echo "==> exporting flow_net"
python flow_net.py "${COMMON_EXPORT_FLAGS[@]}" \
    --out cli/models/flow_net.nnef.tgz

echo "==> exporting flow_lm (init + step)"
python flow_lm.py "${COMMON_EXPORT_FLAGS[@]}" \
    --text-tokens "$T_TEXT" \
    --voice-frames "$T_VOICE" \
    --out-init cli/models/flow_lm_init.nnef.tgz \
    --out-step cli/models/flow_lm_step.nnef.tgz

if [ "$MODE" = mini ]; then
    echo "==> exporting decoder (mini SEANet only)"
    python decoder.py "${COMMON_EXPORT_FLAGS[@]}" \
        --out cli/models/decoder.nnef.tgz
    echo "==> baking voice prompt (mini)"
    python bake_voice.py --mini --out cli/voices/alba.dat
else
    # ``mimi_decode`` declares ``T_LATENT`` as a dynamic axis, so the
    # exported graph runs at any latent frame count. The trace shape below
    # is just an example -- the autoregressive loop in the CLI stops on
    # real EOS, not at a fixed frame budget.
    echo "==> exporting mimi_decode (dynamic T_LATENT)"
    python mimi_decode.py "${COMMON_EXPORT_FLAGS[@]}" \
        --latent-frames 50 \
        --out cli/models/mimi_decode.nnef.tgz
fi

# 4. Build Rust CLI in release ----------------------------------------------
echo "==> building Rust CLI"
( cd cli && cargo build --release )

# 5. Run the binary on the requested text -----------------------------------
TEXT="hello I am a text to speech voice"

echo "==> synthesising: $TEXT"
if [ "$MODE" = mini ]; then
    # mini conditioner has random embeddings + ``n_bins=100`` vocab and
    # ``flow_lm_init`` is traced with ``--text-tokens 4``, so feed exactly
    # 4 placeholder IDs derived from the requested text bytes.
    TOKENS="$(python -c '
text = "'"$TEXT"'"
ids = [(b % 100) + 1 for b in text.encode("utf-8")[:4]]
print(",".join(str(i) for i in ids))
')"
    echo "    tokens (mini placeholder): $TOKENS"
    ./cli/target/release/pocket-tts-tract \
        --models cli/models \
        --voice cli/voices/alba.dat \
        --tokens "$TOKENS" \
        --max-frames 8 \
        --eos-threshold 1e9 \
        "${GPU_CLI_FLAGS[@]+"${GPU_CLI_FLAGS[@]}"}" \
        --out cli/out.wav
else
    echo "    tokenizing via cli/tokenizer.model (full path: tract end-to-end, no Python)"
    ./cli/target/release/pocket-tts-tract \
        --models cli/models \
        --voice cli/voices/alba.dat \
        --tokenizer cli/tokenizer.model \
        --text "$TEXT" \
        --ldim 32 \
        --max-frames 256 \
        "${GPU_CLI_FLAGS[@]+"${GPU_CLI_FLAGS[@]}"}" \
        --out cli/out.wav
fi

echo
echo "Wrote $EXAMPLE_DIR/cli/out.wav"
ls -la cli/out.wav
