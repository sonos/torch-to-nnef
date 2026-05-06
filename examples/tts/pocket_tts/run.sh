#!/usr/bin/env bash
#
# End-to-end Pocket-TTS demo: export every NNEF graph, bake the voice
# prompt, build the Rust CLI, and synthesise a WAV for the fixed text
# "hello I am a text to speech voice".
#
# Usage:
#     ./run.sh
#
# What this does, in order:
#   1. Bootstrap a uv-managed venv + Rust toolchain (idempotent).
#   2. Remove any prior ``cli/out.wav`` so each run starts clean.
#   3. Export the four NNEF graphs (flow_net, flow_lm_init, flow_lm_step,
#      decoder) and bake ``voices/alba.dat`` -- all driven through
#      the same uv-managed Python environment.
#   4. ``cargo build --release`` the Rust CLI.
#   5. Invoke the CLI to write ``cli/out.wav``.
#
# Note on weights: this run.sh wires the *mini* random-weights config end
# to end (~50k params per stage). The pipeline is plumbing-correct -- the
# WAV is well-formed 24 kHz f32 audio -- but the audio itself is noise
# because the weights are random. Switching to the real Pocket-TTS
# checkpoint requires extending each export script with a ``--full`` path
# (load ``TTSModel.load_model()``, harvest its submodules, re-export at
# production dims) and adding an export of ``mimi.decoder_transformer``,
# the projection between FlowLM latents and the SEANet decoder. Tracked
# as a follow-up.
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
rm -f cli/out.wav
mkdir -p cli/models cli/voices

TRACT_VERSION="${TRACT_VERSION:-0.23.0-dev.5}"
COMMON_EXPORT_FLAGS=(--mini --skip-io-check --tract-version "$TRACT_VERSION")

# 3. Export the four NNEF graphs + bake the voice prompt ---------------------
echo "==> exporting flow_net"
python flow_net.py "${COMMON_EXPORT_FLAGS[@]}" \
    --out cli/models/flow_net.nnef.tgz

echo "==> exporting flow_lm (init + step)"
python flow_lm.py "${COMMON_EXPORT_FLAGS[@]}" \
    --voice-frames 4 \
    --out-init cli/models/flow_lm_init.nnef.tgz \
    --out-step cli/models/flow_lm_step.nnef.tgz

echo "==> exporting decoder"
python decoder.py "${COMMON_EXPORT_FLAGS[@]}" \
    --out cli/models/decoder.nnef.tgz

echo "==> baking voice prompt (alba)"
python bake_voice.py --mini --out cli/voices/alba.dat

# 4. Build Rust CLI in release ----------------------------------------------
echo "==> building Rust CLI"
( cd cli && cargo build --release )

# 5. Run the binary on the requested text -----------------------------------
TEXT="hello I am a text to speech voice"
# The mini conditioner has random embeddings + ``n_bins=100`` vocabulary,
# so we feed a deterministic placeholder token sequence sized to the
# decoder's traced latent-frame budget instead of running the real
# SentencePiece tokenizer (which would hand back token IDs >= 100 the
# mini embedding cannot look up). Real-weights mode would replace this
# with ``--text "$TEXT" --tokenizer cli/tokenizer.model``.
# The mini ``flow_lm_init`` graph is traced with ``--text-tokens 4`` so
# the token sequence has to be exactly 4 entries to match the static
# shape; we slice the bytes of $TEXT down to 4 here.
TOKENS="$(python -c '
text = "'"$TEXT"'"
ids = [(b % 100) + 1 for b in text.encode("utf-8")[:4]]
print(",".join(str(i) for i in ids))
')"

echo "==> synthesising: $TEXT"
echo "    tokens: $TOKENS"
./cli/target/release/pocket-tts-tract \
    --models cli/models \
    --voice cli/voices/alba.dat \
    --tokens "$TOKENS" \
    --max-frames 8 \
    --eos-threshold 1e9 \
    --out cli/out.wav

echo
echo "Wrote $EXAMPLE_DIR/cli/out.wav"
ls -la cli/out.wav
