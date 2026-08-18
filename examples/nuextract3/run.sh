#!/usr/bin/env bash
# End-to-end: export the two NNEF graphs, then build + run the Rust demo.
# Defaults to the tiny no-download dummy model (proves the full pipeline and
# checks token parity). Set REPO=numind/NuExtract3 IMAGE=receipt.png to run a
# real image-to-Markdown export; DTYPE=f16 loads it in half the memory.
set -euo pipefail
cd "$(dirname "$0")"

source ../bootstrap-rust.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

REPO=${REPO:-}
DTYPE=${DTYPE:-f32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
if [ -z "$REPO" ]; then
  echo "[nuextract3] exporting tiny dummy model (set REPO=... for a real checkpoint)"
  python export.py --dummy --dtype "$DTYPE" --verify 12 --out ./exp
else
  echo "[nuextract3] exporting $REPO ($DTYPE)"
  hf_pull "$REPO"
  python export.py --repo "$REPO" --dtype "$DTYPE" ${IMAGE:+--image "$IMAGE"} \
    ${PROMPT:+--prompt "$PROMPT"} --out ./exp
fi

(cd nuextract3-rs && cargo run --release -- \
  --dir ../exp --max-new-tokens "$MAX_NEW_TOKENS")
