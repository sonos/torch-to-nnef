#!/usr/bin/env bash
# End-to-end: export the two NNEF graphs, then build + run the Rust demo.
# Defaults to the tiny no-download dummy model (proves the full pipeline and
# checks token parity). Set REPO=Hcompany/Holo-3.1-0.8B (and IMAGE=shot.png) to
# run a real checkpoint of any size; DTYPE=f16 loads it in half the memory.
set -euo pipefail
cd "$(dirname "$0")"

source ../bootstrap-rust.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

REPO=${REPO:-}
DTYPE=${DTYPE:-f32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
if [ -z "$REPO" ]; then
  echo "[holo] exporting tiny dummy model (set REPO=... for a real checkpoint)"
  python export.py --dummy --dtype "$DTYPE" --verify "$MAX_NEW_TOKENS" --out ./exp
else
  echo "[holo] exporting $REPO ($DTYPE)"
  hf_pull "$REPO"
  python export.py --repo "$REPO" --dtype "$DTYPE" ${IMAGE:+--image "$IMAGE"} \
    ${PROMPT:+--prompt "$PROMPT"} --out ./exp
fi

(cd holo-rs && cargo run --release -- --dir ../exp --max-new-tokens "$MAX_NEW_TOKENS")
