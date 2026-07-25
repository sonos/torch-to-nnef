#!/usr/bin/env bash
# End-to-end: export the two NNEF graphs, then build + run the Rust demo.
# Defaults to the tiny no-download dummy model (proves the full pipeline and
# checks token parity). Set REPO=Hcompany/Holo-3.1-0.8B (and IMAGE=shot.png)
# to run a real checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

source ../bootstrap-rust.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

REPO=${REPO:-}
if [ -z "$REPO" ]; then
  echo "[holo] exporting tiny dummy model (set REPO=... for a real checkpoint)"
  python export.py --dummy --verify 12 --out ./exp
else
  echo "[holo] exporting $REPO"
  hf_pull "$REPO"
  python export.py --repo "$REPO" ${IMAGE:+--image "$IMAGE"} \
    ${PROMPT:+--prompt "$PROMPT"} --out ./exp
fi

(cd holo-rs && cargo run --release -- --dir ../exp --max-new-tokens 16)
