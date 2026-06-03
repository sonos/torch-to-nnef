#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Rust toolchain (inherits examples/rust-toolchain.toml).
source ../bootstrap-rust.sh

# Build the prerequisite NNEF model + sample image from the python example
# (which self-bootstraps its own venv and fetches Grace_Hopper.jpg).
if [ ! -f ../getting_started_py/vit_b_16.nnef.tgz ]; then
  (cd ../getting_started_py && ./run.sh)
fi

echo "[getting_started_rs] Building and running the Rust example..."
cargo run --release -- \
  ../getting_started_py/vit_b_16.nnef.tgz \
  ../getting_started_py/Grace_Hopper.jpg

