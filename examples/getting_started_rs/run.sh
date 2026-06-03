#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Rust toolchain (inherits examples/rust-toolchain.toml).
source ../bootstrap-rust.sh

# main.rs reads ./vit_b_16.nnef.tgz and ./Grace_Hopper.jpg from the cwd. Build
# them via the python example (self-bootstrapping) and copy them in.
if [ ! -f ../getting_started_py/vit_b_16.nnef.tgz ]; then
  (cd ../getting_started_py && ./run.sh)
fi
cp -f ../getting_started_py/vit_b_16.nnef.tgz ./vit_b_16.nnef.tgz
cp -f ../getting_started_py/Grace_Hopper.jpg ./Grace_Hopper.jpg

echo "[getting_started_rs] Building and running the Rust example..."
cargo run --release
