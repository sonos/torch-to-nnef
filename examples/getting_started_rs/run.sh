#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[getting_started_rs] Building and running the Rust example..."
echo "Note: ensure you have an exported NNEF at ../getting_started_py/vit_b_16.nnef.tgz"
echo "Usage: cargo run --release -- ../getting_started_py/vit_b_16.nnef.tgz Grace_Hopper.jpg"
cargo run --release -- ../getting_started_py/vit_b_16.nnef.tgz Grace_Hopper.jpg

