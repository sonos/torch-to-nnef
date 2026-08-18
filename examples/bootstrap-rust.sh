#!/usr/bin/env bash
set -euo pipefail

echo "=== Rust bootstrap ==="

# -------------------------------------------------
# 1. Install rustup if missing
# -------------------------------------------------
if ! command -v rustup >/dev/null 2>&1; then
    echo "Installing rustup..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "rustup already installed"
fi

# Ensure cargo env is loaded
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# -------------------------------------------------
# 2. Install toolchain from rust-toolchain file
# -------------------------------------------------
if [ -f "rust-toolchain" ] || [ -f "rust-toolchain.toml" ]; then
    echo "Installing toolchain from rust-toolchain file..."
    rustup install
else
    echo "No rust-toolchain file found, installing stable"
    rustup toolchain install stable
    rustup default stable
fi

# -------------------------------------------------
# 3. Ensure required components are installed
# -------------------------------------------------
echo "Ensuring required components..."

rustup component add rustfmt clippy

# -------------------------------------------------
# 4. Verification
# -------------------------------------------------
echo "=== Rust toolchain ready ==="
rustc --version
cargo --version
