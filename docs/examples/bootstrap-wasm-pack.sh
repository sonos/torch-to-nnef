if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
else
    echo "wasm-pack already installed."
fi
