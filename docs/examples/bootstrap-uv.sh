#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.11.8}"
VENV_DIR=".venv"

# -----------------------------
# Utilities
# -----------------------------
command_exists() { command -v "$1" >/dev/null 2>&1; }

install_uv() {
    echo "Installing uv..."
    # Use pipefail to surface curl or installer failures
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command_exists uv; then
        echo "uv installation failed"
        exit 1
    fi
}

ensure_uv() { command_exists uv || install_uv; }

# -----------------------------
# Ensure Python toolchain
# -----------------------------
ensure_python() {
    echo "Installing Python $PYTHON_VERSION (if needed)..."
    uv python install "$PYTHON_VERSION"
}

# -----------------------------
# Create venv (idempotent)
# -----------------------------
create_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    fi
}

# -----------------------------
# Install requirements
# -----------------------------
install_requirements() {
    if [ ! -f requirements.txt ]; then
        echo "requirements.txt not found"
        exit 1
    fi

    echo "Installing dependencies from requirements.txt..."
    uv pip install \
        --python "$VENV_DIR/bin/python" \
        --requirement requirements.txt
}

# -----------------------------
# Main
# -----------------------------
main() {
    ensure_uv
    ensure_python
    create_venv
    install_requirements

    echo ""
    echo "Environment ready."
    echo "Activate with:"
    echo "  . $VENV_DIR/bin/activate"
}

main "$@"
