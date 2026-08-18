#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the current example's uv project: ensure uv is installed, then
# `uv sync --locked` to materialise the exact locked environment (.venv) from
# pyproject.toml + uv.lock. Source this from an example's run.sh, then either
# `source .venv/bin/activate` or use `uv run`.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Sourced, not executed, so BASH_SOURCE (not $0) locates the repo layout.
_BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command_exists() { command -v "$1" >/dev/null 2>&1; }

# retry <cmd...>: run a command, retrying with exponential backoff. Use it to
# wrap model-download/export steps -- HuggingFace (and torch hub) frequently
# 429 the shared CI IP pool; a re-run resumes from the populated cache.
retry() {
    local n=1 max="${RETRY_MAX:-5}" delay="${RETRY_DELAY:-15}"
    while true; do
        "$@" && return 0
        if [ "$n" -ge "$max" ]; then
            echo "retry: '$*' still failing after $max attempts" >&2
            return 1
        fi
        echo "retry: attempt $n/$max failed; sleeping ${delay}s before retry..." >&2
        sleep "$delay"
        n=$((n + 1))
        delay=$((delay * 2))
    done
}

ensure_uv() {
    command_exists uv && return 0
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command_exists uv || { echo "uv installation failed"; exit 1; }
}

# hf_pull <repo>: pre-download a HuggingFace repo into the local cache so the
# subsequent export reads from cache. Requires the venv active (huggingface_hub
# installed).
#
# The retry/backoff, the permanent-vs-transient classification and the
# Xet-then-plain-LFS fallback all live in scripts/hf_pull.py, shared with the
# CI prefetch steps. Keeping a second copy here is what let the two drift: the
# fallback was added to this file alone, and the next Xet outage took the
# unshared callers down.
hf_pull() {
    python "${_BOOTSTRAP_DIR}/../scripts/hf_pull.py" "$1"
}

main() {
    ensure_uv
    if [ ! -f pyproject.toml ]; then
        echo "no pyproject.toml in $(pwd); this example is not a uv project"
        exit 1
    fi
    echo "Syncing locked environment (uv sync --locked)..."
    uv sync --locked --python "$PYTHON_VERSION"
    echo "Environment ready (.venv). Activate with: . .venv/bin/activate"
}

main "$@"
