#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the current example's uv project: ensure uv is installed, then
# `uv sync --locked` to materialise the exact locked environment (.venv) from
# pyproject.toml + uv.lock. Source this from an example's run.sh, then either
# `source .venv/bin/activate` or use `uv run`.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

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

_hf_snapshot() {
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$1"
}

# hf_pull <repo>: pre-download a HuggingFace repo into the local cache so the
# subsequent export reads from cache. Requires the venv active (huggingface_hub
# installed).
#
# Prefer the default download path (Xet when available), so a healthy Xet is
# used and stays exercised. Only on failure, retry on the plain-LFS path: Xet's
# `xet-read-token` endpoint intermittently returns 404 (it is not on the HF
# status page, so there is no health signal, and a 404 is not auto-retried), so
# an outage degrades to LFS instead of failing CI while a working Xet keeps the
# fast path. The `retry` wraps the LFS fallback so transient 429s recover too.
hf_pull() {
    if _hf_snapshot "$1"; then
        return 0
    fi
    echo "hf_pull: default (Xet) download failed for '$1'; " \
        "falling back to the plain-LFS path" >&2
    (
        export HF_HUB_DISABLE_XET=1
        retry _hf_snapshot "$1"
    )
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
