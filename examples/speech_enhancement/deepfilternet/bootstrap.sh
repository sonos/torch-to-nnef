#!/usr/bin/env bash
# Clone grazder's DeepFilterNet fork so we can pick up
# `torchDF/torch_df_streaming_minimal.py` -- the pure-torch reimplementation
# of the full DFN pipeline (waveform in / waveform out). The torchDF
# subdirectory is *not* pip-installable on its own (its pyproject.toml
# references source files outside the subdir), so we put it on the
# Python path at import time instead.
#
# Usage:
#   ./bootstrap.sh
#
# Idempotent; re-running just runs `git pull` on the existing checkout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="${HERE}/_torchDF_clone"
REPO_URL="https://github.com/grazder/DeepFilterNet.git"
BRANCH="torchDF_main"

if [[ -d "${CLONE_DIR}/.git" ]]; then
    echo "==> updating existing clone at ${CLONE_DIR}"
    git -C "${CLONE_DIR}" fetch --depth=1 origin "${BRANCH}"
    git -C "${CLONE_DIR}" reset --hard "origin/${BRANCH}"
else
    echo "==> cloning ${REPO_URL} @ ${BRANCH}"
    git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${CLONE_DIR}"
fi

echo
echo "torchDF clone is at: ${CLONE_DIR}/torchDF"
echo "export.py / export_stft_variant.py add that path to sys.path automatically."
