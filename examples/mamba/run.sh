#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

REPO=${REPO:-state-spaces/mamba-130m-hf}

echo "[mamba] Exporting external_state shape for repo: $REPO"
pushd external_state >/dev/null
python export.py --repo "$REPO" --out mamba130m.nnef.tgz
popd >/dev/null

echo "[mamba] Exporting pulse shape for repo: $REPO"
pushd pulse >/dev/null
python export.py --repo "$REPO" --out mamba130m_pulse.nnef.tgz
popd >/dev/null

echo "[mamba] Done. Artifacts in external_state/ and pulse/."

