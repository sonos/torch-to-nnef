#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

REPO=${REPO:-state-spaces/mamba-130m-hf}

echo "[mamba] Exporting external_state shape for repo: $REPO"
(
  cd external_state
  source ../../bootstrap-uv.sh
  source .venv/bin/activate
  hf_pull "$REPO"   # retry the HF download, then export once
  python export.py --repo "$REPO" --out mamba130m.nnef.tgz
)

echo "[mamba] Exporting pulse shape for repo: $REPO"
(
  cd pulse
  source ../../bootstrap-uv.sh
  source .venv/bin/activate
  hf_pull "$REPO"   # retry the HF download, then export once
  python export.py --repo "$REPO" --out mamba130m_pulse.nnef.tgz
)

echo "[mamba] Done. Artifacts in external_state/ and pulse/."
