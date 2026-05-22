#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[multi_io_py] Exporting ALBERT multi-IO example..."
python export_albert.py
echo "[multi_io_py] Done. See albert.nnef.tgz."

