#!/bin/bash
set -ex

source ../bootstrap-uv.sh
source .venv/bin/activate

python export.py --out ./fsmn_vad.nnef.tgz
