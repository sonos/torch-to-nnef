#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

t2n_export_llm_to_tract -s "HuggingFaceTB/SmolLM-135M" -e ./dump_model --dump-with-tokenizer-and-conf

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --out-dir ../../docs/html
rm ../../docs/html/.gitignore ../../docs/html/*.ts
find ../../docs/html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
