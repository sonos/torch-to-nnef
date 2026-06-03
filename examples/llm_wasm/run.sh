#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

# SmolLM is fetched from HF (429s on shared CI IPs); retry, cleaning the output
# dir each attempt since t2n refuses to export into an existing directory.
# (deps are pinned to transformers<5 + torch 2.7 in pyproject.toml, where the
#  export passes parity at default tolerance -- no dtype/tolerance workarounds.)
retry bash -c 'rm -rf ./dump_model && t2n_export_llm_to_tract -s "HuggingFaceTB/SmolLM-135M" -e ./dump_model --dump-with-tokenizer-and-conf'

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --out-dir ../../docs/html
rm ../../docs/html/.gitignore ../../docs/html/*.ts
find ../../docs/html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
