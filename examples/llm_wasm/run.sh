#!/bin/bash

set -ex

source ../bootstrap-rust.sh
source ../bootstrap-wasm-pack.sh
source ../bootstrap-uv.sh
source .venv/bin/activate

# --force-module-dtype f32: SmolLM ships bf16 weights which mismatch the
#   float rotary/SDPA tensors during export ("q/k/v dtype" RuntimeError).
# --tract-check-io-tolerance close: f32 export has a single ~0.35% logit
#   outlier vs torch (1/49152); "approximate" allows 0 outliers, "close" passes.
# retry: SmolLM is fetched from HF, which 429s the shared CI IP pool.
retry t2n_export_llm_to_tract -s "HuggingFaceTB/SmolLM-135M" -e ./dump_model --dump-with-tokenizer-and-conf --force-module-dtype f32 --tract-check-io-tolerance close

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --out-dir ../../docs/html
rm ../../docs/html/.gitignore ../../docs/html/*.ts
find ../../docs/html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
