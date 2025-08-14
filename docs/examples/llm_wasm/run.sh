# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# t2n_export_llm_to_tract -s "HuggingFaceTB/SmolLM-135M" -e ./dump_model -c min_max_q4_0_all --dump-with-tokenizer-and-conf

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --out-dir ../../html
rm ../../html/.gitignore ../../html/*.ts
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
