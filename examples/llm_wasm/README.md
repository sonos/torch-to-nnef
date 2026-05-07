# Small LLM in wasm

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HuggingFaceTB%2FSmolLM--135M-yellow)](https://huggingface.co/HuggingFaceTB/SmolLM-135M) [![demo](https://img.shields.io/badge/live-demo-brightgreen)](https://sonos.github.io/torch-to-nnef/latest/html/demo_poem_generator.html)

Companion to the [LLM export tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/5_llm/). Exports HuggingFaceTB's SmolLM-135M (135M-parameter decoder-only LLM) to NNEF via `t2n_export_llm_to_tract`, then compiles a tract-backed Rust crate to WebAssembly for in-browser text generation.

## Run

```bash
cd examples/llm_wasm
./run.sh
```

The `run.sh` script:
1. Sets up `.venv` + Rust toolchain via the bootstrap helpers
2. Runs `t2n_export_llm_to_tract -s HuggingFaceTB/SmolLM-135M ...` to produce the NNEF archive bundled with tokenizer / config
3. Builds the Rust crate to wasm with `wasm-pack` (uses tract's `causal_llm` runtime)
4. Drops the wasm + JS glue into `docs/html/` for the live demo

Live demo: [https://sonos.github.io/torch-to-nnef/latest/html/demo_poem_generator.html](https://sonos.github.io/torch-to-nnef/latest/html/demo_poem_generator.html).

The tract dependencies are pinned to the `feat/wasm-llm` branch in `Cargo.toml` because the wasm causal-LLM runtime is still maturing in tract main.
