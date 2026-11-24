# Inkling 2 Mini text export

Export the text decoder of
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
with the standard LLM command. Inkling support requires a recent Transformers
checkout; the integration is tested against commit
`d8f10a82213a3a9db48385aba3a4b57ab535de50` (`5.24.0.dev0`).

From the repository root:

```sh
uv sync --extra llm
uv pip install 'transformers @ git+https://github.com/huggingface/transformers.git@d8f10a82213a3a9db48385aba3a4b57ab535de50'
.venv/bin/t2n_export_llm_to_tract \
  -s thinkingmachines/Inkling-2-Mini \
  -e /tmp/inkling2-export \
  --force-module-dtype f32 \
  --attn-implementation eager \
  --tract-specific-path /path/to/tract
```

Use `--local-dir /path/to/checkpoint` instead of `-s` for a local snapshot.
Upstream Transformers converts the original mixed NVFP4/MXFP8 expert weights
to dense tensors on load. This export does **not** preserve native low-bit
storage: float32 weights alone require roughly 106 GB, with additional memory
needed for loading and tracing. No full-checkpoint export has been validated.

For Q4_0 export, use tract 0.23.10 and add
`-c min_max_q4_0_with_embeddings_and_experts --tract-check-io-tolerance very`
to the command above. Quantized matmuls also round activations in tract,
so the default float32 IO tolerance may be too strict. The exporter
automatically retries unsupported fake-tensor boundaries with concrete weight
expansion; no legacy tracing flag is required. This retry can increase tracing
memory and computation, and inputs and registered buffers are snapshotted
before tracing to preserve recurrent state if a retry is needed.
The registry quantizes eligible linear/embedding weights and the shared
selected-expert adapter's weight tables. Routers remain floating point.
Expert tables are gathered and dequantized before floating-point matmuls;
this reduces stored weight size but does not provide a fused low-bit MoE
kernel. Loading still requires upstream dequantization, and selected expert
matrices can consume substantial temporary memory. Other quantization
registry entries are not covered by the Inkling regression tests.

The decoder supports unpadded batch-one text, chunked prefill, and single-token
decoding. Image inputs and the speculative draft model are not included.
Inkling's original loop schedule and message gates are retained. The 24 physical
layers have 48 independent sets of state: the first 24 follow physical layer
order, followed by layers 6-17 for each additional pass.

For every cache index, feed these outputs back into their matching inputs:

| State | Initial shape | Type |
| --- | --- | --- |
| `in_cache_key_N` / `out_cache_key_N` | `[1, 8, 0, 128]` | model dtype |
| `in_cache_value_N` / `out_cache_value_N` | `[1, 8, 0, 128]` | model dtype |
| `in_cache_conv0_N` / `out_cache_conv0_N` | `[1, 1024, 3]` | float32 |
| `in_cache_conv1_N` / `out_cache_conv1_N` | `[1, 1024, 3]` | float32 |
| `in_cache_conv2_N` / `out_cache_conv2_N` | `[1, 2304, 3]` | float32 |
| `in_cache_conv3_N` / `out_cache_conv3_N` | `[1, 2304, 3]` | float32 |

Initialize all state to zero. Both full and sliding attention retain all KV
history on the dynamic `P` axis; the sliding mask still limits visibility to
the configured window. Convolution history has fixed width. Expert selection
uses tensor gathers and matmuls to preserve Inkling's softcapped SwiGLU and
bias-corrected routing; it does not use `tract_moe_ffn` and can require large
temporary tensors during prefill. Start with short chunks.

Run the small random-model regression suite without downloading model weights:

```sh
TRACT_BIN=/path/to/tract .venv/bin/python -m pytest \
  packages/llm/tests/test_inkling2_handler.py
```

Tests compare streaming logits with the unmodified upstream model, exercise
different sequence lengths after tracing, and check NNEF execution in tract.
The float32 tiny model was checked with tract `0.23.5-pre` and PyTorch `2.14.1`.
Float32 and Q4_0 regressions also run with tract `0.23.10`. The Q4_0 test
checks stored expert tensor headers and compares every streaming output
against the dequantized PyTorch reference with `atol=rtol=2e-3`, tighter
than the CLI's `very` preset. This checks exporter/runtime agreement;
it does not measure the full checkpoint's language-model quality after
requantization.
