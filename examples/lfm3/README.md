# LFM3-1.7B-Base to NNEF

Export [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base)
with its original remote modeling code. The example pins revision
`7bca5038823c0d0f14efe06439109354a10e014c`; loading it executes that code.

From the repository root, install the LLM dependencies (the model requires
Transformers 5.12.1 or newer; tested with 5.19.0):

```sh
uv sync --extra llm
uv run --with 'transformers>=5.12.1,<6' python examples/lfm3/export.py \
  --tract /path/to/tract --out /tmp/lfm3-export
```

The default is float32. `--dtype f16` reduces weight storage, but float32 is
the validated path. The checkpoint download is about 3.5 GB; loading in float32
and tracing require substantially more RAM. `--model-dir /path/to/snapshot`
uses an already downloaded checkpoint. `--tiny` uses random small layers with
the same three layer types and a four-token sliding window; it still downloads
configuration, tokenizer, and modeling code, but no weights.

The export checks tract outputs against PyTorch and writes `model.nnef.tgz`,
tokenizer/configuration files, and `state.json`. The latter records each state
input's paired output and initial shape. For generation:

1. Initialize every state to zeros with the shapes in `state.json` (KV time
   dimension zero, convolution history width six).
2. Supply an unpadded batch of one prompt as `input_ids` (`int64`).
3. Feed each state output back into its paired input for every subsequent
   chunk or token. `outputs` contains the final token's logits.

One graph supports prompt prefill, chunked prefill, and single-token decode.
Full and sliding attention both retain all past KV tensors on the dynamic `P`
axis. Sliding layers apply the 256-token attention mask, but their storage is
currently unbounded. Convolution states have fixed shape `[1, 2048, 6]`.
Packed documents, padding, and batches larger than one are outside this export
contract. The model is a base checkpoint; use plain completion prompts.

The standard `t2n_export_llm_to_tract` command also selects the LFM3 handler
from `model_type=lfm3`.

## Regression checks

```sh
LFM3_MODEL_DIR=/path/to/snapshot TRACT_BIN=/path/to/tract \
  uv run --extra llm pytest packages/llm/tests/test_lfm3_handler.py
```

The full float32 checkpoint was exported and checked with tract 0.23.5-pre,
PyTorch 2.14.1, and Transformers 5.19.0. The NNEF artifact is about 6.5 GiB.

The tests cover small three-layer and full-depth 24-layer configurations.
They compare streaming logits against the original cacheless model and run
one exported graph with empty history, single-token decode, and chunks crossing
the sliding-window and convolution boundaries. Integration tests skip when the
local snapshot or tract binary is not configured.
