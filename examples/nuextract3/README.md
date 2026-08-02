# NuExtract3: image to Markdown with a Rust + tract runner

Export **numind/NuExtract3** (`model_type = "qwen3_5"`) to NNEF and run a
minimal Rust + tract generation loop that converts a document image to Markdown.

This example is intentionally stacked on the Qwen3.5 dense VLM support from
PR #305. NuExtract3 uses the same hybrid decoder family: most layers are
gated-delta-net linear-attention layers carrying a streaming conv state plus a
matrix recurrent state, with full-attention KV-cache layers interleaved. The
Rust runtime threads all of those states explicitly.

## Layout

- `export.py`: exports `vision.nnef.tgz`, `decoder.nnef.tgz`, a
  `nuextract3.json` runtime manifest, the sample tensors, and `tokenizer.json`.
- `nuextract3-rs/`: single Rust binary. Loads both graphs, runs vision encoder
  then decoder prefill/decode in tract, decodes generated token ids with the
  exported tokenizer, and prints Markdown.

## Export

```bash
# Tiny random model, no download. Proves the plumbing, output is meaningless.
python export.py --dummy --out ./exp

# Real NuExtract3 image-to-Markdown export.
python export.py --repo numind/NuExtract3 \
    --dtype f16 \
    --image receipt.png \
    --prompt "Convert this document image to Markdown." \
    --out ./exp
```

This writes:

| file | what |
|---|---|
| `vision.nnef.tgz` | dynamic-resolution vision tower |
| `decoder.nnef.tgz` | streaming hybrid decoder, usable for prefill and decode |
| `nuextract3.json` | graph paths, token ids, hidden size, and state layout |
| `tokenizer.json` | tokenizer used by the Rust runner to decode Markdown |
| `*.dat` | sample `input_ids`, `position_ids`, and `pixel_values` tensors |

## Run

```bash
cd nuextract3-rs
cargo run --release -- --dir ../exp --max-new-tokens 256
```

Or run everything:

```bash
# dummy, no download
./run.sh

# real checkpoint
REPO=numind/NuExtract3 IMAGE=receipt.png DTYPE=f16 MAX_NEW_TOKENS=256 ./run.sh
```

The dummy path is suitable for CI and checks graph/runtime wiring. For a real
checkpoint, the Rust runner prints the decoded Markdown produced by tract.
