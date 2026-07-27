# Holo-3.1 (Qwen3.5-VL): single-binary tract demo

Export **Hcompany/Holo-3.1** (a GUI-agent VLM, `model_type = "qwen3_5"`) to NNEF
and run the whole thing in a **single Rust + tract binary**: a screenshot's
patches go through the vision tower, the embeddings splice into the prompt, and
the hybrid gated-delta-net decoder greedy-generates the answer (UI-grounding
click coordinates), threading its state across steps.

The decoder is the interesting part. Per `config.layer_types`, three of every
four layers are **gated-delta-net (GDN) linear-attention** layers, a streaming
depthwise conv state plus a matrix recurrent state, whose recurrence lowers to
the `t2n_extra::gated_delta_scan` op (a `tract_core_scan`). The fourth is a
standard attention layer with a KV cache. The Rust runtime threads all three
state kinds across the generation loop, exactly like a KV cache.

## Layout

- `export.py`: produces the two NNEF graphs + `holo.json` manifest + a sample
  input, from a real checkpoint or a tiny random dummy (no download).
- `holo-rs/`: the Rust binary. Loads both graphs, runs encoder then decoder
  prefill then greedy decode, prints the generated token ids.

## 1. Export

```bash
# tiny random model, no download, proves the pipeline (CI / plumbing):
python export.py --dummy --out ./exp

# a real checkpoint + screenshot:
python export.py --repo Hcompany/Holo-3.1-0.8B \
    --image screenshot.png --prompt "Click the search bar" --out ./exp

# a bigger checkpoint in half precision (fits in half the memory):
python export.py --repo <bigger-qwen3_5-vlm> --dtype f16 \
    --image screenshot.png --out ./exp
```

**Any size is just the `--repo` parameter.** Everything downstream (layer
count, the gated-delta / attention split from `config.layer_types`, hidden size,
GDN and attention head dims, vision depth) is read from the checkpoint config,
so a bigger Holo / Qwen3.5-VL model needs no code change. `--dtype f16` loads
and exports in half precision for the large checkpoints; the Rust runtime casts
each input to whatever dtype the graph expects, so it needs no change either
(the gated-delta recurrent state stays f32, as in the reference kernels).

This writes to `./exp`:

| file | what |
|---|---|
| `vision.nnef.tgz` | vision tower, dynamic resolution (grid axes symbolic) |
| `decoder.nnef.tgz` | streaming hybrid decoder (one graph: prefill + decode) |
| `holo.json` | shapes + per-layer state layout + decoder I/O order |
| `*.bin` | sample `input_ids` / `position_ids` / `pixel_values` |

The **streaming** decoder differs from the manifest joint export
(`t2n_export_multimodal_to_tract`): the integer `position_ids`, the causal mask,
and every layer's state are runtime **inputs**, so one dynamic graph serves both
prefill (S>1, zero states) and decode (S=1, carried states) and stays exact on
tract (`check_io` passes).

RoPE (interleaved mRoPE) is computed **in-graph** from `position_ids`. Upstream
does the mRoPE channel interleave with a strided in-place scatter that does not
lower faithfully to tract; the export replaces it with an equivalent
constant-masked sum (bit-exact), so the runtime feeds only integer positions.
The one part still done host-side is the position *layout*: `export.py` uses
transformers' `get_rope_index` to place the image span (a 2-D grid) into the
prompt positions, and the Rust side continues positions sequentially per
generated token. The `--verify N` flag prints a torch greedy-decode of the same
loop so you can confirm the Rust tokens match bit-for-bit (they do on the dummy
export).

## 2. Run

```bash
cd holo-rs
cargo run --release -- --dir ../exp --max-new-tokens 16
```

On the `--dummy` export the weights are random, so the token ids are not
meaningful. The point is that the full two-graph tract pipeline (vision encoder
+ hybrid GDN/attention decoder with conv + recurrent + KV state) runs end to
end. Point `--dir` at a real-checkpoint export and decode the ids with the
model's tokenizer to read the grounding coordinates.

## Notes

- `tract-nnef` is pinned to the same revision as the `mamba` examples (the last
  main rev before the f32 AMX dispatch regression on M=1 matmuls).
- The vision tower exports **once** and runs at any resolution; only the decoder
  needs the streaming treatment.
