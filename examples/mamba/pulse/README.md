# Pulse-mode Mamba

The graph is a prefill graph with the sequence axis declared symbolic
(`S`). tract's pulse pipeline lowers it to a per-pulse streaming
runner: one token in, one logits row out, internal conv + SSM state
carried across pulses by tract.

Caller-visible shape:

```
input_ids[1, S=1 per pulse]  ->  logits[1, S=1 per pulse, vocab]
```

Per prompt: spawn a fresh `SimpleState`, feed prompt tokens, then
greedy-decode. The state object owns the per-layer conv buffers and
SSM `h_t`. No tensors threaded by the caller.

## What's different from `external_state/`

- Uses the `t2n_extra::ssm_scan_y` variant (no `h_final` output);
  tract's Scan pulsifier rejects `"last"` outputs.
- `h_init` is baked into the graph as a constant zeros buffer.
- Export declares `dynamic_axes={"input_ids": {1: "S"}}`. One artifact
  serves any sequence length at runtime.
- Rust runtime calls `PulsedModel::new(&typed, sym, &1.to_dim())` and
  spawns a fresh `SimpleState` per prompt.

## Quick start

```
pip install -r ../external_state/requirements.txt
python export.py --repo state-spaces/mamba-130m-hf --out mamba_pulse.nnef.tgz
cd mamba-rs
cargo run --release -- \
    --model ../mamba_pulse.nnef.tgz \
    --tokenizer ../tokenizer.json \
    --prompt "Once upon a time, in a quiet little village," \
    --max-new-tokens 20
```

Greedy continuation matches HF `model.generate(do_sample=False)`.
