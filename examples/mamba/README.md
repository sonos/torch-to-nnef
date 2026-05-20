# Mamba: streaming inference via NNEF + tract

End-to-end demo of running a HF `MambaForCausalLM` checkpoint from
Rust through tract, no Python at deploy time. Ships **two** runtime
shapes side-by-side so you can pick the one that fits your deploy.

## Two designs

### `external_state/`: per-token graph, caller threads state

The graph is per-token (one input id in, one logits out). Per-layer
conv buffers and SSM states are **explicit IO**: the caller threads
them across `run()` calls.

```
input_id[1], conv_states[L,1,D,K], ssm_states[L,1,D,N]
    -> logits[1,vocab], conv_states_out, ssm_states_out
```

Pros:
- Works on stock tract 0.22; no pulse pipeline needed.
- State is a plain tensor: trivial to checkpoint, branch, serialize, batch.
- Simplest possible graph shape (no symbolic axes).

Cons:
- Caller writes the state-threading loop and the rolling-buffer logic.
- The state tensors are part of the IO surface (~3 MiB for mamba-130m).

### `pulse/`: symbolic-S streaming graph, state internal to tract

The graph is a prefill graph with the sequence length declared
symbolic (`S`). tract's pulse pipeline lowers it to a per-pulse
runner. Each `run()` consumes one token and advances internal
state automatically. The caller never sees conv buffers or SSM
states.

```
input_ids[1, S=1 per pulse]  ->  logits[1, S=1 per pulse, vocab]
```

Pros:
- Cleanest runtime: `state.run(token) -> logits`, fresh `state =
  model.spawn()` per prompt.
- One artifact handles any sequence length.
- Conv1d rolling buffer + SSM h_t carried inside the scan and the
  pulsed conv op, no caller bookkeeping.

Cons:
- Requires tract's pulse pipeline.
- Uses the `t2n_extra::ssm_scan_y` variant (no `h_final` output)
  because tract's Scan pulsifier rejects `"last"` outputs.
- Per-prompt reset is a new `model.spawn()` (cheap, but different
  mental model than "zero the state tensor").

## Tradeoff summary

| Concern                          | external_state           | pulse                     |
| -------------------------------- | ------------------------ | ------------------------- |
| Runtime API                      | thread states explicitly | `state.run(token)`        |
| Caller bookkeeping               | conv + ssm state         | none                      |
| Tract version                    | stock 0.22               | pulse pipeline            |
| Compile-time T                   | not a concept            | symbolic `S`              |
| Conv rolling buffer              | manual in wrapper        | automatic                 |
| State checkpoint / clone         | tensor copy              | `state.clone()` semantics |
| Multi-prompt batching            | trivial                  | one state per stream      |

## Quick start

Each subdir has its own `export.py` + `mamba-rs/` runtime. From this
directory:

```
pip install -r external_state/requirements.txt

# external-state
cd external_state
python export.py --repo state-spaces/mamba-130m-hf --out mamba130m.nnef.tgz
cd mamba-rs && cargo run --release -- \
    --model ../mamba130m.nnef.tgz --tokenizer ../tokenizer.json \
    --prompt "Once upon a time," --max-new-tokens 20

# pulse
cd pulse
python export.py --repo state-spaces/mamba-130m-hf --out mamba130m_pulse.nnef.tgz
cd mamba-rs && cargo run --release -- \
    --model ../mamba130m_pulse.nnef.tgz --tokenizer ../tokenizer.json \
    --prompt "Once upon a time," --max-new-tokens 20
```

Greedy output is byte-identical to HF `model.generate(do_sample=False)`
in both shapes.

## Notes

- The pulse path depends on two tract-side bits merged on `main`:
  the Gather `axes_mapping` (commit 9ceee1b5c) and the `PushSliceUp`
  prefix-boundary fix (commit 803bdad5a, was PR sonos/tract#2247).
  `pulse/mamba-rs/Cargo.toml` pins to a tract main rev (`f070a6d7f`)
  that includes both. Stock 0.22.1 from crates.io is missing both
  fixes and will not pulse this graph.
- End-to-end on mamba-130m: identical decoded text to HF generate,
  ~54 ms/step median in pulse-mode Rust runtime (vs ~40 ms/step in
  external_state). The pulse runner is slightly slower per step
  because the pulse pipeline produces a different op shape than the
  per-token static graph; the per-step matmul packing cost is the
  same in both. Per-token max abs diff vs PyTorch: 1.3e-4, argmax
  identical.
- For day-to-day deployment of mamba-130m today, `external_state/`
  is the simpler choice and works on stock tract 0.22 with no patch.
- The `t2n_extra::ssm_scan_y` op (pulse-friendly variant of
  `ssm_scan`, drops `h_final`) is bundled in the t2n PR alongside
  the original; both are covered by
  `tests/test_t2n_extra_ssm_scan.py`.
