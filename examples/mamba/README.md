# Mamba streaming export and Rust runtime

End-to-end demo: export a HF `MambaForCausalLM` checkpoint as a
per-token NNEF streaming graph and run it from Rust through tract, no
Python at deploy time.

## Layout

- `streaming_wrapper.py`: a `nn.Module` that exposes the per-token
  decoding body of Mamba's mixer as
  `(input_id, conv_states, ssm_states) -> (logits, conv', ssm')`.
- `export.py`: loads a HF Mamba checkpoint, wraps it, runs the t2n
  exporter, and writes a sidecar JSON manifest plus the HF
  `tokenizer.json` next to the artifact.
- `mamba-rs/`: minimal Rust binary that loads the artifact, tokenizes a
  prompt, threads the conv and ssm states across tokens, and
  greedy-decodes new tokens.

## Quick start

```
pip install -r requirements.txt
python export.py --repo state-spaces/mamba-130m-hf --out mamba130m.nnef.tgz
cd mamba-rs
cargo run --release -- \
    --model ../mamba130m.nnef.tgz \
    --tokenizer ../tokenizer.json \
    --prompt "Once upon a time, in a quiet little village," \
    --max-new-tokens 20
```

The runtime prints the decoded continuation and per-step latency.
Greedy output is byte-identical to HF `model.generate(..., do_sample=False)`
with the matching patched tract.

## Notes

- The streaming artifact is per-token. Whole-sequence prefill in a
  single tract call is also supported by exporting with `seq_len > 1`,
  but requires the `PushSliceUp` boundary fix from tract PR #2247 for
  any prompt with `len >= 4`.
- The artifact stores all weights inline (`mamba-130m` is ~520 MB
  fp32). Per-token state is ~3 MB.
