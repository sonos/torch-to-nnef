# Export and run an Inkling decoder

This example exports a small, randomly initialized Inkling 2 decoder to NNEF
and runs it in tract. It demonstrates chunked prefill and single-token decoding,
including the KV and convolution state carried between calls. It downloads no
model weights and does not generate meaningful text.

From the repository root, install the LLM dependencies and the Transformers
revision used by this example:

```sh
uv sync --extra llm
uv pip install 'transformers @ git+https://github.com/huggingface/transformers.git@d8f10a82213a3a9db48385aba3a4b57ab535de50'
.venv/bin/python examples/inkling2/export.py \
  --tract /path/to/tract --out /tmp/inkling2-f32
```

Use tract 0.23.10 or newer. To try the standard Q4_0 registry, add `--q4`:

```sh
.venv/bin/python examples/inkling2/export.py \
  --tract /path/to/tract --out /tmp/inkling2-q4 --q4
```

The script exports one graph, then feeds chunks of 3, 1, 5, and 1 tokens through
it. Each step uses the previous **tract outputs** as its next state and compares
all outputs with a separate PyTorch run. Successful execution prints a matching
result for each step.

The output directory contains:

- `model.nnef.tgz`: the decoder with dynamic token and KV-history lengths.
- `state.json`: each state input's paired output, initial shape, and dtype.
- `step-N-inputs.npz` and `step-N-outputs.npz`: the inputs and tract outputs for
  each streaming step.

To build a generation loop, initialize the states described in `state.json`
to zero, pass token IDs as an unpadded batch of one, and feed each state output
back into its paired input. `outputs` contains logits for the supplied tokens.
Sliding attention currently retains full KV history, so memory grows with the
sequence length even though attention visibility is limited.

## Export a pretrained checkpoint

The standard CLI selects the Inkling handler automatically:

```sh
.venv/bin/t2n_export_llm_to_tract \
  -s thinkingmachines/Inkling-2-Mini \
  -e /tmp/inkling2-pretrained \
  --force-module-dtype f32 \
  --attn-implementation eager \
  --tract-specific-path /path/to/tract
```

This full-checkpoint path has **not been validated**. Loading converts native
NVFP4/MXFP8 experts to dense weights; float32 weights alone require roughly
106 GB. Q4_0 export reduces stored weights but does not avoid dense loading or
provide a fused quantized expert kernel. Image inputs and the speculative draft
model are outside this text export's scope.
