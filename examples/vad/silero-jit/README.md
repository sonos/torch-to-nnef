# Silero-VAD JIT export

End-to-end demo of the JIT-only export path on a real-world artifact. Documented in detail in the [JIT-only models tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/12_jit_only_models/).

## What this demonstrates

[Silero-VAD](https://github.com/snakers4/silero-vad) ships as a `silero_vad.jit` file produced by `torch.jit.save`. The Python source for the inner classes (`__torch__.vad.model.vad_annotator.SileroVadBlock`, etc.) is *not* on the import path of a normal install, so the standard t2n recursive parser would raise `ModuleNotFoundError` before reaching the op handlers.

`export_model_to_nnef` auto-detects `torch.jit.ScriptModule` inputs and applies `harden_jit_for_export` internally. The example is therefore the trivial four-line export call you'd expect from any `nn.Module`.

This example is *not* the same as [`examples/vad/FSMN-wasm/`](../FSMN-wasm/), which is a wasm runtime demo of the existing dynamic-axes export path. Use this directory if you have a `.jit` artifact whose Python source is unavailable; use `examples/vad/FSMN-wasm/` if you want to see VAD running in the browser.

## Run

```bash
# from the repo root
cd examples/vad/silero-jit

# install the example's dependencies (uses the in-repo torch-to-nnef)
pip install -r requirements.txt

# export
python export.py
```

Expected output: a log line confirming auto-harden ran, followed by the standard export progress and the path of the exported `silero_vad.nnef.tgz`. The tract round-trip check (`check_io=True`) runs as part of the export and will fail loudly if the NNEF doesn't match PyTorch numerically.

## Opt-out / fine-grained control

If you want per-pass fold counts (for debugging an unfamiliar JIT artifact), call the helper yourself and pass `auto_harden_jit=False` to the exporter:

```python
from torch_to_nnef import (
    TractNNEF,
    export_model_to_nnef,
    harden_jit_for_export,
)

diagnostics: dict[str, object] = {}
hardened = harden_jit_for_export(raw_model, args, diagnostics=diagnostics)
print(diagnostics)  # per-pass fold counts + freeze flag

export_model_to_nnef(
    model=hardened, args=args, ..., auto_harden_jit=False
)
```

## torch version

Tested on torch 2.11.0. The data-dependent If fold calls `torch._C._jit_interpret_graph`, an undocumented internal API exposed since torch 1.10. Earlier 2.x should work but isn't CI-gated.
