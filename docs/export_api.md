---
search:
  boost: 2


---



# :rocket: Main export API's

For details on the exported artifact (directory vs `.tar` vs `.tgz`) and how
`compression_level` and the output path suffix influence it, see
[Artifacts and Compression](./tutos/2_nnef_archive.md#artifacts-and-compression).

### See Also

- NeMo ASR export tutorial: [Export and run NeMo ASR](./tutos/10_nemo.md)
- Transformers/LLM export tutorial: [LLM export guide](./tutos/5_llm.md)
- Shapes remodeler tutorial: [Provider-agnostic remodeler](./tutos/11_remodeler.md)

## Choosing the Target Runtime

`torch_to_nnef` exports to an inference-target abstraction. The most common choice is `TractNNEF`.

- Use `TractNNEF.latest()` for the most recent supported tract.
- Pin a specific tract version: `TractNNEF(SemanticVersion.from_str("0.23.0"))`.
- Pass dynamic-axes constraints and feature toggles (e.g., SDPA reification) through the inference target when needed.

Example

```python
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.utils import SemanticVersion

target = TractNNEF.latest()  # or TractNNEF(SemanticVersion.from_str("0.23.0"))
export_model_to_nnef(
    model=my_model.eval(),
    args=(x,),
    file_path_export="/tmp/model.nnef.tgz",  # suffix expresses archive intent
    inference_target=target,
    input_names=["inp"],
    output_names=["out"],
    compression_level=1,  # 1..9 => .tgz, 0 => .tar, None => .nnef dir
)
```

::: torch_to_nnef.export
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false

## JIT-only model hardening

For JIT-only artifacts whose Python source isn't on the import path
(e.g. `silero_vad.jit`), the standard recursive parser cannot resolve
the inner classes. `harden_jit_for_export` runs a chain of opt-in
graph passes that specialize the JIT graph for your example inputs,
producing a graph the standard exporter can consume. See the
[JIT-only models tutorial](./tutos/12_jit_only_models.md) for the full
chain and rationale per pass.

::: torch_to_nnef.torch_graph.harden
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false

## Remodeler

For boundary‑only transforms (collapse, bind, alias, outputs_keep), see the
dedicated tutorial: [Provider‑agnostic remodeler](./tutos/11_remodeler.md).
