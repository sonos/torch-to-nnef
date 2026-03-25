---
search:
  boost: 2


---



# :rocket: Main export API's

For details on the exported artifact (directory vs `.tar` vs `.tgz`) and how
`compression_level` and the output path suffix influence it, see
[Artifacts and Compression](./tutos/2_nnef_archive.md#artifacts-and-compression).

### See Also

- NeMo ASR export tutorial: [Export and run NeMo ASR](./examples/nemo_asr/README.md)
- Transformers/LLM export tutorial: [LLM export guide](./tutos/5_llm.md)
- Shapes remodeler tutorial: [Provider-agnostic remodeler](./tutos/11_remodeler.md)

## Choosing the Target Runtime

`torch_to_nnef` exports to an inference-target abstraction. The most common choice is `TractNNEF`.

- Use `TractNNEF.latest()` for the most recent supported Tract.
- Pin a specific Tract version: `TractNNEF(SemanticVersion.from_str("0.23.0"))`.
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

## Remodeler API (overview)

For boundary-only transforms, the remodeler provides a small, typed API and a
strict nested YAML/JSON config.

Key entry points (see tutorial for end-to-end examples):

```python
from torch_to_nnef.remodeler import (
  Stage as RemodelStage,
  dump_registry_from_signatures,
  load_config,
  plan_from_registry,
  save_config,
  validate_registry_against_signatures,
)
```

Providers implement discovery/apply (e.g., `NemoProvider`). The NeMo CLI wires
this when `--shape-config` is provided.
