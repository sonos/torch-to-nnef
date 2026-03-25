---
search:
  boost: 2
---

# 11. Shapes Remodeler (provider-agnostic)

This tutorial introduces the boundary remodeler: a provider-agnostic way to
describe and apply IO-boundary transforms (collapse, bind, and symbol aliases)
for subnets/modules at export time.

Why use it
- Stabilize IO contracts across environments by naming dynamic axes.
- Collapse size-1 dynamic dims externally while keeping internal shapes.
- Bind scalars to dynamic sizes to simplify external inputs.
- Keep a subset of outputs per subnet.

Core concepts
- Nested config per subnet:
  - `inputs`: per-input settings
    - `original_shape`: list of dims (ints or symbols)
    - `collapse_dims` (optional): symbols to drop at boundary
    - `bind_scalar_to_dim_size` (optional): `subnet.input.SYMBOL`
  - `renamed_symbols` (optional): `{ TARGET: [SOURCES...] }` for backend-facing
    symbol unification
  - `outputs_keep` (optional): list of outputs to keep (template pre-fills)

Generate a starter config (CLI)

```bash
t2n_export_nemo \
  --inspect-signatures \
  --dump-shape-config ./shapes.yaml \
  --split-joint-decoder \
  --model-slug nvidia/parakeet-tdt-0.6b-v3
```

Programmatic usage (Python)

```python
import torch
from pathlib import Path

from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nemo_tract.model_loader import load_asr_model_from_nemo_slug
from torch_to_nnef.nemo_tract.provider import NemoProvider
from torch_to_nnef.remodeler import (
    Stage as RemodelStage,
    dump_registry_from_signatures,
    load_config,
    plan_from_registry,
    save_config,
    validate_registry_against_signatures,
)

# Discover and dump a starter config
asr = load_asr_model_from_nemo_slug("nvidia/parakeet-tdt-0.6b-v3").eval()
target = TractNNEF.latest()
prov = NemoProvider(inference_target=target, split_joint_decoder=True)
signatures = prov.discover_signatures(asr, RemodelStage.RAW)
registry = dump_registry_from_signatures(signatures)
save_config(Path("./shapes.yaml"), registry)

# Validate and apply a user-edited config
cfg = load_config(Path("./shapes.yaml"))
validate_registry_against_signatures(signatures, cfg)
plan = plan_from_registry(cfg)
wrapped = prov.apply(asr, plan)  # {"encoder": nn.Module, ...}
```

Export a wrapped subnet

```python
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme

enc = wrapped["encoder"].eval()
# Prefer the module-provided example if available
ie = enc.input_example() if hasattr(enc, "input_example") else ()
args = ie if isinstance(ie, tuple) else tuple(ie)

export_model_to_nnef(
    model=enc,
    args=args,
    file_path_export="./encoder.nnef.tgz",
    inference_target=target,
    input_names=getattr(enc, "input_names", []),
    output_names=getattr(enc, "output_names", []),
    nnef_variable_naming_scheme=VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
)
```

Validation
- The remodeler validates configs early against discovered signatures:
  - Rejects unknown subnets/inputs
  - Ensures `outputs_keep` is a subset of outputs
  - Verifies `bind_to_dim` sources and symbols exist among dynamic axes
  - Verifies `collapse_dims` symbols exist among dynamic axes per input
  - Verifies `renamed_symbols` sources exist among the subnet’s dynamic axes

Exporting
- Use wrapped subnets with `export_model_to_nnef` if you need direct control;
  the NeMo CLI already applies the remodel plan during export when a
  `--shape-config` is provided.

See also
- NeMo tutorial (integrates the remodeler): ../examples/nemo_asr/README.md
- Export API overview: ../export_api.md
