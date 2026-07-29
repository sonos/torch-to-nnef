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
    - `eval_symbols` (optional): `{ SYMBOL: int_value }` -- pin dynamic symbols
      to concrete sizes in test inputs during export
  - `outputs` (optional): per-output settings
    - `collapse_dims`: list of axis indices to squeeze from the output
  - `renamed_symbols` (optional): `{ TARGET: [SOURCES...] }` for backend-facing
    symbol unification
  - `outputs_keep` (optional): list of outputs to keep (template pre-fills)
  - `extensions` (optional): list of custom extension strings injected into
    the exported NNEF (e.g., `tract_assert` constraints for pulsification)

Symbol conventions
- Symbols are uppercased. Providers may namespace per input (e.g., `TARGETS__TIME`, `TARGETS__BATCH`).
- Batch symbols end with `__BATCH`. To unify multiple batch-like symbols under a single tract-facing `BATCH`, declare:
  `renamed_symbols: { BATCH: [TARGETS__BATCH, STATES_0__BATCH, ...] }`.
- Aliases listed in `renamed_symbols` are honored wherever symbols are referenced (collapse/bind/validation).

Provider-agnostic Python workflow

The remodeler API is provider-agnostic. Any provider that can discover
subnet signatures can participate. The typical flow is the same:

1) Discover RAW signatures → dump a starter registry
2) Edit the YAML (collapse/bind/alias/outputs_keep)
3) Validate the edited config against discovered signatures
4) Build a plan and apply it → wrapped subnets expose the remodeled boundary

```python
from pathlib import Path

# Conceptually, `provider` can be any implementation exposing
#   discover_signatures(model, Stage) and apply(model, plan).
# For a concrete example, see the NeMo provider below.

from torch_to_nnef.remodeler import (
  Stage,
  plan_from_registry,
  save_config,
)
from torch_to_nnef_nemo.registry_utils import (
  dump_registry_from_signatures,
  validate_registry_against_signatures,
)
from torch_to_nnef_nemo.axis_registry import load_axis_symbol_registry

# 1) Discover RAW signatures (provider-specific model omitted here)
signatures = provider.discover_signatures(model, Stage.RAW)

# 2) Dump a starter YAML registry (pre-fills outputs_keep)
registry = dump_registry_from_signatures(signatures)
save_config(Path("./shapes.yaml"), registry)

# 3) Validate user-edited config against discovered signatures
cfg = load_axis_symbol_registry(Path("./shapes.yaml"))
validate_registry_against_signatures(signatures, cfg)

# 4) Apply the plan: returns {subnet_name: wrapped_module}
plan = plan_from_registry(cfg)
wrapped = provider.apply(model, plan)
```

Generate a starter config (NeMo CLI example)

```bash
t2n_export_nemo \
  --inspect-signatures \
  --dump-shape-config ./shapes.yaml \
  --split-joint-decoder \
  --model-slug nvidia/parakeet-tdt-0.6b-v3
```

NeMo provider (Python example)

```python
import torch
from pathlib import Path

from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef_nemo.model_loader import load_asr_model_from_nemo_slug
from torch_to_nnef_nemo.provider import NemoProvider
from torch_to_nnef.remodeler import (
    Stage,
    plan_from_registry,
    save_config,
)
from torch_to_nnef_nemo.registry_utils import (
    dump_registry_from_signatures,
    validate_registry_against_signatures,
)
from torch_to_nnef_nemo.axis_registry import load_axis_symbol_registry

# Discover and dump a starter config
asr = load_asr_model_from_nemo_slug("<your-nemo-asr-model>").eval()
target = TractNNEF.latest()
prov = NemoProvider(inference_target=target, split_joint_decoder=True)
signatures = prov.discover_signatures(asr, Stage.RAW)
registry = dump_registry_from_signatures(signatures)
save_config(Path("./shapes.yaml"), registry)

# Validate and apply a user-edited config
cfg = load_axis_symbol_registry(Path("./shapes.yaml"))
validate_registry_against_signatures(signatures, cfg)
plan = plan_from_registry(cfg)
wrapped = prov.apply(asr, plan)  # {"encoder": nn.Module, ...}
```

Pretty printing with Rich

```python
from torch_to_nnef.remodeler.rich_render import print_signatures_rich
from torch_to_nnef.remodeler import Stage
import rich

signatures = prov.discover_signatures(asr, Stage.FINAL)
print_signatures_rich(signatures, diff=True, rich=rich, model_label="MyNeMo")
```

Export a wrapped subnet (generic)

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
    file_path_export="./wrapped_subnet.nnef.tgz",
    inference_target=target,
    input_names=getattr(enc, "input_names", []),
    output_names=getattr(enc, "output_names", []),
    nnef_variable_naming_scheme=VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
)
```

Eval symbols

The `eval_symbols` field lets you pin dynamic symbols to concrete values in
test inputs at export time. This resizes the test tensors that are traced
through the model, which is useful when the default test-input size is not
representative of the target inference scenario (e.g., single-step decoding
where `TIME = 1`).

- Values are per-input, mapping symbol names to integer sizes.
- Symbols are automatically uppercased.
- Dimensions smaller than the target are zero-padded; dimensions larger are
  narrowed.

Example:

```yaml
decoder:
  inputs:
    targets:
      original_shape: [TARGETS__BATCH, TARGETS__TIME]
      collapse_dims: []
      eval_symbols: {TARGETS__TIME: 1}
    states_0:
      original_shape: [2, STATES_0__BATCH, 640]
      collapse_dims: []
      eval_symbols: {STATES_0__BATCH: 1}
```

In this example the `targets` tensor's TIME axis is pinned to 1 and the
`states_0` tensor's BATCH axis is pinned to 1 before tracing.

Custom extensions

The `extensions` field lets you inject arbitrary extension strings into the
exported NNEF subnet. This is typically used for `tract_assert` constraints
that encode dimensionality limits implied by the model architecture (e.g.,
max receptive field of a conv stack) which are required for correct
pulsification.

- Extensions are per-subnet, as a list of strings.
- Each string is passed verbatim to the NNEF export.
- For known NeMo pretrained models, the CLI auto-populates extensions from a
  built-in registry (`slug_extensions.py`). User-provided extensions take
  precedence.

Example:

```yaml
encoder:
  extensions:
    - "tract_assert AUDIO_SIGNAL__TIME<=39993"
  inputs:
    audio_signal:
      original_shape: [AUDIO_SIGNAL__BATCH, 128, AUDIO_SIGNAL__TIME]
      collapse_dims: [AUDIO_SIGNAL__BATCH]
```

Output collapse
- When `collapse_dims` removes a batch axis from inputs, the internal module
  still runs with that axis (size 1). The `outputs.collapse_dims` setting
  squeezes configured axes from each output tensor so the exported model
  produces batch-free tensors.
- Each output can declare its own `collapse_dims` as a list of axis indices.
- For NeMo models, the registry auto-populates `collapse_dims: [0]` for all
  outputs when batch collapse is detected on inputs. Explicit config takes
  precedence over auto-population.

Example:

```yaml
encoder:
  inputs:
    audio_signal:
      original_shape: [AUDIO_SIGNAL__BATCH, 128, AUDIO_SIGNAL__TIME]
      collapse_dims: [AUDIO_SIGNAL__BATCH]
  outputs_keep: [outputs]
  outputs:
    outputs:
      collapse_dims: [0]
```

Validation
- The remodeler validates configs early against discovered signatures:
  - Rejects unknown subnets/inputs
  - Ensures `outputs_keep` is a subset of outputs
  - Ensures `outputs.collapse_dims` references known output names with valid axis indices
  - Verifies `bind_scalar_to_dim_size` sources and symbols exist among dynamic axes
  - Verifies `collapse_dims` symbols exist among dynamic axes per input
  - Verifies `renamed_symbols` sources exist among the subnet's dynamic axes

Exporting
- Use wrapped subnets with `export_model_to_nnef` if you need direct control;
  the NeMo CLI already applies the remodel plan during export when a
  `--shape-config` is provided.

See also
- [NeMo tutorial](./10_nemo.md) (integrates the remodeler)
- [Export API overview](../export_api.md)
