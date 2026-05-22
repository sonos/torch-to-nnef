# 8. Custom operators

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Control the transformation to NNEF of `nn.Module` as you wish.
        This is often useful in case those modules are not representable in the
        jit graph of PyTorch or because you wish to use custom NNEF operator for
        your inference engine.

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

Sometimes you want to control how an operation is exported to NNEF.
This can happen because you want to target a custom fragment in your
inference engine instead of a long chain of primitives, or simply
because the logic is not traceable faithfully.

This page shows two supported patterns:
- New: t2n_extra custom ops (function-level, external-friendly)
- ModuleInfoExtractor (module-level, legacy/advanced)

Both are first-class and can co-exist in the same model.

New: t2n_extra custom ops
---------------------------------

Use PyTorch’s `torch.library.custom_op` to declare an opaque
operation under the `t2n_extra::<name>` namespace, call it in your
model, and register a small NNEF emit function with
`torch_to_nnef.op.extras.register("<name>")`.

Minimal end-to-end example:

```python
# 1) Define a custom PyTorch op (anywhere that runs before export)
import torch
lib = torch.library.Library("t2n_extra", "DEF")
lib.define("my_relu(Tensor x) -> Tensor")

# Optional: eager/meta kernels for runtime use, not required for export-only
# lib.impl(...)

# 2) Register the NNEF handler in a module you can import
from torch_to_nnef.op.extras import register

@register("my_relu")
def my_relu(g, node, name_to_tensor, null_ref, *, torch_graph, inference_target, op_helper, **_):
    # Convert the input IR node to an NNEF tensor
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    # Emit the NNEF op and bind its output to the traced tensor name
    y = op_helper.add_single_output_op_from_nnef_tensors(
        node=node,
        nnef_op_type="relu",
        inputs=x,
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    return []  # no custom fragment keys emitted

# 3) Call it from your model
class M(torch.nn.Module):
    def forward(self, x):
        return torch.ops.t2n_extra.my_relu(x)

# 4) Ensure the handler module is imported before export
#    Option A: explicitly import your module
import my_project.t2n_to_nnef_handlers  # noqa: F401

#    Option B: let the exporter auto-import it
from torch_to_nnef import export_model_to_nnef, TractNNEF
export_model_to_nnef(
    M(),
    args=(torch.randn(2, 3),),
    file_path_export="/tmp/m.nnef",
    inference_target=TractNNEF.latest(),
    load_extra_op_modules=["my_project.t2n_to_nnef_handlers"],
)

#    Option C: via env var (comma-separated modules)
#    TORCH_TO_NNEF_EXTRA_MODULES=my_project.t2n_to_nnef_handlers python export.py
```

Notes
- Namespace: use `t2n_extra::<name>` only — the exporter routes those to
  your handler via `torch_to_nnef.op.extras`.
- Helper: the `op_helper` offers small utilities to convert inputs and
  emit ops; see `torch_to_nnef.op.helper.OpHelper`.
- Fragments: if your op maps to a custom fragment, return its key(s)
  from the handler so they are written into the archive.

See also
- Example repo folder with a runnable script:
  https://github.com/sonos/torch-to-nnef/tree/main/examples/t2n_extra_custom_op

Module-level extractors (ModuleInfoExtractor)
---------------------------------------------

Alternatively, you can intercept a `torch.nn.Module` call site and
author its NNEF directly by subclassing
[`torch_to_nnef.op.custom_extractors.ModuleInfoExtractor`](../../reference/torch_to_nnef/op/custom_extractors/base/):

<div class="grid cards" markdown>
- ::: torch_to_nnef.op.custom_extractors.ModuleInfoExtractor
    handler: python
</div>

To make it work you need to accomplish 4 steps:

1. sub-classing it
2. defining its `MODULE_CLASS` attribute.
3. defining its `convert_to_nnef`
4. ensuring that the subclass you just defined is imported in your export script

This would look like:

```python
from torch_to_nnef.op.custom_extractors import ModuleInfoExtractor

class MyCustomHandler(ModuleInfoExtractor):
    MODULE_CLASS = MyModuleToCustomConvert

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        **kwargs
    ):
        pass
```

You can take inspiration from our own management of RNN layers like:
<div class="grid cards" markdown>
- ::: torch_to_nnef.op.custom_extractors.LSTMExtractor
    handler: python
</div>

But ultimately this is just a chain of op's that needs to be written,
inside the `g` graph, like we do when [adding new aten operator](../../contributing/add_new_aten_op)

Which one should I use?
-----------------------

- t2n_extra custom ops: best for function-style ops that you can call
  from Python (easy to ship externally; no coupling to a module class).
  You patch your model to call `torch.ops.t2n_extra.<name>(...)` and
  register one small handler function.
- ModuleInfoExtractor: best when you need to preserve a module call
  boundary (e.g., complex `nn.Module` like RNNs) or need tight control
  over inputs/outputs independent of the traced function usage.

Both routes produce a single opaque node in the IR and then emit your
custom NNEF subgraph; pick the one that matches your authoring style
and where you want the “hook” to live (op call vs. module call).
