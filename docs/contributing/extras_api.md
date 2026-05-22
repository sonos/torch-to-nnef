# Extras API (t2n_extra handlers)

This page summarizes the external handler API for `t2n_extra::<name>`
custom ops, and the helper utilities available when emitting NNEF.

Basics
- Declare your op with `torch.library.custom_op("t2n_extra::<name>", ...)`.
- Register a handler in a module that you can import before export:

```python
from torch_to_nnef.op.extras import register

@register("my_op")
def my_op(
    g,
    node,
    name_to_tensor,
    null_ref,
    *,
    torch_graph,
    inference_target,
    op_helper,
    **kwargs,
):
    # Convert input IR nodes to NNEF tensors
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    # Emit an NNEF op and bind to the traced output name
    op_helper.add_single_output_op_from_nnef_tensors(
        node=node,
        nnef_op_type="relu",
        inputs=x,
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    return []  # or ["my_fragment_key"] if you emit a custom fragment
```

Loading handlers
- Pass modules explicitly: `export_model_to_nnef(..., load_extra_op_modules=["my_pkg.handlers"])`.
- Or via env var: `TORCH_TO_NNEF_EXTRA_MODULES=my_pkg.handlers`.
- Or install a plugin that exposes an entry point under
  `torch_to_nnef.extras` (module path as the entry point value).
 - Load order (first wins on duplicates): explicit list → env var → entry points.
 - CI option: set `strict_extra_imports=True` to fail fast if any module fails to import.

Publishing a plugin
- In your package's `pyproject.toml`, add an entry point so
  `export_model_to_nnef(..., discover_extra_entrypoints=True)` can load it:

```toml
[project.entry-points."torch_to_nnef.extras"]
my_pkg = "my_pkg.handlers"
```
- `my_pkg.handlers` should import `torch_to_nnef.op.extras.register` and
  register your handlers at module import time (see the example above).

Handler signature
- Parameters
  - `g`: the NNEF graph under construction.
  - `node`: the IR op node (`t2n_extra::<name>`).
  - `name_to_tensor`: mapping of IR tensor names to `nnef_tools.model.Tensor`.
  - `null_ref`: a scalar `nnef_tools.model.Tensor` (shape=()).
  - `torch_graph`: current `TorchModuleIRGraph` (access IR context if needed).
  - `inference_target`: `TractNNEF` or `KhronosNNEF` (target-specific behavior).
  - `op_helper`: helper for common tasks (see below).
  - `**kwargs`: reserved for future expansion.
- Return: `List[str | Fragment]` — names/instances of custom fragments to ship
  with the archive (empty if your handler emits only built-ins).

OpHelper cheatsheet
- `get_or_add_tensor_variable_in_nnef(node) -> NTensor`:
  bring an input IR node into the NNEF graph.
- `add_single_output_op_from_nnef_tensors(node, nnef_op_type, inputs, ...)`:
  emit a single-output NNEF op bound to `node.outputs[0]`.
- `add_multi_output_op_from_nnef_tensors(node, nnef_op_type, inputs, ...)`:
  same for multi-output ops (returns a list of NTensor).
- `add_intermediate_op(src, op_type, attrs, new_shape, suffix) -> NTensor`:
  emit an intermediate op producing a fresh NTensor (helpful for
  multi-step decompositions).
- `resolve_attr_axis_size(op_helper, input_node, axis) -> int|Identifier`:
  shape-derived attribute that remains dynamic under `TractNNEF(dynamic_axes)`.

Eager vs. meta forward
- The exporter runs one forward pass to infer output structures. If your
  custom op lacks a CPU kernel, the exporter falls back to running on
  meta tensors. Provide `@custom_op.register_fake` for meta support.
- You can force the meta path with `export_model_to_nnef(..., skip_eager_forward=True)`.

Safety note
- Importing external modules (via explicit list, env var, or entry points)
  executes their top-level code in the current process. Only install and
  enable plugins you trust.
