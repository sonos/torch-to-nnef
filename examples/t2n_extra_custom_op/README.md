Custom t2n_extra operator example

This example shows how to define a simple custom op under the
`t2n_extra::` namespace, register an export handler for it, and export
an NNEF model that calls it.

Layout
- `t2n_custom/handlers.py`: declares `t2n_extra::my_relu` and registers
  its NNEF handler via `torch_to_nnef.op.extras.register`.
- `model.py`: a tiny `nn.Module` that calls `torch.ops.t2n_extra.my_relu`.
- `export.py`: runs export and auto-imports the handler module via
  `load_extra_op_modules=["t2n_custom.handlers"]`.

Run
- From the repo root:
  - `cd examples/t2n_extra_custom_op`
  - `python export.py`
  - This writes `my_relu.nnef.tgz` in the same directory.

Notes
- The handler maps `my_relu` to the standard NNEF `relu` op, but it
  could just as well emit a custom fragment call and return its key,
  e.g. `return ["my_company_relu"]`.
- Instead of passing `load_extra_op_modules`, you can also set the env var:
  `TORCH_TO_NNEF_EXTRA_MODULES=t2n_custom.handlers python export.py`.

