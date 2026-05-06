# Exporting JIT-only models

Some PyTorch models ship as a TorchScript artifact (a single `.jit` /
`.pt` file produced by `torch.jit.save`) without their training-time
Python source on the import path. Examples include `silero-vad` and
`funasr/fsmn-vad`: the JIT carries qualified type names like
`__torch__.vad.model.vad_annotator.SileroVadBlock`, but importing
`vad.model.vad_annotator` raises `ModuleNotFoundError` on a normal
install.

`torch_to_nnef`'s recursive parser identifies the class behind every
`prim::CallMethod` via `importlib.import_module(qualname.module_path)`.
When that import fails, the parser cannot recurse, so vanilla
`export_model_to_nnef(jit_module, ...)` blows up before reaching the op
handlers.

The `torch_to_nnef.torch_graph` module ships a chain of opt-in passes
that reshape the JIT graph in place so that, after the chain, every
`prim::CallMethod` left in the graph targets an importable class
(`torch.nn.*`) and every other unsupported construct collapses into ops
that the standard parser handles.

## The chain

```python
import torch
from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.torch_graph import (
    fold_constant_ifs,
    fold_constant_scalar_arithmetic,
    inline_unresolvable_submodules,
    replace_size_calls_with_constants,
    strip_assertion_ifs,
    strip_prim_data,
)

inner = torch.jit.load("model.jit").eval()
example_inputs = (x, state)  # tensors with concrete shapes

# Phase 1: inline only the JIT submodules whose source class is not
# importable; keep torch.nn.* boundaries so existing module-level
# extractors (LSTM, GRU, RNN) still fire.
inline_unresolvable_submodules(inner.graph, inner)
torch._C._jit_pass_dce(inner.graph)

# Phase 2: fold aten::dim/size/len/numel against the example inputs'
# shapes via complete_shape_analysis. Once size queries collapse,
# scalar arithmetic and prim::If conditions can fold too.
replace_size_calls_with_constants(inner.graph, [inner, *example_inputs])

# Standalone constant-fold replacement for `_jit_pass_constant_propagation`.
# Walks aten::eq/ne/lt/le/gt/ge, aten::__not__, aten::__contains__,
# aten::Bool/Int/Float when their operands are `prim::Constant`. We
# avoid the upstream pass because it has been observed to trip an
# internal `setInsertPoint` assertion on graphs that mix Phase 1
# inlined submodules and Phase 2 size-fold constants.
fold_constant_scalar_arithmetic(inner.graph)

# Drop prim::If whose condition is now a constant boolean: keep the
# chosen branch's body, destroy the other.
fold_constant_ifs(inner.graph)

# `prim::data` is `.data` access on a tensor (autograd detach). It is a
# no-op in inference; the parser doesn't handle it, so we elide.
strip_prim_data(inner.graph)

# Drop any remaining prim::If whose one branch is purely a
# RaiseException (PyTorch's compiled-in dim-check assertions).
strip_assertion_ifs(inner.graph)

torch._C._jit_pass_dce(inner.graph)

# Standard t2n export from this point on.
export_model_to_nnef(
    model=inner,
    args=example_inputs,
    file_path_export="model.nnef.tgz",
    inference_target=TractNNEF(version=TractNNEF.latest_version()),
    input_names=["x", "state"],
    output_names=["prob", "new_state"],
)
```

## Why each pass exists

- **`inline_unresolvable_submodules`**: Without it, recursion into
  non-importable JIT submodules raises `ModuleNotFoundError`. Inlining
  exposes their bodies in the parent graph; importable submodules
  (`torch.nn.*`) remain as `prim::CallMethod` so existing module
  extractors continue to handle them.
- **`replace_size_calls_with_constants`**: After Phase 1, the graph
  often contains `prim::If` nodes gated on runtime size queries (e.g.
  `if input.dim() == 2:`). Folding the size queries to constants is the
  precondition for collapsing those branches.
- **`fold_constant_scalar_arithmetic`** + **`fold_constant_ifs`**:
  Together they simulate `_jit_pass_constant_propagation` on the
  bool/int arithmetic that gates the surviving `prim::If` nodes. Used
  instead of the upstream pass because of the assertion crash noted
  above.
- **`strip_prim_data`**: Replaces `prim::data(t)` with `t`; the parser
  doesn't have a handler for that op kind.
- **`strip_assertion_ifs`**: Drops `prim::If` whose one branch is purely
  a `RaiseException`. Picks up assertions that depend on values that
  remain symbolic at trace time.

## Module-level extractor preservation

The chain leaves importable `torch.nn.*` calls intact. That matters for:

- `nn.LSTM` / `nn.GRU` / `nn.RNN`: handled by the dedicated extractors
  in `op/custom_extractors/rnn.py` (NNEF custom fragments).
- `nn.LSTMCell`: decomposed to primitive NNEF ops by the
  `LSTMCellExtractor`. The decomposition body lives in
  `op/aten/rnn.py::emit_lstm_cell_decomposition` and is also wired to
  the `aten::lstm_cell` aten handler, so an inlined JIT graph that
  exposes the underlying `_VF.lstm_cell` directly produces the same
  NNEF ops as the module-level path.

## Limitations and known gaps

The chain handles the common patterns in real-world JIT artifacts
(Silero-VAD, FunASR), but production JITs sometimes use IR constructs
that are not yet covered here:

- `prim::TupleIndex(state, k)`: indexing a tuple by a constant k. Today
  the parser does not have a handler. Workaround: avoid tuple-typed
  state at the model boundary, or pre-process the JIT graph to replace
  `TupleIndex` with `aten::select` on the underlying tensor.
- Other `prim::*` constructs that survive Phase 1 inlining (rare). If
  your model trips one, please open an issue with a minimal repro.

## Pytest gotcha

PyTorch's JIT type cache leaks between scripted modules in the same
process. After compiling several small `nn.Module`s in a pytest run and
then inlining a non-importable submodule into a JIT artifact,
`_jit_pass_constant_propagation` and `_jit_pass_peephole` can trip an
internal `setInsertPoint` assertion. The standalone fold passes
(`fold_constant_scalar_arithmetic`, `fold_constant_ifs`) sidestep this,
which is one reason they exist as user-callable helpers rather than as
a transparent wrapper around the upstream pass.
