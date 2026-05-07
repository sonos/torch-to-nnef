# Exporting JIT-only models

Some PyTorch models ship as a TorchScript artifact (a single `.jit` /
`.pt` file produced by `torch.jit.save`) without their training-time
Python source on the import path. The canonical example is
`silero-vad`: the JIT carries qualified type names like
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

## The easy path: pass the JIT directly

`export_model_to_nnef` auto-detects `torch.jit.ScriptModule` inputs and
applies the JIT-only export hardening chain internally. The trivial
case is just:

```python
import torch
from torch_to_nnef import TractNNEF, export_model_to_nnef

inner = torch.jit.load("model.jit").eval()
example_inputs = (x, state)

export_model_to_nnef(
    model=inner,
    args=example_inputs,
    file_path_export="model.nnef.tgz",
    inference_target=TractNNEF(version=TractNNEF.latest_version()),
    input_names=["x", "state"],
    output_names=["prob", "new_state"],
)
```

A log line confirms the auto-harden ran. Each pass in the chain is a
no-op on graphs that don't carry the relevant pattern, so the wrapper
is safe on any ScriptModule.

If you want fine-grained control (per-pass diagnostics, custom
ordering, partial chain for debugging), opt out and call the helper
yourself:

```python
from torch_to_nnef import (
    TractNNEF,
    export_model_to_nnef,
    harden_jit_for_export,
)

diagnostics: dict[str, object] = {}
model = harden_jit_for_export(
    inner, example_inputs, diagnostics=diagnostics
)
# `diagnostics` now holds per-pass fold counts and the freeze flag.

export_model_to_nnef(
    model=model,
    args=example_inputs,
    file_path_export="model.nnef.tgz",
    inference_target=TractNNEF(version=TractNNEF.latest_version()),
    auto_harden_jit=False,  # already hardened above
    input_names=["x", "state"],
    output_names=["prob", "new_state"],
)
```

`args` to `harden_jit_for_export` is the same shape as
`export_model_to_nnef(..., args=...)` (forward arguments only); the
helper prepends the `self` receiver internally.

For the lowest-level entry, the individual passes are exposed too.
See the next section.

> **torch version**: tested on torch 2.11.0. The data-dependent If fold
> calls `torch._C._jit_interpret_graph`, an undocumented internal API
> exposed since torch 1.10. Earlier 2.x should work; only 2.11.0 is
> CI-gated.

## The chain

```python
import torch
from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.torch_graph import (
    fold_constant_ifs,
    fold_constant_scalar_arithmetic,
    fold_data_dependent_ifs,
    fold_tuple_index_through_tuple_construct,
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

# Rewrite `prim::TupleIndex(prim::TupleConstruct(...), k)` to the k-th
# tuple input. Inlined JIT graphs surface this when call-site `(h, c)`
# tuples are consumed via positional indexing.
fold_tuple_index_through_tuple_construct(inner.graph)

# `prim::data` is `.data` access on a tensor (autograd detach). It is a
# no-op in inference; the parser doesn't handle it, so we elide.
strip_prim_data(inner.graph)

# Drop any remaining prim::If whose one branch is purely a
# RaiseException (PyTorch's compiled-in dim-check assertions).
strip_assertion_ifs(inner.graph)

# Specialize remaining `prim::If` nodes whose conditions are
# data-dependent (e.g. `nn.LSTMCell`'s `if input.dim() == 1: ...`) by
# evaluating the condition under the user's example inputs and inlining
# the chosen branch. PyTorch's JIT shape analysis does not propagate
# through tensor-producing Ifs, so this catches what the size folds miss.
fold_data_dependent_ifs(inner.graph, [inner, *example_inputs])

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
- **`fold_tuple_index_through_tuple_construct`**: Rewrites
  `prim::TupleIndex(tuple_const, k)` to the k-th input of a direct
  `prim::TupleConstruct`. Inlined call-site `(h, c)` tuples consumed via
  positional indexing leave behind this chain; t2n's parser knows
  `TupleConstruct` and `TupleUnpack` but not `TupleIndex`.
- **`strip_assertion_ifs`**: Drops `prim::If` whose one branch is purely
  a `RaiseException`. Picks up assertions that depend on values that
  remain symbolic at trace time.
- **`fold_data_dependent_ifs`**: Evaluates each remaining `prim::If`
  condition by re-executing the graph with the example inputs through
  `torch._C._jit_interpret_graph`, then inlines the chosen branch.
  Catches runtime dim/shape checks (notably `nn.LSTMCell`'s
  `if input.dim() == 1: ...`) that survive the size folds because their
  output is a tensor and JIT shape analysis does not propagate through
  tensor-producing Ifs.

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
(Silero-VAD), but production JITs sometimes use IR constructs that
are not yet covered here:

- Other `prim::*` constructs that survive Phase 1 inlining (rare). If
  your model trips one, please open an issue with a minimal repro.

## Why the standalone constant-fold passes exist

Running `torch._C._jit_pass_constant_propagation` on a graph that has
been through `inline_unresolvable_submodules` + `replace_size_calls_with_constants`
(without first running `torch.jit.freeze`) trips a c10 INTERNAL ASSERT
on torch 2.11 (`n->owningGraph() == this && n->inBlockList()`; on older
torch the same crash surfaced as `setInsertPoint`). The crash terminates
the interpreter with `libc++abi: terminating`, so a wrapping
`try/except` does not save you.

The standalone passes (`fold_constant_scalar_arithmetic`, `fold_constant_ifs`)
only walk the specific node kinds we need, never invoke the upstream
constant-propagation machinery, and stay safe under that pre-condition.

`harden_jit_for_export(model, args)` defaults to `freeze=True`, which
sidesteps the issue entirely because `torch.jit.freeze` already
constant-folds module attributes before our chain runs. The standalone
passes still matter for callers who pass `freeze=False`, or for graphs
that cannot be frozen.
