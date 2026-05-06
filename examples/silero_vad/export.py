"""Export Silero-VAD's `silero_vad.jit` straight to NNEF.

Demonstrates the JIT-only model export chain end-to-end on a real-world
model: Silero loads as a `torch.jit.ScriptModule` whose Python source is
not on the import path, so the standard tracer cannot reach it. The
chain below specializes the JIT graph for the example inputs and hands
the result to `export_model_to_nnef`.
"""

from pathlib import Path

import torch
from silero_vad import load_silero_vad

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

raw_model = load_silero_vad()._model.eval()
x = torch.randn(1, 576, dtype=torch.float32) * 0.1
state = torch.zeros(2, 1, 128, dtype=torch.float32)

# `torch.jit.freeze` resolves CallMethod/CallFunction/GetAttr in one go.
model = torch.jit.freeze(raw_model)
torch._C._jit_pass_dce(model.graph)

# JIT-only specialization chain. Each pass is a no-op on already-clean
# graphs, so the order is safe to apply unconditionally.
inline_unresolvable_submodules(model.graph, model)
torch._C._jit_pass_dce(model.graph)
replace_size_calls_with_constants(model.graph, [model, x, state])
fold_constant_scalar_arithmetic(model.graph)
fold_constant_ifs(model.graph)
fold_tuple_index_through_tuple_construct(model.graph)
strip_prim_data(model.graph)
strip_assertion_ifs(model.graph)
fold_data_dependent_ifs(model.graph, [model, x, state])
torch._C._jit_pass_dce(model.graph)

out_path = Path("silero_vad.nnef.tgz")
export_model_to_nnef(
    model=model,
    args=(x, state),
    file_path_export=out_path,
    inference_target=TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=True,
    ),
    input_names=["x", "state"],
    output_names=["prob", "new_state"],
)
print(f"exported {out_path.absolute()}")
