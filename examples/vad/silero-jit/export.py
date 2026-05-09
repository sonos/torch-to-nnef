"""Export Silero-VAD's `silero_vad.jit` straight to NNEF.

`export_model_to_nnef` auto-detects `torch.jit.ScriptModule` inputs and
applies `harden_jit_for_export` internally. Silero loads as a JIT
artifact whose Python source isn't on the import path, so without the
chain the standard recursive parser would raise `ModuleNotFoundError`
before reaching the op handlers; with the chain the export just works.

Pass `auto_harden_jit=False` if you want to drive the chain yourself
(e.g. to inspect per-pass fold counts via the `diagnostics` dict).
"""

from pathlib import Path

import torch
from silero_vad import load_silero_vad

from torch_to_nnef import TractNNEF, export_model_to_nnef

raw_model = load_silero_vad()._model.eval()
x = torch.randn(1, 576, dtype=torch.float32) * 0.1
state = torch.zeros(2, 1, 128, dtype=torch.float32)

out_path = Path("silero_vad.nnef.tgz")
export_model_to_nnef(
    model=raw_model,
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
