"""Produce grazder's reference ONNX export of the DFN3 streaming model.

Mirrors `_torchDF_clone/torchDF/model_onnx_export.py` (minimal flow) but
trimmed down so it works on torch 2.11 (the upstream import path
`torch.onnx._internal.jit_utils.GraphContext` is gone). Same
`torch.jit.script` + `torch.onnx.export` setup, same custom op symbolic
for `aten::fft_rfft`, no benchmark / simplify / inference helpers.

The exported model has the same input / output shape as ours (per-frame,
12 state tensors threaded by the caller), so it's directly comparable
against the t2n NNEF export on tract.

Run:
    python export_onnx_baseline.py --out deepfilternet3.onnx
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TORCH_DF_PATH = _HERE / "_torchDF_clone" / "torchDF"
if not _TORCH_DF_PATH.exists():
    raise SystemExit(f"missing {_TORCH_DF_PATH}; run ./bootstrap.sh first")
sys.path.insert(0, str(_TORCH_DF_PATH))


def _patch_torchaudio_audio_meta_data() -> None:
    """Stub `torchaudio.backend.common.AudioMetaData` for modern torchaudio."""
    import torchaudio

    if "AudioMetaData" in dir(torchaudio):
        return
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(self, *args, **kwargs) -> None:
            pass

    common.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
    backend.common = common  # type: ignore[attr-defined]
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules.setdefault("torchaudio.backend.common", common)


_patch_torchaudio_audio_meta_data()

import torch  # noqa: E402
from torch.onnx import symbolic_helper  # noqa: E402
from torch_df_streaming_minimal import TorchDFMinimalPipeline  # noqa: E402

OPSET_VERSION = 17


@symbolic_helper.parse_args("v", "i", "i", "s")
def _onnx_custom_rfft(g, X, n, dim, norm):
    """Symbolic for `aten::fft_rfft` -- mirror grazder's `custom_rfft`.

    DFN3's streaming model only ever calls `fft_rfft` on a rank-1 buffer
    (the windowed audio frame), so the only valid axis is 0. tract's
    ONNX loader rejects negative `axis` on `DFT`, so we hard-code the
    positive value rather than passing through the traced `dim=-1`.
    """
    return g.op("DFT", X, axis_i=0, onesided_i=1)


def _onnx_custom_identity(g, X):
    """Symbolic for `aten::view_as_real` -- pass-through."""
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./deepfilternet3.onnx"),
        help="Destination ONNX file.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    pipeline = TorchDFMinimalPipeline().eval()
    torch_df = pipeline.torch_streaming_model
    states = tuple(s.detach().clone() for s in pipeline.states)
    input_frame = torch.rand(pipeline.hop_size, dtype=torch.float32)
    input_features = (input_frame,) + states

    # Sanity-check the inference path before tracing.
    with torch.no_grad():
        torch_df(*input_features)

    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::fft_rfft",
        symbolic_fn=_onnx_custom_rfft,
        opset_version=OPSET_VERSION,
    )
    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::view_as_real",
        symbolic_fn=_onnx_custom_identity,
        opset_version=OPSET_VERSION,
    )

    scripted = torch.jit.script(torch_df)

    # `dynamo=False` keeps the legacy TorchScript-based exporter which is
    # the only path that accepts a `ScriptModule` (matches grazder's
    # original script). The default `dynamo=True` on torch 2.11 routes
    # through `torch.export` which rejects ScriptModule.
    torch.onnx.export(
        scripted,
        input_features,
        str(args.out),
        verbose=False,
        input_names=pipeline.input_names,
        output_names=pipeline.output_names,
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    print(f"ONNX exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
