"""Export DeepFilterNet 3 (waveform -> enhanced waveform) to NNEF.

The official DeepFilterNet PyTorch model (`DfNet`) is frequency-domain only:
STFT, ERB feature extraction, complex deep-filter coefficient processing,
and iSTFT all live in `libDF` (Rust) outside the model. To get a single
waveform-in / waveform-out graph, this example uses @grazder's pure-torch
reimplementation (`torchDF/torch_df_streaming_minimal.py`), which packs the
full pipeline -- STFT, features, DfNet, gains, deep filter, iSTFT -- into
one `nn.Module`.

Variant A (this file, default) is grazder's model **verbatim**. Its STFT
and iSTFT are hand-rolled as matrix multiplies with precomputed FFT
matrices for ONNX compatibility. That keeps the apples-to-apples ONNX
baseline meaningful (both formats emit dense matmul-FFT), at the cost of
not exercising tract's native `tract_core_stft`.

Variant B (see `export_stft_variant.py`, if present) swaps the matmul-FFT
for `torch.stft` + a decomposed iSTFT (`fft_irfft` + window + `col2im`
overlap-add). Both variants land in NNEF; only A round-trips cleanly
through ONNX. The bench script reports per-frame latency for all three.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# grazder's `torch_df_streaming_minimal` lives in `torchDF/` inside the fork;
# that subdir is not pip-installable on its own, so bootstrap.sh clones the
# repo and we add the path here. Re-run bootstrap.sh if the import fails.
_HERE = Path(__file__).resolve().parent
_TORCH_DF_PATH = _HERE / "_torchDF_clone" / "torchDF"
if not _TORCH_DF_PATH.exists():
    raise SystemExit(f"missing {_TORCH_DF_PATH}; run ./bootstrap.sh first")
sys.path.insert(0, str(_TORCH_DF_PATH))


def _patch_torchaudio_audio_meta_data() -> None:
    """Stub `torchaudio.backend.common.AudioMetaData` for modern torchaudio.

    DeepFilterNet 0.5.6 (latest PyPI release) still imports
    `from torchaudio.backend.common import AudioMetaData`, which was removed
    in torchaudio >= 2.7. The class is only used for type hints and as the
    return type of `torchaudio.info(...)` inside `df.io.load_audio` -- a
    function we never call (the example feeds dummy in-memory waveforms).
    A no-op stub is enough to keep the chain importable.
    """
    # pylint: disable=import-outside-toplevel
    import types

    import torchaudio

    if "AudioMetaData" in dir(torchaudio):
        return
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # noqa: D401
        def __init__(self, *args, **kwargs) -> None:
            pass

    common.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
    backend.common = common  # type: ignore[attr-defined]
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules.setdefault("torchaudio.backend.common", common)


_patch_torchaudio_audio_meta_data()

import torch  # noqa: E402
from torch_df_streaming_minimal import TorchDFMinimalPipeline  # noqa: E402

from torch_to_nnef import TractNNEF, export_model_to_nnef  # noqa: E402


def build_streaming_model():
    """Load the pretrained DFN3 per-frame streaming model + its initial state.

    `TorchDFMinimalPipeline.torch_streaming_model` is the
    `ExportableStreamingMinimalTorchDF` instance that operates on one audio
    frame at a time, threading 13 state tensors in and out. That's the
    *deployable* shape of DFN3: a Rust runtime (tract, pulse-mode) or a
    Python caller manages the per-frame loop and carries the state between
    invocations.

    Returns `(model, input_frame, states, input_names, output_names)` so
    the caller can wire the trace without re-deriving any of those.
    """
    pipeline = TorchDFMinimalPipeline().eval()
    model = pipeline.torch_streaming_model
    states = tuple(s.detach().clone() for s in pipeline.states)
    input_frame = torch.zeros(pipeline.hop_size, dtype=torch.float32)
    return (
        model,
        input_frame,
        states,
        list(pipeline.input_names),
        list(pipeline.output_names),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./deepfilternet3.nnef.tgz"),
        help="Destination NNEF archive.",
    )
    parser.add_argument(
        "--tract-version",
        type=str,
        default=None,
        help="Override tract version (default: latest supported).",
    )
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract round-trip parity check.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    model, input_frame, states, input_names, output_names = (
        build_streaming_model()
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"streaming model params: {n_params}")
    print(
        f"frame size: {input_frame.shape[-1]} samples ; states: {len(states)}"
    )

    with torch.no_grad():
        outputs = model(input_frame, *states)
    enhanced_frame = outputs[0]
    print(f"PyTorch enhanced_frame shape: {tuple(enhanced_frame.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=model,
        args=(input_frame,) + states,
        file_path_export=args.out,
        inference_target=TractNNEF(version=tract_version, check_io=check_io),
        input_names=input_names,
        output_names=output_names,
        debug_bundle_path=Path("./debug_deepfilternet3.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
