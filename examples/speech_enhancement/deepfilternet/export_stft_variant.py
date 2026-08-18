"""Export DeepFilterNet 3 with native `torch.fft.irfft` on the synthesis side.

Variant B of the example. Identical to `export.py` except for one
substitution: `ExportableStreamingMinimalTorchDF.frame_synthesis` is
patched to call `torch.fft.irfft` instead of `torch.einsum("fi,fij->j",
x, self.irfft_matrix) * self.fft_size`.

The analysis side already uses `torch.fft.rfft` in grazder's code (the
matmul variant there is commented out), so the only matmul-FFT remaining
in variant A lives on the synthesis side. Replacing it lets the NNEF
graph emit `tract_core_fft` for the inverse transform, which is what
tract's native FFT machinery actually optimises.

Why this is a one-method patch and not a model rewrite:

- DFN's streaming inner module runs **one frame at a time**, so no
  per-buffer overlap-add op (`col2im` / `F.fold`) is needed: the
  overlap-add already happens between frames via `synthesis_mem`
  passthrough (`output = x_first + synthesis_mem`).
- `torch.fft.irfft` directly replaces the einsum; the window multiply
  and the cumulative-mem fold-in stay identical.

Run:
    python export_stft_variant.py --out deepfilternet3_stft.nnef.tgz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TORCH_DF_PATH = _HERE / "_torchDF_clone" / "torchDF"
if not _TORCH_DF_PATH.exists():
    raise SystemExit(f"missing {_TORCH_DF_PATH}; run ./bootstrap.sh first")
sys.path.insert(0, str(_TORCH_DF_PATH))


def _patch_torchaudio_audio_meta_data() -> None:
    """See export.py for context: stubs the removed AudioMetaData class."""
    # pylint: disable=import-outside-toplevel
    import types

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
from torch import Tensor  # noqa: E402
from torch_df_streaming_minimal import (  # noqa: E402
    ExportableStreamingMinimalTorchDF,
    TorchDFMinimalPipeline,
)

from torch_to_nnef import TractNNEF, export_model_to_nnef  # noqa: E402


def _native_irfft_frame_synthesis(
    self: ExportableStreamingMinimalTorchDF,
    x: Tensor,
    synthesis_mem: Tensor,
) -> tuple[Tensor, Tensor]:
    """Drop-in replacement for `frame_synthesis` using `torch.fft.irfft`.

    Original grazder body (see `torch_df_streaming_minimal.py`):

        x = (
            torch.einsum("fi,fij->j", x, self.irfft_matrix)
            * self.fft_size
            * self.window
        )

    Here `x` is the rank-2 real layout of a single half-spectrum frame
    (shape `(F, 2)` where `F = window_size // 2 + 1`). We rebuild a
    complex tensor and let `torch.fft.irfft` do the inverse transform;
    the `* self.fft_size` factor disappears because PyTorch's `irfft`
    already includes the `1/N` normalisation that the einsum form was
    cancelling. The window multiply stays the same.
    """
    # `torch.view_as_complex` tags the (..., 2) real layout as complex64
    # without changing the IR shape; t2n's `fft_irfft` handler detects
    # that layout via `input_node.shape[-1] == 2` and adjusts its
    # complex-axis / FFT-axis indexing accordingly. This is the path
    # that exercises a native `torch.fft.irfft` end-to-end (vs. the
    # matmul-iFFT in variant A).
    x_complex = torch.view_as_complex(x.contiguous())
    x_time = torch.fft.irfft(x_complex, n=self.window_size)
    x_windowed = x_time * self.window
    x_first, x_second = torch.split(
        x_windowed,
        [self.frame_size, self.window_size - self.frame_size],
    )
    output = x_first + synthesis_mem
    return output, x_second.view(self.window_size - self.frame_size)


def build_streaming_model_b():
    """Build variant-B per-frame streaming model (native irfft).

    Mirrors `build_streaming_model()` in `export.py` but patches the
    inner module's `frame_synthesis` to use `torch.fft.irfft` instead
    of the matmul-iFFT. The pipeline is built only to fetch the
    pretrained weights + the initial state; only the inner module is
    exported.
    """
    pipeline = TorchDFMinimalPipeline().eval()
    inner = pipeline.torch_streaming_model
    # pylint: disable-next=no-value-for-parameter
    bound = _native_irfft_frame_synthesis.__get__(inner, inner.__class__)
    inner.frame_synthesis = bound  # type: ignore[method-assign]
    states = tuple(s.detach().clone() for s in pipeline.states)
    input_frame = torch.zeros(pipeline.hop_size, dtype=torch.float32)
    return (
        inner,
        input_frame,
        states,
        list(pipeline.input_names),
        list(pipeline.output_names),
    )


def verify_parity_against_a(
    model_b,
    input_frame: Tensor,
    states_b,
    atol: float = 5e-3,
    rtol: float = 1e-2,
) -> None:
    """Per-frame parity: B vs unmodified A on the same frame + state.

     The two are not bitwise identical by construction. grazder's
     matmul-iFFT uses ``irfft_matrix = torch.linalg.pinv(rfft_matrix)``
    : a per-frequency least-squares pseudo-inverse. ``torch.fft.irfft``
     is the analytic inverse DFT. They agree up to the pinv's numerical
     accuracy: typical max abs diff on the enhanced audio frame is well
     below auditory thresholds, but above the ~1e-6 drift you'd see from
     two implementations of the same algorithm.

     A failure: much larger diff, or NaN: would suggest a real bug
     (e.g. a missing window multiply or a transpose). The thresholds
     are sized to catch that without firing on the expected pinv drift.
    """
    pipeline_a = TorchDFMinimalPipeline().eval()
    inner_a = pipeline_a.torch_streaming_model
    states_a = tuple(s.detach().clone() for s in pipeline_a.states)
    with torch.no_grad():
        y_a = inner_a(input_frame, *states_a)[0]
        y_b = model_b(input_frame, *states_b)[0]
    diff = (y_a - y_b).abs().max().item()
    if not torch.allclose(y_a, y_b, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"variant B diverges from A: max abs diff = {diff:.3e} "
            f"(atol={atol}, rtol={rtol}). Within ~1e-3 is expected (pinv "
            f"approximation); much larger is a real bug: probably a "
            f"window-multiply factor or a missing transpose."
        )
    print(
        f"parity vs variant A (per-frame): "
        f"max abs diff = {diff:.3e} (pinv drift)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./deepfilternet3_stft.nnef.tgz"),
    )
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument("--skip-io-check", action="store_true")
    parser.add_argument("--skip-parity-check", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    model, input_frame, states, input_names, output_names = (
        build_streaming_model_b()
    )
    print(
        f"streaming model params: {sum(p.numel() for p in model.parameters())}"
    )
    print(
        f"frame size: {input_frame.shape[-1]} samples ; states: {len(states)}"
    )

    if not args.skip_parity_check:
        verify_parity_against_a(model, input_frame, states)

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
        debug_bundle_path=Path("./debug_deepfilternet3_stft.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
