"""Export CEVA's DPDFNet-2 (16 kHz) as a single per-frame NNEF artifact.

Wraps the upstream `DPDFNet` NN with in-graph STFT analysis +
iFFT synthesis so the deploy is a *single* artifact running on tract
(no libDF / DSP companion). Per-frame I/O:

    in : audio_frame[hop_size=160]            (float32)
       + stft_buf[win_len=320]                (float32, rolling input window)
       + nn_state[45424]                      (float32, flat DPDFNet state)
       + ola_buf[win_len=320]                 (float32, overlap-add buffer)

    out: enhanced_frame[hop_size=160]         (float32)
       + stft_buf'[win_len=320]
       + nn_state'[45424]
       + ola_buf'[win_len=320]

Caller (tract or any host) threads the state across frames. Same shape
as DFN3 variant B in `../deepfilternet/export_stft_variant.py`.

Vorbis window, n_fft=win_len=320, hop_size=160 (20 ms frames @ 16 kHz).
DPDFNet expects `spec` of shape `(1, 1, freq_bins=161, 2)` in t2n's
view-tagged complex convention -- that matches the rfft output layout
out-of-the-box.

Variant B path: native `torch.fft.irfft` on synthesis (sized via the
`fft_irfft` t2n handler). The complex-IR view-tagging fix landed
earlier in this branch makes this work end-to-end on tract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CLONE = HERE / "_dpdfnet_clone"
if not CLONE.exists():
    raise SystemExit(f"missing {CLONE}; run ./bootstrap.sh first")
sys.path.insert(0, str(CLONE))

from onnx_model.dpdfnet import DPDFNet, correct_state_dict  # noqa: E402
from onnx_model.layers import convert_grouped_linear_to_einsum  # noqa: E402

from torch_to_nnef.export import export_model_to_nnef  # noqa: E402
from torch_to_nnef.inference_target import TractNNEF  # noqa: E402

SAMPLE_RATE = 16_000
N_FFT = 320
HOP_SIZE = 160
FREQ_BINS = N_FFT // 2 + 1  # 161
NN_STATE_SIZE = 45_424


def _build_dpdfnet2() -> DPDFNet:
    """Build DPDFNet-2 (matches CEVA's `export_dpdfnet_to_onnx.py`)."""
    model = DPDFNet(
        conv_kernel_inp=(3, 3),
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type="subpixel",
        group_linear_type="loop",
        point_wise_type="cnn",
        separable_first_conv=True,
        dprnn_num_blocks=2,
    )
    return model


def _vorbis_window(win_len: int) -> torch.Tensor:
    """Vorbis window: `sin(pi/2 * sin^2(pi*(n+0.5)/win_len))`.

    Matches the constant in `real_time_demo.py` and the
    `vorbis_window` helper in the model package; baked as a model
    buffer so it ships with the NNEF artifact.
    """
    n = torch.arange(win_len, dtype=torch.float32)
    sin_inner = torch.sin(0.5 * torch.pi * (n + 0.5) / (win_len / 2))
    return torch.sin(0.5 * torch.pi * sin_inner * sin_inner)


class StreamingDPDFNet(torch.nn.Module):
    """Per-frame streaming DPDFNet: WAV in -> WAV out, state threaded.

    All DSP (windowing, rfft, irfft, OLA) lives in this module so the
    NNEF artifact is self-contained. The inner DPDFNet handles its own
    state via the flat `nn_state` tensor.
    """

    def __init__(self, inner: DPDFNet) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("window", _vorbis_window(N_FFT))

    def forward(
        self,
        audio_frame: torch.Tensor,  # (HOP_SIZE,)
        stft_buf: torch.Tensor,  # (N_FFT,)
        nn_state: torch.Tensor,  # (NN_STATE_SIZE,)
        ola_buf: torch.Tensor,  # (N_FFT,)
    ):
        # 1. Update rolling STFT buffer: drop oldest hop_size, append new.
        new_stft_buf = torch.cat([stft_buf[HOP_SIZE:], audio_frame], dim=0)

        # 2. Window + rfft -> view-tagged complex (1, 1, freq_bins, 2).
        windowed = new_stft_buf * self.window
        spec_c = torch.fft.rfft(windowed)  # complex (freq_bins,)
        spec = torch.view_as_real(spec_c)  # (freq_bins, 2)
        spec = spec.view(1, 1, FREQ_BINS, 2)

        # 3. DPDFNet inference.
        spec_e, new_nn_state = self.inner(spec, nn_state)

        # 4. iSTFT: spec_e (1, 1, freq_bins, 2) -> frame (n_fft,).
        spec_e_flat = spec_e.view(FREQ_BINS, 2)
        spec_e_c = torch.view_as_complex(spec_e_flat.contiguous())
        frame = torch.fft.irfft(spec_e_c, n=N_FFT)
        frame = frame * self.window

        # 5. OLA: shift left by hop_size, zero-pad tail, add frame.
        shifted = torch.cat(
            [ola_buf[HOP_SIZE:], torch.zeros(HOP_SIZE, dtype=ola_buf.dtype)],
            dim=0,
        )
        new_ola = shifted + frame

        # 6. Emit first hop_size samples; ola buffer carries the rest.
        enhanced_frame = new_ola[:HOP_SIZE]
        return enhanced_frame, new_stft_buf, new_nn_state, new_ola


def build_streaming_model(checkpoint_path: Path):
    """Load DPDFNet-2 weights and wrap with the streaming module."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        raw["state_dict"]
        if isinstance(raw, dict) and "state_dict" in raw
        else raw
    )
    inner = _build_dpdfnet2()
    inner.load_state_dict(correct_state_dict(state_dict), strict=True)
    convert_grouped_linear_to_einsum(inner)
    inner.eval()

    streaming = StreamingDPDFNet(inner).eval()

    audio_frame = torch.zeros(HOP_SIZE, dtype=torch.float32)
    stft_buf = torch.zeros(N_FFT, dtype=torch.float32)
    nn_state = inner.initial_state(dtype=torch.float32)
    ola_buf = torch.zeros(N_FFT, dtype=torch.float32)

    input_names = ["audio_frame", "stft_buf", "nn_state", "ola_buf"]
    output_names = [
        "enhanced_frame",
        "stft_buf_out",
        "nn_state_out",
        "ola_buf_out",
    ]
    return (
        streaming,
        (audio_frame, stft_buf, nn_state, ola_buf),
        input_names,
        output_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "dpdfnet2.nnef.tgz",
        help="Destination NNEF artifact path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=HERE / "_checkpoints" / "dpdfnet2.pth",
        help="Path to DPDFNet-2 .pth checkpoint (see bootstrap.sh).",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    model, example_inputs, input_names, output_names = build_streaming_model(
        args.checkpoint
    )

    # Sanity-check the wrapper before export.
    with torch.no_grad():
        outs = model(*example_inputs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DPDFNet streaming: params={n_params}")
    print(f"  audio_frame in: {tuple(example_inputs[0].shape)}")
    print(f"  enhanced_frame: {tuple(outs[0].shape)}")
    print(f"  nn_state size : {NN_STATE_SIZE}")

    target = TractNNEF(version=TractNNEF.latest_version(), check_io=True)
    export_model_to_nnef(
        model=model,
        args=example_inputs,
        file_path_export=args.out,
        inference_target=target,
        input_names=input_names,
        output_names=output_names,
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
