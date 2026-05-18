"""Export any CEVA DPDFNet variant as a single per-frame NNEF artifact.

Wraps the upstream DPDFNet NN with in-graph STFT analysis + iFFT
synthesis so the deploy is a *single* artifact running on tract (no
libDF / DSP companion). Per-frame I/O:

    in : audio_frame[hop_size]                (float32)
       + stft_buf[n_fft]                      (float32, rolling input)
       + nn_state[state_size]                 (float32, flat DPDFNet state)
       + ola_buf[n_fft]                       (float32, overlap-add buffer)

    out: enhanced_frame[hop_size]             (float32)
       + stft_buf'[n_fft]
       + nn_state'[state_size]
       + ola_buf'[n_fft]

The audio params (`sample_rate`, `n_fft`, `hop_size`, `state_size`) are
read directly off the loaded model and written into a sidecar JSON
manifest next to the NNEF artifact so downstream consumers (bench.py,
wav-cleaner-rs) don't need to hard-code shapes per variant.

Supported variants: 16 kHz hop=160 n_fft=320, 48 kHz hop=480
n_fft=960; `dprnn_num_blocks` differs per checkpoint:

    baseline             dprnn_num_blocks=0  (16 kHz)
    dpdfnet2             dprnn_num_blocks=2  (16 kHz)
    dpdfnet4             dprnn_num_blocks=4  (16 kHz)
    dpdfnet8             dprnn_num_blocks=8  (16 kHz)
    dpdfnet2_48khz_hr    dprnn_num_blocks=2  (48 kHz HR)
    dpdfnet8_48khz_hr    dprnn_num_blocks=8  (48 kHz HR)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CLONE = HERE / "_dpdfnet_clone"
if not CLONE.exists():
    raise SystemExit(f"missing {CLONE}; run ./bootstrap.sh first")
sys.path.insert(0, str(CLONE))

# ruff: noqa: E402, I001
# Imports below sit after a `sys.path.insert` for the cloned upstream
# repo, hence the file-level E402 / I001 silencing.
from onnx_model.dpdfnet import (
    DPDFNet,
    correct_state_dict as correct_state_dict_16k,
)
from onnx_model.dpdfnet_48khz_hr import (
    DPDFNet48HR,
    correct_state_dict as correct_state_dict_48k,
)
from onnx_model.layers import convert_grouped_linear_to_einsum

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF


@dataclass(frozen=True)
class VariantSpec:
    """Per-variant overrides for the DPDFNet constructor and HF checkpoint."""

    name: str
    cls: type
    correct_state_dict: callable
    dprnn_num_blocks: int
    sample_rate: int


VARIANTS: dict[str, VariantSpec] = {
    "baseline": VariantSpec(
        "baseline", DPDFNet, correct_state_dict_16k, 0, sample_rate=16_000
    ),
    "dpdfnet2": VariantSpec(
        "dpdfnet2", DPDFNet, correct_state_dict_16k, 2, sample_rate=16_000
    ),
    "dpdfnet4": VariantSpec(
        "dpdfnet4", DPDFNet, correct_state_dict_16k, 4, sample_rate=16_000
    ),
    "dpdfnet8": VariantSpec(
        "dpdfnet8", DPDFNet, correct_state_dict_16k, 8, sample_rate=16_000
    ),
    "dpdfnet2_48khz_hr": VariantSpec(
        "dpdfnet2_48khz_hr",
        DPDFNet48HR,
        correct_state_dict_48k,
        dprnn_num_blocks=2,
        sample_rate=48_000,
    ),
    "dpdfnet8_48khz_hr": VariantSpec(
        "dpdfnet8_48khz_hr",
        DPDFNet48HR,
        correct_state_dict_48k,
        dprnn_num_blocks=8,
        sample_rate=48_000,
    ),
}

# Constructor kwargs shared by every CEVA upstream variant. `n_fft`,
# `hop_length` etc. are left to the model's defaults (320/160 @ 16 kHz,
# 960/480 @ 48 kHz HR) so we don't have to track them here.
_COMMON_KWARGS = {
    "conv_kernel_inp": (3, 3),
    "conv_ch": 64,
    "enc_gru_dim": 256,
    "erb_dec_gru_dim": 256,
    "df_dec_gru_dim": 256,
    "enc_lin_groups": 32,
    "lin_groups": 16,
    "upsample_conv_type": "subpixel",
    "group_linear_type": "loop",
    "point_wise_type": "cnn",
    "separable_first_conv": True,
}


def _build_inner(spec: VariantSpec):
    """Instantiate the per-variant DPDFNet model class with shared kwargs."""
    return spec.cls(dprnn_num_blocks=spec.dprnn_num_blocks, **_COMMON_KWARGS)


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

    def __init__(self, inner, n_fft: int, hop_size: int) -> None:
        super().__init__()
        self.inner = inner
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.freq_bins = n_fft // 2 + 1
        self.register_buffer("window", _vorbis_window(n_fft))

    def forward(
        self,
        audio_frame: torch.Tensor,  # (hop_size,)
        stft_buf: torch.Tensor,  # (n_fft,)
        nn_state: torch.Tensor,  # (state_size,)
        ola_buf: torch.Tensor,  # (n_fft,)
    ):
        # 1. Update rolling STFT buffer: drop oldest hop_size, append new.
        new_stft_buf = torch.cat(
            [stft_buf[self.hop_size :], audio_frame], dim=0
        )

        # 2. Window + rfft -> view-tagged complex (1, 1, freq_bins, 2).
        windowed = new_stft_buf * self.window
        spec_c = torch.fft.rfft(windowed)  # complex (freq_bins,)
        spec = torch.view_as_real(spec_c)  # (freq_bins, 2)
        spec = spec.view(1, 1, self.freq_bins, 2)

        # 3. DPDFNet inference.
        spec_e, new_nn_state = self.inner(spec, nn_state)

        # 4. iSTFT: spec_e (1, 1, freq_bins, 2) -> frame (n_fft,).
        spec_e_flat = spec_e.view(self.freq_bins, 2)
        spec_e_c = torch.view_as_complex(spec_e_flat.contiguous())
        frame = torch.fft.irfft(spec_e_c, n=self.n_fft)
        frame = frame * self.window

        # 5. OLA: shift left by hop_size, zero-pad tail, add frame.
        shifted = torch.cat(
            [
                ola_buf[self.hop_size :],
                torch.zeros(self.hop_size, dtype=ola_buf.dtype),
            ],
            dim=0,
        )
        new_ola = shifted + frame

        # 6. Emit first hop_size samples; ola buffer carries the rest.
        enhanced_frame = new_ola[: self.hop_size]
        return enhanced_frame, new_stft_buf, new_nn_state, new_ola


def build_streaming_model(spec: VariantSpec, checkpoint_path: Path):
    """Load DPDFNet weights and wrap with the streaming module."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        raw["state_dict"]
        if isinstance(raw, dict) and "state_dict" in raw
        else raw
    )
    inner = _build_inner(spec)
    corrected = spec.correct_state_dict(state_dict)
    # 48 kHz HR checkpoints from HF carry leftover BatchNorm
    # `num_batches_tracked` counters that the streaming model class
    # doesn't have (it doesn't use BN). Drop them before strict load.
    corrected = {
        k: v
        for k, v in corrected.items()
        if not k.endswith("num_batches_tracked")
    }
    inner.load_state_dict(corrected, strict=True)
    convert_grouped_linear_to_einsum(inner)
    inner.eval()

    # Read audio params straight off the model: n_fft / hop / state_size
    # are computed from the constructor args, sample_rate is the one
    # value not exposed as an attribute (it's a constructor-only kwarg)
    # so we carry it on the variant spec.
    n_fft = int(inner.stft.n_fft)
    hop_size = int(inner.stft.hop)
    sample_rate = spec.sample_rate
    state_size = int(inner.state_size())

    streaming = StreamingDPDFNet(inner, n_fft, hop_size).eval()

    audio_frame = torch.zeros(hop_size, dtype=torch.float32)
    stft_buf = torch.zeros(n_fft, dtype=torch.float32)
    nn_state = inner.initial_state(dtype=torch.float32)
    ola_buf = torch.zeros(n_fft, dtype=torch.float32)

    manifest = {
        "variant": spec.name,
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_size": hop_size,
        "freq_bins": n_fft // 2 + 1,
        "state_size": state_size,
        "input_names": [
            "audio_frame",
            "stft_buf",
            "nn_state",
            "ola_buf",
        ],
        "output_names": [
            "enhanced_frame",
            "stft_buf_out",
            "nn_state_out",
            "ola_buf_out",
        ],
    }
    return streaming, (audio_frame, stft_buf, nn_state, ola_buf), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="dpdfnet2",
        help="DPDFNet variant to export (default: dpdfnet2).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination NNEF artifact path "
        "(defaults to <variant>.nnef.tgz next to this script).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the .pth checkpoint "
        "(defaults to _checkpoints/<variant>.pth; see bootstrap.sh).",
    )
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    checkpoint = args.checkpoint or (HERE / "_checkpoints" / f"{spec.name}.pth")
    out_path = args.out or (HERE / f"{spec.name}.nnef.tgz")
    if not checkpoint.exists():
        raise SystemExit(
            f"missing checkpoint at {checkpoint}; "
            f"run `./bootstrap.sh {spec.name}` first"
        )

    torch.manual_seed(0)
    model, example_inputs, manifest = build_streaming_model(spec, checkpoint)

    with torch.no_grad():
        outs = model(*example_inputs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DPDFNet streaming ({spec.name}): params={n_params}")
    print(
        f"  sample_rate={manifest['sample_rate']} Hz  "
        f"hop={manifest['hop_size']}  n_fft={manifest['n_fft']}"
    )
    print(f"  audio_frame in: {tuple(example_inputs[0].shape)}")
    print(f"  enhanced_frame: {tuple(outs[0].shape)}")
    print(f"  nn_state size : {manifest['state_size']}")

    target = TractNNEF(version=TractNNEF.latest_version(), check_io=True)
    export_model_to_nnef(
        model=model,
        args=example_inputs,
        file_path_export=out_path,
        inference_target=target,
        input_names=manifest["input_names"],
        output_names=manifest["output_names"],
    )
    print(f"Exported NNEF artifact to {out_path.absolute()}")

    manifest_path = out_path.with_suffix("").with_suffix(".json")
    if manifest_path.suffix != ".json":
        manifest_path = out_path.parent / (out_path.name + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote audio-params manifest to {manifest_path.absolute()}")


if __name__ == "__main__":
    main()
