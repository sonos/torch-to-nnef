"""Export the SD 1.5 UNet (noise predictor) via torch-to-nnef.

Target: `runwayml/stable-diffusion-v1-5` UNet (`UNet2DConditionModel`).

Inputs:
- `sample`: (B, 4, H, W) noisy latent
- `timestep`: scalar float (broadcast over batch via sinusoidal embedding)
- `encoder_hidden_states`: (B, L, 768) CLIP text embedding

Output:
- `noise_pred`: (B, 4, H, W) predicted noise

Size: ~860M params (~3.4 GB fp32). The full 64x64 latent + 77-token text path
is heavy; start at 8x8 latent + 10 tokens for a first smoke run, then scale up.

Run:
    python unet.py --skip-io-check             # fastest, just emit NNEF
    python unet.py                             # with tract I/O check (slow)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef

HF_REPO = "runwayml/stable-diffusion-v1-5"


class UNetWrapper(nn.Module):
    """Flatten the UNet2DConditionModel API to three positional tensors."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet.eval()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./sd15_unet.nnef.tgz")
    )
    parser.add_argument("--latent-h", type=int, default=8)
    parser.add_argument("--latent-w", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (very slow on UNet; use to verify "
        "the NNEF graph writes out at all).",
    )
    args = parser.parse_args()

    print(f"Loading UNet from {HF_REPO}")
    unet = UNet2DConditionModel.from_pretrained(
        HF_REPO, subfolder="unet", torch_dtype=torch.float32
    )
    pipeline = UNetWrapper(unet).eval()

    sample = torch.randn(
        1, 4, args.latent_h, args.latent_w, dtype=torch.float32
    )
    timestep = torch.tensor(10.0, dtype=torch.float32)
    encoder_hidden_states = torch.randn(
        1, args.seq_len, 768, dtype=torch.float32
    )

    with torch.no_grad():
        out = pipeline(sample, timestep, encoder_hidden_states)
    print(f"PyTorch output shape: {tuple(out.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=pipeline,
        args=(sample, timestep, encoder_hidden_states),
        file_path_export=args.out,
        inference_target=TractNNEF(version=tract_version, check_io=check_io),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        debug_bundle_path=Path("./debug_unet.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
