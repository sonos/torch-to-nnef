"""Export the SD 1.5 VAE decoder (latent -> image) via torch-to-nnef.

Target: `runwayml/stable-diffusion-v1-5` VAE decoder. The decoder upsamples a
`(1, 4, 64, 64)` latent into a `(1, 3, 512, 512)` image. Its upsample blocks
use `F.interpolate` which lowers to NNEF ``resize`` -- the op that was the
historical blocker here. We export on fp32 CPU and run tract I/O check to
validate against PyTorch.

Run:
    python vae_decoder.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torch_to_nnef import TractNNEF, export_model_to_nnef

HF_REPO = "runwayml/stable-diffusion-v1-5"


class VaeDecoderWrapper(nn.Module):
    """Forward takes a latent, returns the decoded image."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae.eval()
        self.scaling_factor = float(
            getattr(vae.config, "scaling_factor", 0.18215)
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents / self.scaling_factor
        return self.vae.decode(x, return_dict=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./sd15_vae_decoder.nnef.tgz")
    )
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (very slow on full VAE; use to verify "
        "the NNEF graph writes out at all).",
    )
    args = parser.parse_args()

    from diffusers import AutoencoderKL

    print(f"Loading VAE from {HF_REPO}")
    vae = AutoencoderKL.from_pretrained(
        HF_REPO, subfolder="vae", torch_dtype=torch.float32
    )
    pipeline = VaeDecoderWrapper(vae).eval()

    latents = torch.randn(
        1, 4, args.latent_h, args.latent_w, dtype=torch.float32
    )
    with torch.no_grad():
        out = pipeline(latents)
    print(f"PyTorch output shape: {tuple(out.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=pipeline,
        args=latents,
        file_path_export=args.out,
        inference_target=TractNNEF(version=tract_version, check_io=check_io),
        input_names=["latents"],
        output_names=["image"],
        debug_bundle_path=Path("./debug_vae_decoder.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
