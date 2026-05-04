"""Export the Flux-Schnell transformer (DiT denoiser) via torch-to-nnef.

Target: `black-forest-labs/FLUX.1-schnell` `FluxTransformer2DModel`.

Flux is a MM-DiT (multi-modal diffusion transformer): image and text tokens
share the same transformer stack through "double-stream" blocks that mix them
via cross-attention, followed by "single-stream" blocks over the joint token
sequence. 12B params at fp32 -> ~48GB, so first runs will need --skip-io-check
and a tiny sequence / image to even fit in memory.

Inputs (diffusers naming):
- hidden_states: (B, N_img_tokens, joint_attention_dim=3072) image latent tokens
- encoder_hidden_states: (B, N_text_tokens, joint_attention_dim) text tokens
- pooled_projections: (B, pooled_projection_dim=768) CLIP pooled
- timestep: (B,) float
- img_ids: (N_img_tokens, 3) rotary-embedding positions for image tokens
- txt_ids: (N_text_tokens, 3) positions for text tokens
- guidance (optional): (B,) classifier-free guidance scalar (schnell does NOT
  use guidance; FLUX.1-dev does).

Output: (B, N_img_tokens, joint_attention_dim) predicted velocity / noise.

Run:
    # Export-only (fast, validates graph emit):
    python transformer.py --skip-io-check --img-tokens 16 --txt-tokens 4
    # With tract I/O (very slow on this scale):
    python transformer.py --img-tokens 16 --txt-tokens 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torch_to_nnef import TractNNEF, export_model_to_nnef

HF_REPO = "black-forest-labs/FLUX.1-schnell"


class FluxTransformerWrapper(nn.Module):
    """Strip diffusers' kwargs plumbing; take a fixed positional set."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./flux_schnell_transformer.nnef.tgz")
    )
    parser.add_argument(
        "--img-tokens",
        type=int,
        default=16,
        help="Number of image tokens (real Flux uses H*W/4 at VAE latent res)",
    )
    parser.add_argument(
        "--txt-tokens",
        type=int,
        default=4,
        help="Number of text tokens from T5 (real Flux uses 512 or 256)",
    )
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (required for non-trivial sizes; "
        "the full 12B model exceeds memory of most setups).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Instantiate a tiny Flux transformer from config (random weights) "
        "instead of downloading the 12B schnell checkpoint. Same architecture, "
        "a few M params. Useful for exploring export without gated HF auth.",
    )
    args = parser.parse_args()

    from diffusers import FluxTransformer2DModel

    if args.mini:
        print("Instantiating mini FluxTransformer2DModel (random weights)")
        # Real schnell: num_layers=19, num_single_layers=38, head_dim=128,
        # num_attention_heads=24, joint_attention_dim=4096,
        # pooled_projection_dim=768. We shrink every dim so the model is a
        # few M params while keeping the MM-DiT double-stream +
        # single-stream architecture identical.
        transformer = (
            FluxTransformer2DModel(
                patch_size=1,
                in_channels=64,
                num_layers=2,
                num_single_layers=2,
                attention_head_dim=16,
                num_attention_heads=4,
                joint_attention_dim=64,
                pooled_projection_dim=32,
                guidance_embeds=False,
                axes_dims_rope=(8, 4, 4),
            )
            .to(torch.float32)
            .eval()
        )
    else:
        print(f"Loading FluxTransformer2DModel from {HF_REPO}")
        transformer = FluxTransformer2DModel.from_pretrained(
            HF_REPO, subfolder="transformer", torch_dtype=torch.float32
        )
    pipeline = FluxTransformerWrapper(transformer).eval()

    cfg = transformer.config
    joint_dim = cfg.joint_attention_dim  # 3072 for schnell
    pooled_dim = cfg.pooled_projection_dim  # 768
    print(f"joint_attention_dim={joint_dim} pooled_projection_dim={pooled_dim}")

    hidden_states = torch.randn(
        1, args.img_tokens, joint_dim, dtype=torch.float32
    )
    encoder_hidden_states = torch.randn(
        1, args.txt_tokens, joint_dim, dtype=torch.float32
    )
    pooled_projections = torch.randn(1, pooled_dim, dtype=torch.float32)
    timestep = torch.tensor([10.0], dtype=torch.float32)
    # Rotary positions: 3D (axis, h, w) format. Zero works for a smoke test
    # (breaks spatial structure but keeps all shapes correct).
    img_ids = torch.zeros(args.img_tokens, 3, dtype=torch.float32)
    txt_ids = torch.zeros(args.txt_tokens, 3, dtype=torch.float32)

    with torch.no_grad():
        out = pipeline(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
        )
    print(f"PyTorch output shape: {tuple(out.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=pipeline,
        args=(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
        ),
        file_path_export=args.out,
        inference_target=TractNNEF(
            version=tract_version,
            check_io=check_io,
            # tract 0.22.1's native SDPA chokes on Flux's RoPE'd Q/K shapes
            # ("Undetermined symbol"); fall back to the primitive fragment.
            reify_sdpa_operator=False,
        ),
        input_names=[
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
        ],
        output_names=["noise_pred"],
        debug_bundle_path=Path("./debug_flux_transformer.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
