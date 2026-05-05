"""Export the Sana transformer (linear-attention DiT) via torch-to-nnef.

Target: NVIDIA's Sana family
(e.g. ``Efficient-Large-Model/Sana_1600M_1024px_diffusers``)
``SanaTransformer2DModel``.

What's architecturally distinct from Flux/SD3 (already covered in this dir):

- Self-attention is **ReLU linear attention**, not softmax SDPA. The block is
  ``Q @ (Kᵀ @ V) / Q @ sum(K)`` (with ReLU on Q/K), so the export hits a
  different attention pattern through tract.
- FFN is **Mix-FFN**: linear -> depth-wise conv -> linear. Standard ``conv2d``
  with ``groups=channels`` but exercised inside an FFN.
- Cross-attention to the text encoder still uses softmax attention.

Inputs (diffusers naming):

- hidden_states: (B, in_channels, H, W) latent tokens (Sana uses DC-AE so
  H, W are tiny vs. SD's f8 latents).
- encoder_hidden_states: (B, N_text_tokens, caption_channels) Gemma-2 (or T5)
  encoded captions, projected via ``caption_projection`` inside the model.
- timestep: (B,) float.

Output: (B, out_channels, H, W) predicted velocity / noise.

Run:
    # Mini, fast:
    python transformer.py --mini --skip-io-check
    # Mini with check_io against tract:
    python transformer.py --mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import SanaTransformer2DModel
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef

HF_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


class SanaTransformerWrapper(nn.Module):
    """Strip diffusers' kwargs plumbing; take a fixed positional set."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./sana_transformer.nnef.tgz")
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=8,
        help="Spatial size H=W of the latent tokens (real Sana 1024px uses 32 "
        "via DC-AE f32 compression).",
    )
    parser.add_argument(
        "--txt-tokens",
        type=int,
        default=4,
        help="Number of caption tokens (real Sana uses 300).",
    )
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (required for the full 1.6B / 4.8B "
        "checkpoints; default ON for --mini fits comfortably).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Instantiate a tiny Sana transformer from config (random weights) "
        "instead of downloading the 1.6B checkpoint. Same architecture, a few "
        "M params. Useful for exploring export without the gated download.",
    )
    args = parser.parse_args()

    if args.mini:
        print("Instantiating mini SanaTransformer2DModel (random weights)")
        # Real Sana-1.6B 1024px: num_layers=20, num_attention_heads=70,
        # attention_head_dim=32, cross_attention_dim=2240,
        # caption_channels=2304, mlp_ratio=2.5, sample_size=32,
        # in/out_channels=32. Shrink every dim while keeping the
        # linear-attention + Mix-FFN architecture identical.
        transformer = (
            SanaTransformer2DModel(
                in_channels=4,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=8,
                num_layers=2,
                num_cross_attention_heads=2,
                cross_attention_head_dim=8,
                cross_attention_dim=16,
                caption_channels=24,
                mlp_ratio=2.0,
                sample_size=8,
                patch_size=1,
            )
            .to(torch.float32)
            .eval()
        )
    else:
        print(f"Loading SanaTransformer2DModel from {HF_REPO}")
        transformer = SanaTransformer2DModel.from_pretrained(
            HF_REPO, subfolder="transformer", torch_dtype=torch.float32
        )
    pipeline = SanaTransformerWrapper(transformer).eval()

    cfg = transformer.config
    in_channels = cfg.in_channels
    caption_channels = cfg.caption_channels
    print(
        f"in_channels={in_channels} caption_channels={caption_channels} "
        f"latent={args.latent_size}x{args.latent_size}"
    )

    hidden_states = torch.randn(
        1, in_channels, args.latent_size, args.latent_size, dtype=torch.float32
    )
    encoder_hidden_states = torch.randn(
        1, args.txt_tokens, caption_channels, dtype=torch.float32
    )
    timestep = torch.tensor([10.0], dtype=torch.float32)

    with torch.no_grad():
        out = pipeline(hidden_states, encoder_hidden_states, timestep)
    print(f"PyTorch output shape: {tuple(out.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=pipeline,
        args=(hidden_states, encoder_hidden_states, timestep),
        file_path_export=args.out,
        inference_target=TractNNEF(version=tract_version, check_io=check_io),
        input_names=["hidden_states", "encoder_hidden_states", "timestep"],
        output_names=["noise_pred"],
        debug_bundle_path=Path("./debug_sana_transformer.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
