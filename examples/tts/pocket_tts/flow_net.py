"""Export Pocket-TTS' flow_net (``SimpleMLPAdaLN``) via torch-to-nnef.

The flow_net is the small AdaLN-modulated MLP that turns one Gaussian noise
sample into a denoised audio latent, conditioned on the FlowLM transformer's
output. It is called ``lsd_decode_steps`` times (default 4) per generated
audio frame -- the Rust runtime will invoke this graph once per inner step
of the LSD decode loop, so it ships as its own NNEF artifact.

Forward signature (matches ``SimpleMLPAdaLN.forward``):

    flow_net(c, s, t, x) -> flow_dir

- c : (B, cond_channels)  conditioning from FlowLM (transformer_out).
- s : (B, 1)              start time of this LSD step (scalar in [0, 1]).
- t : (B, 1)              target time of this LSD step.
- x : (B, ldim)           current latent (Gaussian noise on the first step,
                          partially denoised on subsequent steps).
- flow_dir : (B, ldim)    flow direction; the runtime adds ``flow_dir / K`` to
                          ``x`` between steps.

Run:
    # Mini, fast (random weights, ~13k params, no HF download):
    python flow_net.py --mini --skip-io-check
    # Mini with check_io against tract:
    python flow_net.py --mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pocket_tts import TTSModel as _TTSModel
from pocket_tts.modules.mlp import SimpleMLPAdaLN

from torch_to_nnef import TractNNEF, export_model_to_nnef


def build_mini_flow_net() -> SimpleMLPAdaLN:
    """Tiny SimpleMLPAdaLN mirroring Pocket-TTS' real flow config.

    Real Pocket-TTS: ``model_channels=512``, ``num_res_blocks=8``,
    ``cond_channels=512``, ``ldim=64``. Here we shrink every dim while keeping
    the AdaLN-modulated residual structure identical.
    """
    # ``cond_channels`` matches the FlowLM transformer's ``d_model`` (=16 in
    # the mini config in ``flow_lm.py``) so the same exported graph can be
    # plugged into the autoregressive loop without a separate adapter.
    return SimpleMLPAdaLN(
        in_channels=8,
        model_channels=16,
        out_channels=8,
        cond_channels=16,
        num_res_blocks=2,
        num_time_conds=2,
    ).eval()


def load_full_flow_net() -> SimpleMLPAdaLN:
    """Extract the trained ``flow_net`` from the real Pocket-TTS checkpoint.

    Triggers a HuggingFace download on first call (gated for the voice-cloning
    variant; ``kyutai/pocket-tts-without-voice-cloning`` is public).
    """
    return _TTSModel.load_model().flow_lm.flow_net.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("./pocket_tts_flow_net.nnef.tgz")
    )
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (only validates the graph emit).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Tiny random-weights config (~13k params) instead of the real "
        "Pocket-TTS checkpoint. Default if ``--full`` is not passed.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load the real Pocket-TTS checkpoint and export its trained "
        "``flow_net`` at production dims.",
    )
    args = parser.parse_args()
    if not args.full:
        args.mini = True

    torch.manual_seed(0)
    if args.full:
        model = load_full_flow_net()
        print("loaded real Pocket-TTS flow_net")
    else:
        model = build_mini_flow_net()
    print(f"flow_net params: {sum(p.numel() for p in model.parameters())}")

    batch = 1
    cond_dim = model.cond_embed.in_features
    ldim = model.in_channels
    cond = torch.randn(batch, cond_dim, dtype=torch.float32)
    t_start = torch.zeros(batch, 1, dtype=torch.float32)
    t_end = torch.full((batch, 1), 0.25, dtype=torch.float32)
    x = torch.randn(batch, ldim, dtype=torch.float32)

    with torch.no_grad():
        flow_dir = model(cond, t_start, t_end, x)
    print(f"PyTorch flow_dir shape: {tuple(flow_dir.shape)}")

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    print(f"Exporting to NNEF with tract {tract_version} (check_io={check_io})")
    export_model_to_nnef(
        model=model,
        args=(cond, t_start, t_end, x),
        file_path_export=args.out,
        inference_target=TractNNEF(version=tract_version, check_io=check_io),
        input_names=["cond", "t_start", "t_end", "x"],
        output_names=["flow_dir"],
        debug_bundle_path=Path("./debug_pocket_tts_flow_net.tgz"),
    )
    print(f"Exported to {args.out.absolute()}")


if __name__ == "__main__":
    main()
