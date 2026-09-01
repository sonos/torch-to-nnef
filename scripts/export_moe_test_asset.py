#!/usr/bin/env python3
"""Export a tiny Qwen3-MoE NNEF model + reference I/O for tract tests.

Produces a directory with graph.nnef + .dat weight files + io.npz.

Usage:
    cd /path/to/t2n_main
    .venv/bin/python scripts/export_moe_test_asset.py /path/to/output_dir
"""
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeSparseMoeBlock,
)

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.exceptions import T2NError
from torch_to_nnef.inference_target.tract import TractCli


class Qwen3TinyMoE(nn.Module):
    """Minimal Qwen3 MoE block: 4 experts, top-2, hidden=16, intermediate=32."""

    def __init__(self):
        super().__init__()
        cfg = Qwen3MoeConfig(
            hidden_size=16,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            hidden_act="silu",
            # real Qwen3-MoE checkpoints renormalize the top-k gates; the
            # adapter rejects norm_topk_prob=False (the config default).
            norm_topk_prob=True,
        )
        self.moe = Qwen3MoeSparseMoeBlock(cfg)
        # transformers MoE experts allocate weights with torch.empty
        # (uninitialized); a standalone block must be seeded or the reference
        # output is NaN.
        torch.manual_seed(0)
        with torch.no_grad():
            for p in self.moe.parameters():
                nn.init.normal_(p, std=0.02)

    def forward(self, x):
        # Qwen3MoeSparseMoeBlock expects [batch, seq, hidden]
        return self.moe(x)


def _inference_target() -> TractNNEF:
    # Prefer a tract build that has tract_moe_ffn (the op is unreleased); fall
    # back to the latest official version (export still writes the asset even
    # if the post-export IO check then fails).
    tract_path = os.environ.get("T2N_TEST_TRACT_PATH")
    if tract_path:
        cli = TractCli(Path(tract_path))
        return TractNNEF(
            cli.version, specific_tract_binary_path=Path(tract_path)
        )
    return TractNNEF.latest()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = Qwen3TinyMoE()
    model.eval()

    # Test input: [batch=1, seq=3, hidden=16]
    torch.manual_seed(123)
    test_input = torch.randn(1, 3, 16)

    with torch.no_grad():
        ref_output = model(test_input)
    assert torch.isfinite(ref_output).all(), "reference output is not finite"

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "qwen3_moe_tiny.nnef"
        try:
            exported = export_model_to_nnef(
                model=model,
                args=(test_input,),
                file_path_export=export_path,
                inference_target=_inference_target(),
                input_names=["input_0"],
                output_names=["output_0"],
                compression_level=None,
                log_level=logging.INFO,
            )
        except T2NError as e:
            exported = export_path
            if not exported.exists():
                raise RuntimeError(f"Export failed: {e}") from e
            print(f"WARNING: post-export validation failed (expected): {e}")

        nnef_dir = Path(exported)
        for item in nnef_dir.iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    np.savez(
        output_dir / "io.npz",
        input_0=test_input.numpy(),
        output_0=ref_output.numpy(),
    )

    print(f"Exported to: {output_dir}")
    total = 0
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total += size
            print(f"  {f.relative_to(output_dir)}: {size} bytes")
    print(f"  TOTAL: {total} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
