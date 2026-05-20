r"""Export HF MambaForCausalLM as a pulse-mode streaming NNEF artifact.

The graph is a prefill graph with the sequence axis declared as the
symbolic streaming dimension `S`. Downstream tract:

    tract --nnef-tract-core mamba_pulse.nnef.tgz --pulse S=1 ...

lowers it to a per-pulse runner: one timestep in, one logits row out,
internal conv + ssm state carried implicitly across pulses.

I/O:
    input_ids[1, S]   int64
        -> logits[1, S, vocab]   float32

Usage:
    python export.py \\
        --repo state-spaces/mamba-130m-hf \\
        --out mamba_pulse.nnef.tgz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import patch_pulse
import torch
from huggingface_hub import hf_hub_download
from transformers.models.mamba.modeling_mamba import MambaForCausalLM

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF

patch_pulse.install()


class _MambaForward(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        return self.inner(input_ids).logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--repo", default="state-spaces/mamba-130m-hf")
    p.add_argument("--out", type=Path, default=Path("mamba_pulse.nnef.tgz"))
    p.add_argument(
        "--trace-len",
        type=int,
        default=8,
        help=(
            "Sequence length used at trace time (any S >= 1 works at runtime)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out.resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading {args.repo}")
    model = MambaForCausalLM.from_pretrained(
        args.repo, torch_dtype=torch.float32
    )
    model.eval()
    # MambaModel.forward auto-creates a DynamicCache when use_cache is
    # True (HF default); under tracing that cache_params would route
    # us back to the unrolled per-token slow_forward and erase the
    # ssm_scan_y custom op. Disable it for the prefill trace.
    model.config.use_cache = False
    wrap = _MambaForward(model).eval()
    L = model.config.num_hidden_layers
    V = model.config.vocab_size
    print(f"[export] L={L} vocab={V} trace_len={args.trace_len}")

    ids = torch.arange(args.trace_len, dtype=torch.long).unsqueeze(0)

    target = TractNNEF(
        dynamic_axes={"input_ids": {1: "S"}},
        version=TractNNEF.latest_version(),
        check_io=False,
    )
    print(f"[export] exporting to {out_path}")
    export_model_to_nnef(
        model=wrap,
        args=(ids,),
        file_path_export=out_path,
        inference_target=target,
        input_names=["input_ids"],
        output_names=["logits"],
    )

    manifest_path = out_dir / (out_path.name.split(".")[0] + ".json")
    manifest = {
        "repo": args.repo,
        "num_layers": L,
        "vocab_size": V,
        "streaming_symbol": "S",
        "input_names": ["input_ids"],
        "output_names": ["logits"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[export] wrote manifest {manifest_path}")

    tok_src = hf_hub_download(args.repo, "tokenizer.json")
    tok_dst = out_dir / "tokenizer.json"
    shutil.copyfile(tok_src, tok_dst)
    print(f"[export] copied tokenizer {tok_dst}")


if __name__ == "__main__":
    main()
