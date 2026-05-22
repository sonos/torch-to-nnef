"""Export HF MambaForCausalLM as a per-token streaming NNEF artifact.

I/O signature:

    in : input_id[1]                int64
       + conv_states[L, 1, D, K]    float32
       + ssm_states[L, 1, D, N]     float32

    out: logits[1, vocab]           float32
       + conv_states_out[L, 1, D, K]
       + ssm_states_out[L, 1, D, N]

The shape constants (L, D, K, N, vocab_size) and the HF tokenizer hash
are written to a sidecar JSON manifest next to the artifact so the
Rust runtime doesn't need to hard-code them.

Usage:
    python export.py --repo state-spaces/mamba-130m-hf --out mamba130m.nnef.tgz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from streaming_wrapper import StreamingMamba
from transformers.models.mamba.modeling_mamba import MambaForCausalLM

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--repo",
        default="state-spaces/mamba-130m-hf",
        help="HF repo id of the Mamba model.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("mamba130m.nnef.tgz"),
        help="Path to write the NNEF artifact to.",
    )
    p.add_argument(
        "--no-check-io",
        action="store_true",
        help="Skip t2n's io check (faster export).",
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
    stream = StreamingMamba(model).eval()

    num_layers = stream.num_layers
    intermediate_size = stream.intermediate
    conv_kernel = stream.conv_kernel
    state_size = stream.state_size
    vocab_size = stream.vocab_size
    print(
        f"[export] L={num_layers} D={intermediate_size} K={conv_kernel} "
        f"N={state_size} vocab={vocab_size}"
    )

    input_id = torch.zeros(1, dtype=torch.long)
    conv = torch.zeros(
        num_layers, 1, intermediate_size, conv_kernel, dtype=torch.float32
    )
    ssm = torch.zeros(
        num_layers, 1, intermediate_size, state_size, dtype=torch.float32
    )

    target = TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=not args.no_check_io,
    )
    print(f"[export] exporting to {out_path}")
    export_model_to_nnef(
        model=stream,
        args=(input_id, conv, ssm),
        file_path_export=out_path,
        inference_target=target,
        input_names=["input_id", "conv_states", "ssm_states"],
        output_names=["logits", "conv_states_out", "ssm_states_out"],
    )

    manifest_path = out_dir / (out_path.name.split(".")[0] + ".json")
    manifest = {
        "repo": args.repo,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "conv_kernel": conv_kernel,
        "state_size": state_size,
        "vocab_size": vocab_size,
        "input_names": ["input_id", "conv_states", "ssm_states"],
        "output_names": ["logits", "conv_states_out", "ssm_states_out"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[export] wrote manifest {manifest_path}")

    tok_src = hf_hub_download(args.repo, "tokenizer.json")
    tok_dst = out_dir / "tokenizer.json"
    shutil.copyfile(tok_src, tok_dst)
    print(f"[export] copied tokenizer {tok_dst}")


if __name__ == "__main__":
    main()
