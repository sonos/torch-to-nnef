"""Export the pinned LFM3 checkpoint, or a tiny random architecture probe."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli
from torch_to_nnef_llm.config import HFConfigHelper
from torch_to_nnef_llm.models.base import BaseCausal
from torch_to_nnef_llm.models.handlers import Lfm3ArchitectureHandler

REPO = "LiquidAI/LFM3-1.7B-Base"
REVISION = "7bca5038823c0d0f14efe06439109354a10e014c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("lfm3-export"))
    parser.add_argument("--tract", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    source = args.model_dir or snapshot_download(
        REPO,
        revision=REVISION,
        allow_patterns=["*.py", "*.json", "*.jinja"] if args.tiny else None,
    )
    dtype = {"f32": torch.float32, "f16": torch.float16}[args.dtype]
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    if args.tiny:
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 8
        config.num_hidden_layers = 3
        config.layer_types = ["conv", "full_attention", "sliding_attention"]
        config.sliding_window = 4
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="eager",
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    handler = Lfm3ArchitectureHandler()
    handler.prepare_model_for_export(model)
    spec = handler.build_input_spec(
        tokenizer=tokenizer,
        config_helper=HFConfigHelper(config),
        inputs_dtype=dtype,
        sample_text="The nematode C. elegans is",
        n_input_tokens=3,
        n_past_input_tokens=2,
    )
    wrapper = BaseCausal(model, handler=handler, num_logits_to_keep=1)
    cli = TractCli(args.tract.resolve())
    target = TractNNEF(
        version=cli.version,
        specific_tract_binary_path=cli.tract_path,
        dynamic_axes=spec.dynamic_axes,
        check_io=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export_model_to_nnef(
            model=wrapper,
            args=spec.inputs,
            file_path_export=args.out / "model.nnef.tgz",
            inference_target=target,
            input_names=spec.input_names,
            output_names=spec.output_names,
        )
    tokenizer.save_pretrained(args.out)
    model.config.save_pretrained(args.out)
    states = []
    for name, out_name, tensor in zip(
        spec.input_names[1:],
        spec.output_names[1:],
        spec.inputs[1:],
        strict=True,
    ):
        shape = list(tensor.shape)
        if tensor.ndim == 4:
            shape[2] = 0
        states.append(
            {"input": name, "output": out_name, "initial_shape": shape}
        )
    (args.out / "state.json").write_text(
        json.dumps(
            {
                "model": REPO,
                "revision": REVISION,
                "tiny_random": args.tiny,
                "dtype": args.dtype,
                "states": states,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported {args.out / 'model.nnef.tgz'}")


if __name__ == "__main__":
    main()
