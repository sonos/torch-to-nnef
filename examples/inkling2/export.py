"""Export a small random Inkling decoder and run streaming steps in tract."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.compress import DEFAULT_COMPRESSION
from torch_to_nnef.inference_target.tract import TractCli
from torch_to_nnef_llm.config import HFConfigHelper
from torch_to_nnef_llm.models.base import BaseCausal, update_forward_signature
from torch_to_nnef_llm.models.handlers.inkling2 import (
    Inkling2TextArchitectureHandler,
)


def small_model():
    """Keep attention, convolution, experts, and repeated layer visits."""
    config = Inkling2LoopTextConfig(
        vocab_size=32,
        unpadded_vocab_size=32,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=16,
        sliding_window_size=4,
        rel_extent=4,
        d_rel=2,
        n_layers_per_attn_res_block=1,
        loop_layer_index_start=1,
        loop_layer_index_end=3,
        loop_num_passes=3,
        loop_gate_rank=32,
        conv_kernel_size=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        eos_token_id=31,
        pad_token_id=0,
    )
    torch.manual_seed(7)
    model = Inkling2LoopForCausalLM(config).float().eval()
    # Exercise convolution history and loop messages, initialized to zero by HF.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(
                part in name for part in ("conv", "message_gate", "forget_gate")
            ):
                parameter.normal_(std=0.05)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tract", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("inkling2-export"))
    parser.add_argument(
        "--q4", action="store_true", help="quantize eligible weights to Q4_0"
    )
    args = parser.parse_args()
    cli = TractCli(args.tract.resolve())
    if cli.version < "0.23.10":
        parser.error("use tract 0.23.10 or newer")
    model = small_model()
    handler = Inkling2TextArchitectureHandler()
    handler.prepare_model_for_export(model)
    # Random weights need no tokenizer: use IDs in the small vocabulary.
    spec = handler.build_input_spec(
        tokenizer=lambda *a, **kw: SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]])
        ),
        config_helper=HFConfigHelper(model.config),
        inputs_dtype=torch.float32,
        sample_text="",
        n_input_tokens=3,
        n_past_input_tokens=0,
    )
    wrapper = BaseCausal(model, handler=handler, num_logits_to_keep=0)
    update_forward_signature(wrapper, spec)
    if args.q4:
        wrapper = DEFAULT_COMPRESSION[
            "min_max_q4_0_with_embeddings_and_experts"
        ](wrapper)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "model.nnef.tgz"
    target = TractNNEF(
        version=cli.version,
        specific_tract_binary_path=cli.tract_path,
        dynamic_axes=spec.dynamic_axes,
        check_io=not args.q4,
    )
    with torch.no_grad():
        export_model_to_nnef(
            model=wrapper,
            args=spec.inputs,
            file_path_export=path,
            inference_target=target,
            input_names=spec.input_names,
            output_names=spec.output_names,
        )
        # Keep independent PyTorch and tract state across prefill/decode calls.
        reference_state = spec.inputs[1:]
        runtime_state = [tensor.numpy() for tensor in reference_state]
        for step, count in enumerate((3, 1, 5, 1)):
            ids = torch.arange(1, count + 1).reshape(1, -1)
            expected = wrapper(ids, *reference_state)
            reference_state = expected[1:]
            inputs_path = args.out / f"step-{step}-inputs.npz"
            outputs_path = args.out / f"step-{step}-outputs.npz"
            np.savez(
                inputs_path,
                **dict(
                    zip(
                        spec.input_names,
                        [ids.numpy(), *runtime_state],
                        strict=True,
                    )
                ),
            )
            subprocess.run(
                cli.run_save_outputs_cmd_str(path, inputs_path, outputs_path),
                check=True,
                capture_output=True,
            )
            with np.load(outputs_path) as actual:
                for name, reference in zip(
                    spec.output_names, expected, strict=True
                ):
                    tolerance = 2e-3 if args.q4 else 1e-4
                    np.testing.assert_allclose(
                        actual[name],
                        reference.numpy(),
                        atol=tolerance,
                        rtol=tolerance,
                    )
                runtime_state = [actual[name] for name in spec.output_names[1:]]
            print(
                f"Step {step}: {count} tokens, "
                "logits and all state outputs match PyTorch"
            )
    states = [
        {
            "input": name,
            "output": output,
            "initial_shape": list(tensor.shape),
            "dtype": str(tensor.numpy().dtype),
        }
        for name, output, tensor in zip(
            spec.input_names[1:],
            spec.output_names[1:],
            spec.inputs[1:],
            strict=True,
        )
    ]
    (args.out / "state.json").write_text(json.dumps(states, indent=2) + "\n")
    print(
        f"Exported {path}. Random weights demonstrate execution, "
        "not useful text generation."
    )


if __name__ == "__main__":
    main()
