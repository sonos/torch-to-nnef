"""
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

code modeling is from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
"""

import argparse
import logging as log
import os
import typing as T
from enum import Enum
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.torch_graph.ir_graph import VariableNamingScheme


class PHISlugs(str, Enum):
    MINI = "microsoft/Phi-3-mini-4k-instruct"
    SMALL = "microsoft/Phi-3-small-8k-instruct"


class BasicCausal(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, *args):
        # input_ids: [1, S] with torch.int64
        # past_key_values
        # past_key_values: Optional[List[torch.FloatTensor]] = None # type annotation in code WRONG

        past_key_values = []
        tup: T.List[torch.Tensor] = []
        for idx, k_or_v in enumerate(args):
            if idx % 2 == 0 and len(tup):
                assert len(tup) == 2
                past_key_values.append(tup)
                tup = []
            tup.append(k_or_v)
        assert len(tup) == 2
        past_key_values.append(tup)

        out_dic = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        kvs = [k_or_v for kv in out_dic["past_key_values"] for k_or_v in kv]

        assert len(past_key_values) * 2 == len(
            kvs
        ), f"{len(past_key_values) * 2} == {len(kvs)}"
        return [out_dic["logits"]] + kvs


def parser_cli():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-e",
        "--export-filepath",
        required=True,
        help="export file path to dump .nnef.tgz",
    )
    parser.add_argument(
        "-s",
        "--model-slug",
        default=PHISlugs.MINI.value,
        choices=[_.value for _ in PHISlugs],
        help="Default OpenELM huggingface slug to export",
    )
    return parser.parse_args()


def main():
    args = parser_cli()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    default_model_slug = PHISlugs(args.model_slug).value  # check enum type
    tokenizer = AutoTokenizer.from_pretrained(
        default_model_slug, trust_remote_code=True
    )

    test_input = tokenizer("Hello, I am happy" * 50, return_tensors="pt")

    S = 10
    # cache size
    past_values_cache_conf = {
        PHISlugs.MINI: {
            "n_kv": 32,
            "kv_shape": (1, 32, S, 96),
        },
        # TODO: set right shape
        PHISlugs.SMALL: {"n_kv": 28, "kv_shape": (1, 4, S, 64)},
    }[default_model_slug]
    past_key_values = []
    in_cache_names = []
    out_cache_names = []

    dynamic_axes = {
        "input_ids": {1: "S"},
    }
    for idx in range(past_values_cache_conf["n_kv"] * 2):
        if idx % 2 == 0:
            node_name = f"cache_key_{int(idx / 2)}"
        else:
            node_name = f"cache_value_{int((idx -1) / 2)}"
        past_key_values.append(
            torch.rand(past_values_cache_conf["kv_shape"]).float()
        )
        in_cache_name = f"in_{node_name}"
        in_cache_names.append(in_cache_name)
        out_cache_names.append(f"out_{node_name}")
        # past s   dynamic_axes[in_cache_name] = {2: "PAST_S"}
        dynamic_axes[in_cache_name] = {2: "P"}

    causal_llama = AutoModelForCausalLM.from_pretrained(
        default_model_slug, trust_remote_code=True
    )
    striped_model = BasicCausal(causal_llama)

    # generated_ids = causal_llama.generate(**test_input)
    # print(generated_ids)
    # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    # caus_res = striped_model(test_input.input_ids)
    # print("caus_res.shape:", caus_res.shape)
    inputs = tuple([test_input.input_ids] + past_key_values)
    _ = striped_model(*inputs)

    export_model_to_nnef(
        model=striped_model,
        args=inputs,
        file_path_export=Path(args.export_filepath),
        input_names=["input_ids"] + in_cache_names,
        output_names=["outputs"] + out_cache_names,
        log_level=log.INFO,
        check_same_io_as_tract=True,
        dynamic_axes=dynamic_axes,
        renaming_scheme=VariableNamingScheme.NATURAL_VERBOSE,
    )


if __name__ == "__main__":
    main()
