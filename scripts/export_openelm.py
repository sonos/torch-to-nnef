"""

https://huggingface.co/apple/OpenELM

code modeling is from: https://huggingface.co/apple/OpenELM-270M/blob/main/modeling_openelm.py
"""

import argparse
import logging as log
import os
import typing as T
from enum import Enum
from pathlib import Path

import torch
from transformers import (  # LlamaConfig,; LlamaModel,; LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.torch_graph.ir_graph import VariableNamingScheme

LLAMA2_TOK_SLUG = "meta-llama/Llama-2-7b-hf"


class OpenELMSlugs(str, Enum):
    MICRO = "apple/OpenELM-270M-Instruct"
    MINI = "apple/OpenELM-450M-Instruct"
    MEDIUM = "apple/OpenELM-1_1B-Instruct"
    BIG = "apple/OpenELM-3B-Instruct"


class SuperBasicCausal(torch.nn.Module):
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
            input_ids, past_key_values=past_key_values, use_cache=True
        )

        kvs = [k_or_v for kv in out_dic["past_key_values"] for k_or_v in kv]

        assert len(past_key_values) * 2 == len(
            kvs
        ), f"{len(past_key_values) * 2} == {len(kvs)}"
        # key values, (32 tensors) of shape (1, 3, S, 64)
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
        default=OpenELMSlugs.MICRO.value,
        choices=[_.value for _ in OpenELMSlugs],
        help="Default OpenELM huggingface slug to export",
    )
    parser.add_argument(
        "-d",
        "--local-dir",
        help="local dir containing .safetensors compatible with openELM"
        " model size specified in slug",
    )
    return parser.parse_args()


def main():
    args = parser_cli()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    default_model_slug = OpenELMSlugs(args.model_slug).value  # check enum type
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA2_TOK_SLUG, trust_remote_code=True
    )

    test_input = tokenizer("Hello, I am happy", return_tensors="pt")

    S = 10
    # cache size
    past_values_cache_conf = {
        OpenELMSlugs.MICRO: {
            "n_kv": 16,
            "kv_shape": [
                [1, 3, S, 64],  # 0
                [1, 3, S, 64],  # 1
                [1, 3, S, 64],  # 2
                [1, 3, S, 64],  # 3
                [1, 3, S, 64],  # 4
                [1, 3, S, 64],  # 5
                [1, 3, S, 64],  # 6
                [1, 3, S, 64],  # 7
                [1, 3, S, 64],  # 8
                [1, 3, S, 64],  # 9
                [1, 4, S, 64],  # 10
                [1, 4, S, 64],  # 11
                [1, 4, S, 64],  # 12
                [1, 4, S, 64],  # 13
                [1, 4, S, 64],  # 14
                [1, 4, S, 64],  # 15
                [1, 4, S, 64],  # 16
                [1, 4, S, 64],  # 17
                [1, 4, S, 64],  # 18
                [1, 4, S, 64],  # 19
                [1, 4, S, 64],  # 20
                [1, 4, S, 64],  # 21
                [1, 4, S, 64],  # 22
                [1, 4, S, 64],  # 23
                [1, 5, S, 64],  # 24
                [1, 5, S, 64],  # 25
                [1, 5, S, 64],  # 26
                [1, 5, S, 64],  # 27
                [1, 5, S, 64],  # 28
                [1, 5, S, 64],  # 29
                [1, 5, S, 64],  # 30
                [1, 5, S, 64],  # 31
            ],
        },
        OpenELMSlugs.MEDIUM: {
            "n_kv": 28,
            "kv_shape": [
                [1, 4, S, 64],
                [1, 4, S, 64],
                [1, 4, S, 64],
                [1, 4, S, 64],
                [1, 4, S, 64],
                [1, 4, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 5, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 6, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 7, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
                [1, 8, S, 64],
            ],
        },
        # TODO: other sizes
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
            torch.rand(past_values_cache_conf["kv_shape"][idx]).float()
        )
        in_cache_name = f"in_{node_name}"
        in_cache_names.append(in_cache_name)
        out_cache_names.append(f"out_{node_name}")
        # past s   dynamic_axes[in_cache_name] = {2: "PAST_S"}
        dynamic_axes[in_cache_name] = {2: "P"}

    if args.local_dir:
        dir_path = Path(args.local_dir)
        assert dir_path.is_dir(), dir_path
        assert (dir_path / "model.safetensors").is_file(), dir_path
        causal_llm = AutoModelForCausalLM.from_pretrained(
            dir_path, trust_remote_code=True
        )
        print("load from local directory:", dir_path)
    else:
        causal_llm = AutoModelForCausalLM.from_pretrained(
            default_model_slug, trust_remote_code=True
        )
        print(f"default Apple trained model: {default_model_slug}")

    remodeled_llm = SuperBasicCausal(causal_llm)

    # generated_ids = causal_llm.generate(**test_input)
    # print(generated_ids)
    # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    # caus_res = remodeled_llm(test_input.input_ids)
    # print("caus_res.shape:", caus_res.shape)
    inputs = tuple([test_input.input_ids] + past_key_values)
    _ = remodeled_llm(*inputs)

    export_model_to_nnef(
        model=remodeled_llm,
        args=inputs,
        file_path_export=Path(args.export_filepath),
        input_names=["input_ids"] + in_cache_names,
        output_names=["outputs"] + out_cache_names,
        log_level=log.INFO,
        check_same_io_as_tract=True,
        dynamic_axes=dynamic_axes,
        renaming_scheme=VariableNamingScheme.NATURAL_VERBOSE,
        custom_extensions={
            "tract_assert P >= 0",
            "tract_assert S >= 0",
            "tract_assert S+P < 2048",
        },
    )


if __name__ == "__main__":
    main()
