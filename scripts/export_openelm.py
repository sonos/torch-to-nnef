"""

https://huggingface.co/apple/OpenELM

code modeling is from: https://huggingface.co/apple/OpenELM-270M/blob/main/modeling_openelm.py
"""

import argparse
import logging as log
import os
from enum import Enum
from pathlib import Path

import torch
from transformers import (  # LlamaConfig,; LlamaModel,; LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from torch_to_nnef.export import export_model_to_nnef

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

    def forward(self, input_ids: torch.Tensor):
        out_dic = self.model(input_ids)
        return out_dic["logits"]


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
    return parser.parse_args()


def main():
    args = parser_cli()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    default_model_slug = OpenELMSlugs(args.model_slug).value  # check enum type
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA2_TOK_SLUG, trust_remote_code=True
    )

    # NOTE: size of tokenized text need to be very large because of logic inside
    # modeling_llama2 rotary logic that use cache system not JITABLE based on seq len ...
    test_input = tokenizer("Hello, I am happy" * 50, return_tensors="pt")
    causal_llama = AutoModelForCausalLM.from_pretrained(
        default_model_slug, trust_remote_code=True
    )
    striped_model = SuperBasicCausal(causal_llama)

    # generated_ids = causal_llama.generate(**test_input)
    # print(generated_ids)
    # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    # caus_res = striped_model(test_input.input_ids)
    # print("caus_res.shape:", caus_res.shape)

    export_model_to_nnef(
        model=striped_model,
        args=(test_input.input_ids,),
        file_path_export=Path(args.export_filepath),
        input_names=["input_ids"],
        output_names=["outputs"],
        log_level=log.INFO,
        check_same_io_as_tract=True,
        dynamic_axes={"input_ids": {1: "S"}},
        renaming_scheme="natural_verbose",
    )


if __name__ == "__main__":
    main()
