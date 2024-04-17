import argparse
import logging as log
from enum import Enum
from pathlib import Path

import torch
from transformers import (  # LlamaConfig,; LlamaModel,; LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from torch_to_nnef.export import export_model_to_nnef


class SuperBasicCausal(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax()

    def forward(self, input_ids: torch.Tensor):
        """same as calling without any smart caching mechanism self.model.model+lm_head and softmax.

        This export module is extremly ineficient because no caching can be provided ...

        """
        _, seq_length = input_ids.shape[:2]
        past_key_values_length = 0
        # get pos ids {
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        attention_mask = (
            torch.triu(
                torch.full(
                    [seq_length, seq_length], torch.finfo(torch.float32).min
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # }

        hidden_states = inputs_embeds
        for _, decoder_layer in enumerate(self.model.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        logits = self.model.lm_head(hidden_states)
        return self.softmax(logits)


class Llama2SLugs(str, Enum):
    DUMMY = "yujiepan/llama-2-tiny-random"
    TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


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
        default=Llama2SLugs.TINY.value,
        choices=[_.value for _ in Llama2SLugs],
        help="Default llama2 huggingface slug to export",
    )
    return parser.parse_args()


def main():
    args = parser_cli()
    default_model_slug = Llama2SLugs(args.model_slug).value  # check enum type
    tokenizer = AutoTokenizer.from_pretrained(default_model_slug)
    test_input = tokenizer("Hello, I am happy", return_tensors="pt")
    causal_llama = AutoModelForCausalLM.from_pretrained(default_model_slug)
    striped_model = SuperBasicCausal(causal_llama)

    export_model_to_nnef(
        model=striped_model,
        args=(test_input.input_ids,),
        file_path_export=Path(args.export_filepath),
        input_names=["input_ids"],
        output_names=["outputs"],
        log_level=log.INFO,
        check_same_io_as_tract=True,
        dynamic_axes={"input_ids": {1: "S"}},
    )


if __name__ == "__main__":
    main()
