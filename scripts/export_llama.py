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
    LlamaForCausalLM,
)
from transformers.cache_utils import DynamicCache

from torch_to_nnef.export import export_model_to_nnef

# from transformers.models.llama import modeling_llama

# class DynLlamaRotaryEmbedding(torch.nn.Module):
#     """LlamaRotaryEmbedding without 'orignal caching'.
#
#     Original caching is hurtfull in this case because
#     when torch jit trace is applied static max sequence seq_length
#     is done but we do not provide as sample with max size
#     so we add a static arange of very big size to be cached
#     """
#
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (
#             self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
#         )
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#
#         t = torch.arange(10_000).to(self.inv_freq.dtype)
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # freqs = t.unsqueeze(0).T @ self.inv_freq.unsqueeze(0)
#
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("emb_cos", emb.cos(), persistent=False)
#         self.register_buffer("emb_sin", emb.sin(), persistent=False)
#
#     def forward(self, x, seq_len=None):
#         # if isinstance(seq_len, torch.Tensor):
#         #     seq_len = seq_len.tolist()  # fix issue with jit trace and tensor seq_len
#         return (self.emb_cos[:seq_len], self.emb_sin[:seq_len])
#
#
# ISSUE transcript
#
# modeling_llama.LlamaRotaryEmbedding = DynLlamaRotaryEmbedding
#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]
#     # here kv_seq_len is constantized .....
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)


class SuperBasicCausal(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, *args):
        """same as calling without any smart caching mechanism self.model.model+lm_head and softmax.

        This export module is extremly ineficient because no caching can be provided ...

        """
        _, seq_length = input_ids.shape[:2]

        # BUILD cache {
        past_key_values = []
        tup: T.List[torch.Tensor] = []
        for idx, k_or_v in enumerate(args):
            if idx % 2 == 0 and len(tup):
                assert len(tup) == 2
                past_key_values.append(tuple(tup))
                tup = []
            tup.append(k_or_v)
        assert len(tup) == 2
        past_key_values.append(tuple(tup))
        # cache = DynamicCache.from_legacy_cache(tuple(past_key_values))
        cache = DynamicCache()
        # }
        past_key_values_length = cache.get_seq_length()

        # get pos ids {
        cache_position = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = cache_position.unsqueeze(0)
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
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        logits = self.model.lm_head(hidden_states)

        # Extract cache {
        kv_cache_flat_list = [t for kv in cache.to_legacy_cache() for t in kv]
        __import__("ipdb").set_trace()
        # }
        return [logits] + kv_cache_flat_list


class LlamaSLugs(str, Enum):
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
        default=LlamaSLugs.TINY.value,
        choices=[_.value for _ in LlamaSLugs],
        help="Default llama2 huggingface slug to export",
    )
    return parser.parse_args()


def main():
    args = parser_cli()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    default_model_slug = LlamaSLugs(args.model_slug).value  # check enum type
    tokenizer = AutoTokenizer.from_pretrained(default_model_slug)

    S = 10
    past_values_cache_conf = {
        LlamaSLugs.TINY: {
            "n_kv": 22,
            "kv_shape": (1, 4, S, 64),
        },
        LlamaSLugs.DUMMY: {
            "n_kv": 1,
            "kv_shape": (1, 2, S, 4),
        },
    }[default_model_slug]

    dynamic_axes = {
        "input_ids": {1: "S"},
    }
    past_key_values = []
    in_cache_names = []
    out_cache_names = []
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

    # NOTE: size of tokenized text need to be very large because of logic inside
    # modeling_llama2 rotary logic that use cache system not JITABLE based on seq len ...
    test_input = tokenizer("Hello, I am happy", return_tensors="pt")
    causal_llama = AutoModelForCausalLM.from_pretrained(default_model_slug)
    striped_model = SuperBasicCausal(causal_llama)

    # generated_ids = causal_llama.generate(**test_input)
    # print(generated_ids)
    # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    # caus_res = striped_model(test_input.input_ids)
    # print("caus_res.shape:", caus_res.shape)
    inputs = tuple([test_input.input_ids[:, :1]] + past_key_values)
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
        renaming_scheme="natural_verbose",
    )


if __name__ == "__main__":
    main()
