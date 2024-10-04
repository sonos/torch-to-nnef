#
import typing as T
from enum import Enum
from functools import partial

import torch

from torch_to_nnef.llm_tract.models.base import (
    BaseCausal,
    BaseCausalWithDynCacheAndTriu,
)
from torch_to_nnef.log import log

LOGGER = log.getLogger(__name__)


# collection of tested examples for cli {
class PHISlugs(str, Enum):
    DEBUG = "phi_debug"
    ONE_FIVE = "microsoft/phi-1_5"
    MINI = "microsoft/Phi-3-mini-4k-instruct"
    SMALL = "microsoft/Phi-3-small-8k-instruct"


class LlamaSLugs(str, Enum):
    DUMMY = "yujiepan/llama-2-tiny-random"
    TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LLAMA3_8B = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
    LLAMA2_7B_BASE = "meta-llama/Llama-2-7b-hf"


class OpenELMSlugs(str, Enum):
    MICRO = "apple/OpenELM-270M-Instruct"
    MINI = "apple/OpenELM-450M-Instruct"
    MEDIUM = "apple/OpenELM-1_1B-Instruct"
    BIG = "apple/OpenELM-3B-Instruct"


# }
CUSTOM_CONFIGS: T.Dict[str, T.Any] = {}

try:
    from transformers.models.phi3.configuration_phi3 import Phi3Config

    CUSTOM_CONFIGS[PHISlugs.DEBUG] = Phi3Config(
        model_type="phi3debug",
        vocab_size=32064,
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_size=256,
        intermediate_size=512,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
    )
except (ModuleNotFoundError, ImportError) as exp:
    LOGGER.debug(
        f"Phi3 not available since too old version of transformers: {exp}"
    )

REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG: T.Dict[str, str] = {
    "openelm": LlamaSLugs.LLAMA2_7B_BASE.value,
    "phi3debug": PHISlugs.MINI.value,
}


class HFConfigHelper:
    """HuggingFace config helper

    Allow to extract usefull informations
    from config to set export parameters

    """

    def __init__(self, model_slug, conf):
        self.conf = conf
        self.model_slug = model_slug
        if conf.model_type == "openelm":
            self.max_position_embeddings = conf.max_context_length
        else:
            self.max_position_embeddings = conf.max_position_embeddings

        if conf.model_type in ["phi"]:
            self.wrapper_class = BaseCausalWithDynCacheAndTriu
        elif conf.model_type in ["openelm"]:
            self.wrapper_class = partial(BaseCausal, with_dyn_cache=False)
        else:
            self.wrapper_class = BaseCausal
        LOGGER.info(
            f"detected arch:'{conf.model_type}' using wrapper '{self.wrapper_class}'"
        )

    def get_past_value_cache_conf(self, n_past_input_tokens: int):
        if self.conf.model_type == "openelm":
            num_hidden_layers = self.conf.num_transformer_layers
            past_values_cache_conf = {
                "n_kv": num_hidden_layers,
                "kv_shape": [
                    (
                        1,
                        self.conf.num_kv_heads[layer_idx],
                        n_past_input_tokens,
                        self.conf.head_dim,
                    )
                    for layer_idx in range(num_hidden_layers)
                    for _ in range(2)  # k and v
                ],
            }
        else:
            if hasattr(self.conf, "head_dim"):
                shape_last_dim = int(self.conf.head_dim)
            else:
                shape_last_dim = int(
                    self.conf.hidden_size / self.conf.num_attention_heads
                )
            past_values_cache_conf = {
                "n_kv": self.conf.num_hidden_layers,
                "kv_shape": [
                    (
                        1,
                        self.conf.num_key_value_heads,
                        n_past_input_tokens,
                        shape_last_dim,
                    )
                ]
                * self.conf.num_hidden_layers
                * 2,
            }
        return past_values_cache_conf

    def build_kv_cache_infos(
        self, n_past_input_tokens: int, as_float16: bool = False
    ):
        past_values_cache_conf = self.get_past_value_cache_conf(
            n_past_input_tokens
        )
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

            k_or_v = torch.rand(past_values_cache_conf["kv_shape"][idx]).float()
            if as_float16:
                k_or_v = k_or_v.to(torch.float16)
            past_key_values.append(k_or_v)
            in_cache_name = f"in_{node_name}"
            in_cache_names.append(in_cache_name)
            out_cache_names.append(f"out_{node_name}")
            # past s   dynamic_axes[in_cache_name] = {2: "PAST_S"}
            dynamic_axes[in_cache_name] = {2: "P"}
        return in_cache_names, out_cache_names, past_key_values, dynamic_axes
