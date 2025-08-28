#
import logging
import typing as T
from enum import Enum
from functools import partial

import torch
from transformers import AutoConfig, AutoTokenizer

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.llm_tract.models.base import (
    BaseCausal,
    BaseCausalWithDynCacheAndTriu,
)

LOGGER = logging.getLogger(__name__)


class ExportDirStruct(str, Enum):
    DEEP = "deep"
    # it will dump tokenizer files, config.json and model.nnef.tgz
    # with no sub-directories in requested export dir
    # as most huggingface repositories
    FLAT = "flat"


class DtypeStr(str, Enum):
    FLOAT32 = "f32"
    FLOAT16 = "f16"
    BFLOAT16 = "bf16"

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            self.FLOAT32: torch.float32,
            self.FLOAT16: torch.float16,
            self.BFLOAT16: torch.bfloat16,
        }[self.value]

    @classmethod
    def from_torch_dtype(cls, dtype: torch.dtype):
        for ds in DtypeStr:
            if ds.torch_dtype == dtype:
                return ds
        raise T2NErrorNotImplemented(dtype)


# collection of tested examples for cli {
class PHISlugs(str, Enum):
    DEBUG = "phi_debug"
    ONE_FIVE = "microsoft/phi-1_5"
    MINI = "microsoft/Phi-3-mini-4k-instruct"
    SMALL = "microsoft/Phi-3-small-8k-instruct"


class LlamaSlugs(str, Enum):
    DUMMY = "yujiepan/llama-2-tiny-random"
    TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LLAMA3_8B = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
    LLAMA2_7B_BASE = "meta-llama/Llama-2-7b-hf"


class SmolSlugs(str, Enum):
    TINY = "HuggingFaceTB/SmolLM-135M"


class OpenELMSlugs(str, Enum):
    MICRO = "apple/OpenELM-270M-Instruct"
    MINI = "apple/OpenELM-450M-Instruct"
    MEDIUM = "apple/OpenELM-1_1B-Instruct"
    BIG = "apple/OpenELM-3B-Instruct"


class MistralSlugs(str, Enum):
    DEBUG = "mistral_debug"
    MISTRAL_7B_V03 = "mistralai/Mistral-7B-Instruct-v0.3"


class Gemma3Slugs(str, Enum):
    TINY = "google/gemma-3-270m"


class Qwen3Slugs(str, Enum):
    TINY = "Qwen/Qwen3-0.6B"


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
        "Phi3 not available since too old version of transformers: %s", exp
    )
try:
    from transformers.models.mistral.configuration_mistral import MistralConfig

    CUSTOM_CONFIGS[MistralSlugs.DEBUG] = MistralConfig(
        vocab_size=32768,
        hidden_size=256,
        intermediate_size=512,
        head_dim=128,
        sliding_window=None,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        model_type="mistraldebug",
    )
except (ModuleNotFoundError, ImportError) as exp:
    LOGGER.debug(
        "Mistral not available since too old version of transformers: %s", exp
    )


REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG: T.Dict[str, str] = {
    "openelm": LlamaSlugs.LLAMA2_7B_BASE.value,
    "phi3debug": PHISlugs.MINI.value,
    "mistraldebug": MistralSlugs.MISTRAL_7B_V03.value,
}


def register_raw_model_from_slug(model_id, trust_remote_code: bool = True):
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    suffix = "__t2n_debug"
    config.model_type += suffix
    new_model_id = model_id + suffix
    CUSTOM_CONFIGS[new_model_id] = config
    REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG[config.model_type] = model_id
    return new_model_id


def get_tokenizer_from_slug(mdl_slug):
    tok_slug = mdl_slug
    if mdl_slug in CUSTOM_CONFIGS:
        tok_slug = REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG.get(
            CUSTOM_CONFIGS[mdl_slug].model_type, mdl_slug
        )
    return AutoTokenizer.from_pretrained(tok_slug, trust_remote_code=True)


class HFConfigHelper:
    """HuggingFace config helper.

    Allow to extract usefull informations
    from config to set export parameters

    """

    def __init__(self, conf):
        self.conf = conf
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
            "detected arch:'%s' using wrapper '%s'",
            conf.model_type,
            self.wrapper_class,
        )

    def get_head_dim(self):
        if hasattr(self.conf, "head_dim"):
            return int(self.conf.head_dim)
        return int(self.conf.hidden_size / self.conf.num_attention_heads)

    def get_num_kv_heads(self, layer_idx: int):
        if hasattr(self.conf, "num_kv_heads"):
            return self.conf.num_kv_heads[layer_idx]
        return self.conf.num_key_value_heads

    def get_num_transformer_layers(self):
        if self.conf.model_type == "openelm":
            return self.conf.num_transformer_layers
        return self.conf.num_hidden_layers

    def get_past_value_cache_conf(self, n_past_input_tokens: int):
        past_values_cache_conf = {
            "n_kv": self.get_num_transformer_layers(),
            "kv_shape": [
                (
                    1,
                    self.get_num_kv_heads(layer_idx),
                    n_past_input_tokens,
                    self.get_head_dim(),
                )
                for layer_idx in range(self.get_num_transformer_layers())
                for _ in range(2)  # k and v
            ],
        }
        return past_values_cache_conf

    def build_kv_cache_infos(
        self,
        n_past_input_tokens: int,
        force_inputs_dtype: T.Optional[torch.dtype] = None,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
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
            layer_idx = idx % 2
            if layer_idx == 0:
                node_name = f"cache_key_{int(idx / 2)}"
            else:
                node_name = f"cache_value_{int((idx - 1) / 2)}"

            if real_kv_cache:
                kv_dims = real_kv_cache[0].shape
                assert kv_dims[0] == 1
                assert kv_dims[1] == self.get_num_kv_heads(layer_idx)
                assert kv_dims[2] >= n_past_input_tokens
                assert kv_dims[3] == self.get_head_dim()
                k_or_v = torch.from_numpy(
                    real_kv_cache[idx][:, :, :n_past_input_tokens, :]
                ).float()
            else:
                k_or_v = torch.rand(
                    past_values_cache_conf["kv_shape"][idx]
                ).float()
            if force_inputs_dtype is not None:
                k_or_v = k_or_v.to(force_inputs_dtype)
            past_key_values.append(k_or_v)
            in_cache_name = f"in_{node_name}"
            in_cache_names.append(in_cache_name)
            out_cache_names.append(f"out_{node_name}")
            # past s   dynamic_axes[in_cache_name] = {2: "PAST_S"}
            dynamic_axes[in_cache_name] = {2: "P"}
        return in_cache_names, out_cache_names, past_key_values, dynamic_axes
