import argparse
import os
import typing as T
from enum import Enum
from pathlib import Path

import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli, TractNNEF
from torch_to_nnef.log import log
from torch_to_nnef.qtensor.qtract import QTensorTractScaleOnly
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion

try:
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.phi3.configuration_phi3 import Phi3Config

    from torch_to_nnef.llm_tract.models.base import (
        BaseCausal,
        BaseCausalWithDynCacheAndTriu,
    )
except ImportError as exp:
    raise ValueError(
        "Should be used with 'torch_to_nnef[llm_tract]' enabled"
    ) from exp


LOGGER = log.getLogger(__name__)


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


APPLE_OPENELM_STARTSWITH = "apple/OpenELM-"


class OpenELMSlugs(str, Enum):
    MICRO = "apple/OpenELM-270M-Instruct"
    MINI = "apple/OpenELM-450M-Instruct"
    MEDIUM = "apple/OpenELM-1_1B-Instruct"
    BIG = "apple/OpenELM-3B-Instruct"


CUSTOM_CONFIGS: T.Dict[str, T.Any] = {
    PHISlugs.DEBUG: Phi3Config(
        vocab_size=32064,
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_size=256,
        intermediate_size=512,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
    )
}

REMAP_TOKENIZER_FOR_MODEL_STARTING_WITH: T.Dict[str, str] = {
    APPLE_OPENELM_STARTSWITH: LlamaSLugs.LLAMA2_7B_BASE.value,
    PHISlugs.DEBUG.value: PHISlugs.MINI.value,
}


def load_tokenizer(hf_model_slug: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_slug = hf_model_slug
    try:
        tokenizer_slug = next(
            mapped_slug
            for slug_start, mapped_slug in REMAP_TOKENIZER_FOR_MODEL_STARTING_WITH.items()
            if hf_model_slug.startswith(slug_start)
        )
    except StopIteration:
        pass
    return AutoTokenizer.from_pretrained(tokenizer_slug)


def load_model(
    hf_model_slug: str,
    local_dir: T.Optional[T.Union[Path, str]],
    as_float16: bool = False,
):
    model_slug = hf_model_slug

    kwargs: T.Dict[str, T.Any] = {"trust_remote_code": True}
    if as_float16:
        kwargs["torch_dtype"] = "float16"

    custom_config = None
    if hf_model_slug in CUSTOM_CONFIGS:
        custom_config = CUSTOM_CONFIGS[model_slug]
    if custom_config is not None:
        hf_model_causal = AutoModelForCausalLM.from_config(
            custom_config, trust_remote_code=True
        )
        LOGGER.info(f"load custom config: {model_slug}")
    elif local_dir:
        dir_path = Path(local_dir)
        assert dir_path.is_dir(), dir_path
        assert (dir_path / "model.safetensors").is_file(), dir_path
        hf_model_causal = AutoModelForCausalLM.from_pretrained(
            dir_path, **kwargs
        )
        LOGGER.info(f"load '{model_slug}' from local directory: {dir_path}")
    else:
        hf_model_causal = AutoModelForCausalLM.from_pretrained(
            model_slug, **kwargs
        )
        LOGGER.info(
            f"load default trained model from huggingface: {model_slug}"
        )
    return hf_model_causal


class InfosFromSlugAndConfig:
    def __init__(self, model_slug, conf):
        self.conf = conf
        self.model_slug = model_slug
        if model_slug.startswith(APPLE_OPENELM_STARTSWITH):
            self.max_position_embeddings = conf.max_context_length
            self.wrapper_class = BaseCausal
        else:
            self.max_position_embeddings = conf.max_position_embeddings
            self.wrapper_class = BaseCausalWithDynCacheAndTriu

    def get_past_value_cache_conf(self, n_past_input_tokens: int):
        if self.model_slug.startswith(APPLE_OPENELM_STARTSWITH):
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


def quantize_weights_min_max_Q4_0(
    hf_model_causal: nn.Module, args: T.Tuple[T.Any, ...]
):
    with torch.no_grad():
        for name, mod in hf_model_causal.named_modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                LOGGER.info(f"quantize layer: {name}")
                try:
                    q_weight = QTensorTractScaleOnly.build_q4_0_from_min_max_calibration(
                        mod.weight
                    )
                except TorchToNNEFImpossibleQuantization as exp:
                    LOGGER.error(f"quant layer: {name} error: {exp}")
                    continue
                setattr(
                    mod,
                    "weight",
                    nn.Parameter(q_weight, requires_grad=False),
                )

    return hf_model_causal


DEFAULT_COMPRESSION = {"min_max_q4_0": quantize_weights_min_max_Q4_0}


class LLMExport:
    def __init__(
        self,
        hf_model_slug: str,
        local_dir: T.Optional[Path] = None,
        as_float16: bool = False,
    ):
        self.tokenizer = load_tokenizer(hf_model_slug)
        self.hf_model_causal = load_model(
            hf_model_slug, local_dir, as_float16=as_float16
        )
        self.as_float16 = as_float16

        self.model_infos = InfosFromSlugAndConfig(
            hf_model_slug, self.hf_model_causal.config
        )

        self.wrapped_model = self.model_infos.wrapper_class(
            self.hf_model_causal
        )

    def generate_inputs(
        self, n_input_tokens: int = 1, n_past_input_tokens: int = 2
    ):
        test_input = self.tokenizer("Hello, I am happy", return_tensors="pt")
        assert test_input.input_ids.shape[1] >= n_input_tokens
        (
            in_cache_names,
            out_cache_names,
            past_key_values,
            dynamic_axes,
        ) = self.model_infos.build_kv_cache_infos(
            n_past_input_tokens=n_past_input_tokens, as_float16=self.as_float16
        )

        return (
            tuple([test_input.input_ids[:, :n_input_tokens]] + past_key_values),
            in_cache_names,
            out_cache_names,
            dynamic_axes,
        )

    def export_model(
        self,
        export_filepath: Path,
        naming_scheme: VariableNamingScheme = VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
        tract_specific_path: T.Optional[Path] = None,
        tract_specific_version: T.Optional[
            T.Union[SemanticVersion, str]
        ] = None,
        log_level=log.INFO,
    ):
        assert (  # mutualy exclusive arguments
            (tract_specific_path is None and tract_specific_version is None)
            or tract_specific_path is None
            or tract_specific_version is None
        )
        (
            inputs,
            in_cache_names,
            out_cache_names,
            dynamic_axes,
        ) = self.generate_inputs()

        _ = self.wrapped_model(*inputs)

        input_names = ["input_ids"] + in_cache_names
        output_names = ["outputs"] + out_cache_names
        assert (
            len(inputs) == len(input_names) == len(output_names)
        ), f"{len(inputs)} == {len(input_names)} == {len(output_names)}"
        LOGGER.info("start export with 'torch_to_nnef'")
        if tract_specific_version:
            inference_target = TractNNEF(
                SemanticVersion.from_str(tract_specific_version)
                if isinstance(tract_specific_version, str)
                else tract_specific_version
            )
        if tract_specific_path:
            tract_cli_path = Path(tract_specific_path)
            assert tract_cli_path.exists(), tract_cli_path
            tract_cli = TractCli(tract_cli_path)
            inference_target = TractNNEF(
                tract_cli.version, specific_tract_binary_path=tract_cli_path
            )
        else:
            inference_target = TractNNEF.latest()
        inference_target.dynamic_axes = dynamic_axes
        export_model_to_nnef(
            model=self.wrapped_model,
            args=inputs,
            inference_target=inference_target,
            file_path_export=Path(export_filepath),
            input_names=input_names,
            output_names=output_names,
            log_level=log_level,
            nnef_variable_naming_scheme=naming_scheme,
            custom_extensions={
                "tract_assert P >= 0",
                "tract_assert S >= 1",
                f"tract_assert S+P < {self.model_infos.max_position_embeddings}",
            },
        )


def dynamic_load_registry(compression_registry_full_path: str):
    module_str, name = compression_registry_full_path.rsplit(".", maxsplit=1)
    mod = __import__(module_str, fromlist=[""])
    registry = getattr(mod, name)
    assert isinstance(registry, dict)
    return registry


def parser_cli():
    loader_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    full_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    slug_examples = ", ".join(
        [
            f"'{_.value}'"
            for slugsEnums in [LlamaSLugs, PHISlugs, OpenELMSlugs]
            for _ in slugsEnums
        ]
    )
    for parser in [loader_parser, full_parser]:
        parser.add_argument(
            "-e",
            "--export-filepath",
            required=True,
            help="export file path to dump .nnef.tgz",
        )

        parser.add_argument(
            "-s",
            "--model-slug",
            required=True,
            help=f"huggingface slug (web-page 'endpoint') to export by example ({slug_examples})",
        )
        parser.add_argument(
            "-f16",
            "--as-float16",
            action="store_true",
            help="float in 16 bits",
        )
        parser.add_argument(
            "--compression-registry",
            default="torch_to_nnef.llm_tract.cli.DEFAULT_COMPRESSION",
            help="Compression registry to load "
            "(should be a Dict[str, Callable(model, args)]), "
            "can be specified to load arbitrary compression library.",
        )
        parser.add_argument(
            "-d",
            "--local-dir",
            help="local dir containing .safetensors compatible with openELM"
            " model size specified in slug",
        )
        parser.add_argument(
            "-n",
            "--naming-scheme",
            default=VariableNamingScheme.NATURAL_VERBOSE_CAMEL.value,
            choices=[vns.value for vns in VariableNamingScheme],
            help="display debug information",
        )
        parser.add_argument(
            "--tract-specific-path",
            required=False,
            help="tract specific path (instead of latest version)",
        )
        parser.add_argument(
            "--tract-specific-version",
            required=False,
            help="tract specific version",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="display debug information",
        )
    # == hack by using 1st parser without help to fill dynamically 2nd parser ==
    args, _ = loader_parser.parse_known_args()
    possible_compression_ids = list(
        dynamic_load_registry(args.compression_registry).keys()
    )
    parser = full_parser
    parser.add_argument(
        "-c",
        "--compression-method",
        choices=possible_compression_ids,
        help="possible compression method to apply on Model before export",
    )
    return parser.parse_args()


def main():
    args = parser_cli()
    log_level = log.INFO
    if args.verbose:
        log_level = log.DEBUG
    log.getLogger().setLevel(log_level)

    with torch.no_grad():
        try:
            exporter = LLMExport(
                args.model_slug, args.local_dir, args.as_float16
            )
        except OSError as exp:
            if "gated repo" in exp.args[0]:
                print(exp.args[0])
                login()
                exporter = LLMExport(
                    args.model_slug, args.local_dir, args.as_float16
                )
            else:
                raise exp
        if args.compression_method:
            LOGGER.info(f"start compresssion: {args.compression_method}")
            registry = dynamic_load_registry(args.compression_registry)
            inps, *_ = exporter.generate_inputs()
            exporter.wrapped_model = registry[args.compression_method](
                exporter.wrapped_model, inps
            )
            LOGGER.info(
                f"successfully applied compression: {args.compression_method}"
            )
        exporter.export_model(
            args.export_filepath,
            naming_scheme=args.naming_scheme,
            tract_specific_path=args.tract_specific_path,
            tract_specific_version=args.tract_specific_version,
            log_level=log_level,
        )


if __name__ == "__main__":
    main()
