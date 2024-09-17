""" Export any huggingface transformers LLM to tract NNEF

With options to compress it to Q4_0 and use float16

"""
import argparse
import os
import typing as T
from enum import Enum
from functools import partial
from pathlib import Path

import torch
from torch import nn
from transformers import GenerationConfig

from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli, TractNNEF, build_io
from torch_to_nnef.log import log
from torch_to_nnef.qtensor.qtract import (
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion

try:
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from torch_to_nnef.llm_tract.models.base import (
        BaseCausal,
        BaseCausalWithDynCacheAndTriu,
    )
except (ModuleNotFoundError, ImportError) as exp:
    raise ValueError(
        "Should be used with 'torch_to_nnef[llm_tract]' enabled"
    ) from exp


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


def find_subdir_with_filename_in(dirpath: Path, filename: str) -> Path:
    """Find a subdir with filename in it"""
    found_dirs = {p.parent for p in dirpath.glob(f"**/{filename}")}
    if 1 < len(found_dirs):
        raise ValueError(
            f"Found {len(found_dirs)} dirs for with '{filename}' file. "
            f"found_dirs={found_dirs}. "
            "Unable to decide which one should selected..."
        )
    return found_dirs.pop()


def load_tokenizer(
    config,
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_slug = REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG.get(
        config.model_type, hf_model_slug
    )
    if tokenizer_slug is None:
        assert local_dir is not None
    if local_dir is not None:
        local_dir = find_subdir_with_filename_in(local_dir, "tokenizer.json")
    return AutoTokenizer.from_pretrained(local_dir or tokenizer_slug)


def load_model(
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    as_float16: bool = False,
):
    kwargs: T.Dict[str, T.Any] = {"trust_remote_code": True}
    if as_float16:
        kwargs["torch_dtype"] = "float16"

    custom_config = CUSTOM_CONFIGS.get(hf_model_slug or "")
    if custom_config is not None:
        hf_model_causal = AutoModelForCausalLM.from_config(
            custom_config, trust_remote_code=True
        )
        LOGGER.info(f"load custom config: '{hf_model_slug}'")
    elif local_dir:
        dir_path = find_subdir_with_filename_in(local_dir, "config.json")
        assert dir_path.is_dir(), dir_path
        assert (dir_path / "model.safetensors").is_file(), dir_path
        hf_model_causal = AutoModelForCausalLM.from_pretrained(
            dir_path, **kwargs
        )
        LOGGER.info(
            f"load '{hf_model_causal.config.model_type}' "
            f"from local directory: {dir_path}"
        )
    else:
        hf_model_causal = AutoModelForCausalLM.from_pretrained(
            hf_model_slug, **kwargs
        )
        LOGGER.info(
            f"load default trained model from huggingface: '{hf_model_slug}'"
        )
    return hf_model_causal


class InfosFromSlugAndConfig:
    def __init__(self, model_slug, conf):
        self.conf = conf
        self.model_slug = model_slug
        if conf.model_type == "openelm":
            self.max_position_embeddings = conf.max_context_length
        else:
            self.max_position_embeddings = conf.max_position_embeddings

        if conf.model_type in ["llama", "phi"] or conf.model_type.startswith(
            "gemma"
        ):
            self.wrapper_class = BaseCausalWithDynCacheAndTriu
        elif conf.model_type == "openelm":
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


def quantize_weights_min_max_Q4_0(
    hf_model_causal: nn.Module, args: T.Tuple[T.Any, ...]
):
    with torch.no_grad():
        for name, mod in hf_model_causal.named_modules():
            if isinstance(mod, (nn.Linear)):
                # NOTE: nn.Embedding will likely need per channel implem in Tract
                LOGGER.info(f"quantize layer: {name}")
                try:
                    q_weight = fp_to_tract_q4_0_with_min_max_calibration(
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


class StateLessF32LayerNorm(nn.Module):
    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        normalized_shape: T.List[int],
        weight: T.Optional[torch.Tensor] = None,
        bias: T.Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ):
        """Upcast and apply layer norm in f32.
        This is because f16 is not implemented on CPU in PyTorch
        (only GPU) as of torch 2.2.2 (2024-09-10):
        ```
        RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
        ```
        """
        return torch.nn.functional.original_layer_norm(
            input.to(torch.float32),
            normalized_shape=normalized_shape,
            weight=weight if weight is None else weight.to(torch.float32),
            bias=bias if bias is None else bias.to(torch.float32),
            eps=eps,
        ).to(torch.float16)


class LLMExporter:
    def __init__(
        self,
        hf_model_slug: T.Optional[str] = None,
        local_dir: T.Optional[Path] = None,
        as_float16: bool = False,
    ):
        local_dir = Path(local_dir) if local_dir else None
        assert hf_model_slug is not None or local_dir is not None
        self.hf_model_causal = load_model(
            hf_model_slug, local_dir, as_float16=as_float16
        )
        self.tokenizer = load_tokenizer(
            self.hf_model_causal.config,
            hf_model_slug=hf_model_slug,
            local_dir=local_dir,
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

    def generate_test_text(self, prompt: str = "Alan Turing was"):
        LOGGER.info("start to generate testing text from loaded model:")
        generation_config = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            top_k=50,
            eos_token_id=self.hf_model_causal.config.eos_token_id,
        )
        iids = self.hf_model_causal.generate(
            self.tokenizer.encode(prompt, return_tensors="pt"),
            generation_config=generation_config,
        )
        text = self.tokenizer.decode(iids[0])
        LOGGER.info(f"generated text: {text}")

    def apply_f16_fixes(self):
        """Align float dtype arguments in few graph ops

        Indeed all LLM are trained using GPU/TPU/CPU kernels
        related PyTorch backend support f16 dtype in some operators
        contrary to PyTorch CPU inference (@ 2024-09-09).

        To solve this issue we monkey patch in this cli few functional API.
        """

        torch.nn.functional.original_layer_norm = torch.nn.functional.layer_norm
        torch.nn.functional.layer_norm = StateLessF32LayerNorm()

    def export_model(
        self,
        export_dirpath: Path,
        naming_scheme: VariableNamingScheme = VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
        tract_specific_path: T.Optional[Path] = None,
        tract_specific_version: T.Optional[
            T.Union[SemanticVersion, str]
        ] = None,
        log_level=log.INFO,
        dump_with_tokenizer_and_conf: bool = False,
    ):
        assert not export_dirpath.exists(), export_dirpath
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

        if dump_with_tokenizer_and_conf:
            self.hf_model_causal.config.save_pretrained(export_dirpath)
            self.tokenizer.save_pretrained(export_dirpath)
        # Add io.npz test in exproted dir for dbg purpose
        test_dir = export_dirpath / "tests"
        test_dir.mkdir(parents=True)
        build_io(
            self.wrapped_model,
            inputs,
            io_npz_path=test_dir / "io.npz",
            input_names=input_names,
            output_names=output_names,
        )
        export_model_to_nnef(
            model=self.wrapped_model,
            args=inputs,
            inference_target=inference_target,
            file_path_export=export_dirpath / "model.nnef.tgz",
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


def parser_cli(
    fn_parser_adder: T.Optional[
        T.Callable[[argparse.ArgumentParser], None]
    ] = None
):
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
            "--export-dirpath",
            required=True,
            help="export dir path to dump tokenizer infos, model config.json, model.nnef.tgz",
        )

        parser.add_argument(
            "-s",
            "--model-slug",
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
            "-td",
            "--test-display-token-gens",
            action="store_true",
            help="Generate 50 tokens with model, "
            "and after f16/compression if activated "
            "this is meant as a way to detect spurious precision problems "
            "early",
        )
        parser.add_argument(
            "-dwtac",
            "--dump-with-tokenizer-and-conf",
            action="store_true",
            help="dump tokenizer and conf at same dir as model",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="display debug information",
        )
        if fn_parser_adder is not None:
            fn_parser_adder(parser)
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


def dump_llm(
    export_dirpath: T.Union[str, Path],
    model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    tract_specific_path: T.Optional[Path] = None,
    tract_specific_version: T.Optional[str] = None,
    as_float16: bool = False,
    compression_method: T.Optional[str] = None,
    compression_registry: str = "torch_to_nnef.llm_tract.cli.DEFAULT_COMPRESSION",
    test_display_token_gens: bool = False,
    naming_scheme: VariableNamingScheme = VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
    dump_with_tokenizer_and_conf: bool = False,
    log_level: int = log.INFO,
) -> T.Tuple[Path, LLMExporter]:
    """Util to export LLM model"""
    export_dirpath = Path(export_dirpath)
    if export_dirpath.exists():
        raise ValueError(
            f"'export_dirpath' should not exist but found: '{export_dirpath}'"
        )
    log.getLogger().setLevel(log_level)
    if model_slug is None and local_dir is None:
        raise ValueError(
            "You should either provide `model_slug` or a `local_dir`"
        )
    with torch.no_grad():
        try:
            exporter = LLMExporter(model_slug, local_dir, as_float16)
        except OSError as exp:
            if "gated repo" in exp.args[0]:
                print(exp.args[0])
                login()
                exporter = LLMExporter(model_slug, local_dir, as_float16)
            else:
                raise exp
        if test_display_token_gens:
            exporter.generate_test_text()
        if compression_method:
            LOGGER.info(f"start compresssion: {compression_method}")
            registry = dynamic_load_registry(compression_registry)
            inps, *_ = exporter.generate_inputs()
            exporter.wrapped_model = registry[compression_method](
                exporter.wrapped_model, inps
            )
            LOGGER.info(
                f"successfully applied compression: {compression_method}"
            )
        if as_float16:
            exporter.apply_f16_fixes()

        if test_display_token_gens and (compression_method or as_float16):
            LOGGER.info("check testing text post compression/f16 conversion:")
            exporter.generate_test_text()
        exporter.export_model(
            Path(export_dirpath),
            naming_scheme=naming_scheme,
            tract_specific_path=tract_specific_path,
            tract_specific_version=tract_specific_version,
            log_level=log_level,
            dump_with_tokenizer_and_conf=dump_with_tokenizer_and_conf,
        )
    return export_dirpath, exporter


def main():
    args = parser_cli()
    log_level = log.INFO
    if args.verbose:
        log_level = log.DEBUG
    kwargs = vars(args)
    del kwargs["verbose"]
    dump_llm(
        **kwargs,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
