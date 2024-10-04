import os
import typing as T
from pathlib import Path

import torch
from torch import nn
from transformers import GenerationConfig

from torch_to_nnef.exceptions import TorchToNNEFConsistencyError
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli, TractNNEF, build_io
from torch_to_nnef.llm_tract.compress import dynamic_load_registry
from torch_to_nnef.llm_tract.config import (
    CUSTOM_CONFIGS,
    REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG,
    HFConfigHelper,
)
from torch_to_nnef.log import log
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion, torch_version

try:
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from torch_to_nnef.llm_tract.models.base import (
        build_past_kv_dyn_cache,
        build_past_kv_list,
    )
except (ModuleNotFoundError, ImportError) as exp:
    raise ValueError(
        "Should be used with 'torch_to_nnef[llm_tract]' enabled"
    ) from exp

LOGGER = log.getLogger(__name__)


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

        self.model_infos = HFConfigHelper(
            hf_model_slug, self.hf_model_causal.config
        )

        self.wrapped_model = self.model_infos.wrapper_class(
            self.hf_model_causal
        )

    def chech_wrapper_io(self):
        """Checking that wrapper given consistent outputs compared to vanilla model"""
        (
            inputs,
            _,
            out_cache_names,
            _,
        ) = self.generate_inputs()
        wrapped_outs = self.wrapped_model(*inputs)
        if self.wrapped_model.with_dyn_cache:
            past_key_values = build_past_kv_dyn_cache(inputs[1:])
        else:
            past_key_values = build_past_kv_list(inputs[1:])
        outs = self.hf_model_causal(
            input_ids=inputs[0],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        pkv = outs["past_key_values"]
        if self.wrapped_model.with_dyn_cache:
            pkv = pkv.to_legacy_cache()
        out_pkv = [t for kv in pkv for t in kv]

        def err_check(output_name: str, ref: torch.Tensor, cand: torch.Tensor):
            if not torch.allclose(
                ref, cand, atol=1e-3 if self.as_float16 else 1e-4
            ):
                msg = (
                    f"Model: {self.hf_model_causal.__class__} wrapped "
                    f"with: {self.wrapped_model.__class__}, "
                    "give inconsistent results compared to "
                    f"vanilla in '{output_name}': "
                    f"avg diff: {(ref - cand).abs().mean():0.4f}. "
                    "Likely need a torch_to_nnef fix."
                )
                log.error(msg)
                raise TorchToNNEFConsistencyError(msg)

        err_check("logits", wrapped_outs[0], outs["logits"])
        for kv_name, ref, cand in zip(
            out_cache_names, out_pkv, wrapped_outs[1:]
        ):
            err_check(kv_name, ref, cand)
        LOGGER.info(
            f"In PyTorch wrapped_model:{self.model_infos.wrapper_class} "
            f"provide same results as {self.hf_model_causal.__class__}"
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


def prep_exporter(
    model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    as_float16: bool = False,
    compression_method: T.Optional[str] = None,
    compression_registry: str = "torch_to_nnef.llm_tract.cli.DEFAULT_COMPRESSION",
    test_display_token_gens: bool = False,
    wrapper_io_check: bool = True,
    log_level: int = log.INFO,
) -> LLMExporter:
    """Util to prepare export (loading/f16/compression/...) LLM model"""
    log.getLogger().setLevel(log_level)
    if as_float16 and torch_version() < "2.0.0":
        LOGGER.warning(
            "float16 with CPU backend is limited in PyTorch 1.X "
            "(if issues, try to use torch>2.0)"
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
        if wrapper_io_check:
            exporter.chech_wrapper_io()
    return exporter


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
    wrapper_io_check: bool = True,
    log_level: int = log.INFO,
) -> T.Tuple[Path, LLMExporter]:
    """Util to export LLM model"""
    export_dirpath = Path(export_dirpath)
    if export_dirpath.exists():
        raise ValueError(
            f"'export_dirpath' should not exist but found: '{export_dirpath}'"
        )
    exporter = prep_exporter(
        model_slug=model_slug,
        local_dir=local_dir,
        as_float16=as_float16,
        compression_method=compression_method,
        compression_registry=compression_registry,
        test_display_token_gens=test_display_token_gens,
        wrapper_io_check=wrapper_io_check,
        log_level=log_level,
    )
    with torch.no_grad():
        exporter.export_model(
            export_dirpath,
            naming_scheme=naming_scheme,
            tract_specific_path=tract_specific_path,
            tract_specific_version=tract_specific_version,
            log_level=log_level,
            dump_with_tokenizer_and_conf=dump_with_tokenizer_and_conf,
        )
    return export_dirpath, exporter
