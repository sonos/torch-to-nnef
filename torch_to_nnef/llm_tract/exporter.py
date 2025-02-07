import json
import os
import typing as T
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig

from torch_to_nnef.exceptions import (
    TorchToNNEFConsistencyError,
    TorchToNNEFNotFoundFile,
)
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    TractNNEF,
    build_io,
)
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

# NOTE: this assume LLM exported will always 'speak' english
# which may not be the case in the future
# (let's revisit that if we come to it)
EN_SAMPLE_TEXT = """
Electricity is the set of physical phenomena
associated with the presence and motion of matter
possessing an electric charge.
Electricity is related to magnetism,
both being part of the phenomenon of electromagnetism,
as described by Maxwell's equations.
Common phenomena are related to electricity,
including lightning, static electricity, electric heating,
electric discharges and many others.
The presence of either a positive or negative electric charge
produces an electric field.
The motion of electric charges is an electric current
and produces a magnetic field.
In most applications, Coulomb's law determines
the force acting on an electric charge.
Electric potential is the work done to move an electric charge
from one point to another within an electric field,
typically measured in volts.
""".strip().replace(
    "\n", " "
)


def _load_exporter_from(
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    as_float16: bool = False,
):
    local_dir = Path(local_dir) if local_dir else None
    assert hf_model_slug is not None or local_dir is not None
    hf_model_causal = load_model(
        hf_model_slug, local_dir, as_float16=as_float16
    )
    tokenizer = load_tokenizer(
        hf_model_causal.config,
        hf_model_slug=hf_model_slug,
        local_dir=local_dir,
    )
    return LLMExporter(
        hf_model_causal,
        tokenizer,
        as_float16=as_float16,
        local_dir=local_dir,
    )


class LLMExporter:
    def __init__(
        self,
        hf_model_causal: nn.Module,
        tokenizer: AutoTokenizer,
        as_float16: bool = False,
        local_dir: T.Optional[Path] = None,
    ):
        self.hf_model_causal = hf_model_causal
        self.tokenizer = tokenizer
        self.as_float16 = as_float16
        self.local_dir = local_dir

        self.model_infos = HFConfigHelper(
            self.hf_model_causal.config._name_or_path,
            self.hf_model_causal.config,
        )

        self.wrapped_model = self.model_infos.wrapper_class(
            self.hf_model_causal
        )
        self._inference_target_options = {}

    def __repr__(self):
        n_params = self.model_n_params
        model_name = self.hf_model_causal.config._name_or_path
        tokenizer_name = self.tokenizer.name_or_path
        vocab_size = self.tokenizer.vocab_size
        return (
            f"<{self.__class__.__name__} "
            f"model={model_name}(n_params={n_params:,}) "
            f"tokenizer={tokenizer_name}(vocab_size={vocab_size:,})>"
        )

    @property
    def model_n_params(self) -> int:
        return sum([_.numel() for _ in self.hf_model_causal.parameters()])

    @staticmethod
    def load(
        model_slug: T.Optional[str] = None,
        local_dir: T.Optional[Path] = None,
        as_float16: bool = False,
    ):
        """Load from either huggingface model slug hub or local_dir"""
        if as_float16 and torch_version() < "2.0.0":
            LOGGER.warning(
                "float16 with CPU backend is limited in PyTorch 1.X "
                "(if issues, try to use torch>2.0)"
            )
        with torch.no_grad():
            try:
                exporter = _load_exporter_from(
                    model_slug, local_dir, as_float16
                )
            except OSError as exp:
                if "gated repo" in exp.args[0]:
                    print(exp.args[0])
                    login()
                    exporter = _load_exporter_from(
                        model_slug, local_dir, as_float16
                    )
                else:
                    raise exp
        return exporter

    def check_wrapper_io(self):
        """Checking that wrapper given consistent outputs compared to vanilla model"""
        (
            inputs,
            _,
            out_cache_names,
            _,
        ) = self.generate_inputs_io_names_and_dynaxes()
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
            ref = ref.float()
            cand = cand.float()
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

    def generate_inputs_io_names_and_dynaxes(
        self,
        n_input_tokens: int = 1,
        n_past_input_tokens: int = 2,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
    ):
        test_input = self.tokenizer(EN_SAMPLE_TEXT, return_tensors="pt")
        assert test_input.input_ids.shape[1] >= n_input_tokens
        (
            in_cache_names,
            out_cache_names,
            past_key_values,
            dynamic_axes,
        ) = self.model_infos.build_kv_cache_infos(
            n_past_input_tokens=n_past_input_tokens,
            as_float16=self.as_float16,
            real_kv_cache=real_kv_cache,
        )

        input_names = ["input_ids"] + in_cache_names
        output_names = ["outputs"] + out_cache_names
        inputs = tuple(
            [test_input.input_ids[:, :n_input_tokens]] + past_key_values
        )
        assert (
            len(inputs) == len(input_names) == len(output_names)
        ), f"{len(inputs)} == {len(input_names)} == {len(output_names)}"
        return (
            inputs,
            input_names,
            output_names,
            dynamic_axes,
        )

    def build_io_npz(self, io_npz_path: Path, *args, **kwargs):
        (
            inputs,
            input_names,
            output_names,
            _,
        ) = self.generate_inputs_io_names_and_dynaxes(*args, **kwargs)
        build_io(
            self.wrapped_model,
            inputs,
            io_npz_path=io_npz_path,
            input_names=input_names,
            output_names=output_names,
        )

    def dump_all_io_npz_kind(
        self, io_npz_dirpath: Path, size: int = 6
    ) -> T.List[Path]:
        """Realistic dump of IO's"""
        half = size // 2
        prompt_npz_filepath = io_npz_dirpath / "prompt_io.npz"
        self.build_io_npz(
            prompt_npz_filepath,
            n_input_tokens=size,
            n_past_input_tokens=0,
        )
        res = {**np.load(prompt_npz_filepath)}
        out_kv = {}
        for k, v in res.items():
            if k.startswith("out_cache_key_"):
                layer_idx = int(k.replace("out_cache_key_", ""))
                out_kv[layer_idx] = [v, res[f"out_cache_value_{layer_idx}"]]
        real_kv_cache = [
            _
            for idx in range(max(list(out_kv.keys())) + 1)
            for _ in out_kv[idx]
        ]
        prompt_with_past_npz_filepath = (
            io_npz_dirpath / "prompt_with_past_io.npz"
        )
        try:
            self.build_io_npz(
                prompt_with_past_npz_filepath,
                n_input_tokens=half,
                n_past_input_tokens=half,
                real_kv_cache=real_kv_cache,
            )
        except Exception as exp:
            LOGGER.error(
                "Prompt with past, does not run in PyTorch "
                f"(likely modeling limit): {exp}"
            )
        text_gen_npz_filepath = io_npz_dirpath / "text_generation_io.npz"
        self.build_io_npz(
            text_gen_npz_filepath,
            n_input_tokens=1,
            n_past_input_tokens=size - 1,
            real_kv_cache=real_kv_cache,
        )
        return [
            prompt_npz_filepath,
            prompt_with_past_npz_filepath,
            text_gen_npz_filepath,
        ]

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

    def _update_inference_target_options(self, inference_target):
        inference_target.__dict__.update(self._inference_target_options)

    def apply_f16_fixes(self):
        """Align float dtype arguments in few graph ops

        Indeed all LLM are trained using GPU/TPU/CPU kernels
        related PyTorch backend support f16 dtype in some operators
        contrary to PyTorch CPU inference (@ 2024-09-09).

        To solve this issue we monkey patch in this cli few functional API.
        """

        torch.nn.functional.original_layer_norm = torch.nn.functional.layer_norm
        torch.nn.functional.layer_norm = StateLessF32LayerNorm()
        if self.model_infos.conf.model_type == "qwen2":
            self._inference_target_options = {
                "force_attention_inner_in_f32": True,
                "force_linear_accumulation_in_f32": True,
            }

    def prepare(
        self,
        compression_method: T.Optional[str] = None,
        compression_registry: str = "torch_to_nnef.llm_tract.cli.DEFAULT_COMPRESSION",
        test_display_token_gens: bool = False,
        wrapper_io_check: bool = True,
        export_dirpath: T.Optional[Path] = None,
        log_level: int = log.INFO,
    ):
        """Prepare model to export (f16/compression/checks...)"""
        log.getLogger().setLevel(log_level)
        with torch.no_grad():
            if test_display_token_gens:
                self.generate_test_text()

        # compression method may sometime need
        # gradient optimization so avoid context manager no_grad
        if compression_method:
            LOGGER.info(f"start compresssion: {compression_method}")
            registry = dynamic_load_registry(compression_registry)
            self.wrapped_model = registry[compression_method](
                wrapped_model=self.wrapped_model,
                tokenizer=self.tokenizer,
                # may be usefull to dump compression evaluations results
                export_dirpath=export_dirpath,
                # may be usefull to perform internal evaluations
                # when more data than just llm torch is available
                local_dir=self.local_dir,
            )
            LOGGER.info(
                f"successfully applied compression: {compression_method}"
            )

        with torch.no_grad():
            if self.as_float16:
                self.apply_f16_fixes()

            if test_display_token_gens and (
                compression_method or self.as_float16
            ):
                LOGGER.info(
                    "check testing text post compression/f16 conversion:"
                )
                self.generate_test_text()
            if wrapper_io_check:
                self.check_wrapper_io()

    def export_model(
        self,
        export_dirpath: Path,
        naming_scheme: VariableNamingScheme = VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
        tract_specific_path: T.Optional[Path] = None,
        tract_specific_version: T.Optional[
            T.Union[SemanticVersion, str]
        ] = None,
        tract_specific_properties: T.Optional[str] = None,
        log_level=log.INFO,
        dump_with_tokenizer_and_conf: bool = False,
        check_inference_modes: bool = True,
        sample_generation_total_size: int = 0,
        no_verify: bool = False,
        tract_check_io_tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE,
    ):
        """Export model has is currently in self.hf_model_causal

        and dump some npz tests to check io latter-on
        """
        with torch.no_grad():
            assert not export_dirpath.exists(), export_dirpath
            assert sample_generation_total_size >= 2
            assert (  # mutualy exclusive arguments
                (tract_specific_path is None and tract_specific_version is None)
                or tract_specific_path is None
                or tract_specific_version is None
            )
            (
                inputs,
                input_names,
                output_names,
                dynamic_axes,
            ) = self.generate_inputs_io_names_and_dynaxes()

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
            inference_target.specific_properties = tract_specific_properties
            inference_target.check_io_tolerance = tract_check_io_tolerance
            self._update_inference_target_options(inference_target)
            if no_verify:
                LOGGER.info(
                    "tract inference is not checked because 'no_verify=True'"
                )
            inference_target.check_io = not no_verify

            # Add io.npz test in exproted dir for dbg purpose
            test_dir = export_dirpath / "tests"
            test_dir.mkdir(parents=True)

            if check_inference_modes and sample_generation_total_size > 0:
                LOGGER.info(
                    "'inference mode' evaluation started with "
                    f"sample_generation_total_size={sample_generation_total_size}"
                )
                modes = [
                    p.with_suffix("").name.replace("_io", "")
                    for p in self.dump_all_io_npz_kind(
                        test_dir, size=sample_generation_total_size
                    )
                ]
                with (export_dirpath / "modes.json").open(
                    "w", encoding="utf8"
                ) as fh:
                    json.dump({"pytorch_supported_modes": modes}, fh)
                LOGGER.info("'inference mode' evaluation data generated")
            else:
                LOGGER.info("'inference mode' evaluation skipped")

            if dump_with_tokenizer_and_conf:
                self.hf_model_causal.config.save_pretrained(export_dirpath)
                self.tokenizer.save_pretrained(export_dirpath)

            build_io(
                self.wrapped_model,
                inputs,
                io_npz_path=test_dir / "export_io.npz",
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
                custom_extensions=[
                    "tract_assert P >= 0",
                    "tract_assert S >= 1",
                    "tract_assert S+P < "
                    f"{self.model_infos.max_position_embeddings}",
                    # information about modes
                    "tract_assert tg: S==1",  # text generation
                    "tract_assert pp: P==0",  # prompt processing
                ],
            )

    def dump(
        self,
        export_dirpath: T.Union[str, Path],
        tract_specific_path: T.Optional[Path] = None,
        tract_specific_version: T.Optional[str] = None,
        tract_specific_properties: T.Optional[str] = None,
        compression_method: T.Optional[str] = None,
        compression_registry: str = "torch_to_nnef.llm_tract.compress.DEFAULT_COMPRESSION",
        test_display_token_gens: bool = False,
        naming_scheme: VariableNamingScheme = VariableNamingScheme.NATURAL_VERBOSE_CAMEL,
        dump_with_tokenizer_and_conf: bool = False,
        check_inference_modes: bool = True,
        wrapper_io_check: bool = True,
        log_level: int = log.INFO,
        sample_generation_total_size: int = 6,
        no_verify: bool = False,
        ignore_already_exist_dir: bool = False,
        force_f32_attention: T.Optional[bool] = None,
        force_f32_linear_accumulator: T.Optional[bool] = None,
        tract_check_io_tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE,
    ):
        """prepare and export model to NNEF"""
        if force_f32_attention:
            self._inference_target_options["force_attention_inner_in_f32"] = (
                True
            )
        if force_f32_linear_accumulator:
            self._inference_target_options[
                "force_linear_accumulation_in_f32"
            ] = True
        export_dirpath = Path(export_dirpath)
        if no_verify and wrapper_io_check:
            LOGGER.info(
                "force disable 'wrapper_io_check' because 'no_verify=True'"
            )
            wrapper_io_check = False
        if no_verify and test_display_token_gens:
            LOGGER.info(
                "force disable 'test_display_token_gens' because "
                "'no_verify=True'"
            )
            test_display_token_gens = False
        if export_dirpath.exists() and not ignore_already_exist_dir:
            raise ValueError(
                "'export_dirpath' should not exist but "
                f"found: '{export_dirpath}'"
            )

        self.prepare(
            compression_method=compression_method,
            compression_registry=compression_registry,
            test_display_token_gens=test_display_token_gens,
            wrapper_io_check=wrapper_io_check,
            export_dirpath=export_dirpath,
            log_level=log_level,
        )
        tract_specific_properties = tract_specific_properties or {}
        tract_specific_properties.update(
            {
                "hf_model_type": self.model_infos.conf.model_type,
                "n_parameters": str(self.model_n_params),
                "as_float16": "1" if self.as_float16 else "0",
            }
        )
        if compression_method is not None:
            tract_specific_properties.update(
                {
                    "compression_method": compression_method,
                    "compression_registry": compression_registry,
                }
            )
        if not self.hf_model_causal.config._name_or_path.startswith("/tmp"):
            tract_specific_properties["name_or_path"] = (
                self.hf_model_causal.config._name_or_path
            )
        if hasattr(self.hf_model_causal, "peft_config"):
            for k, conf in self.hf_model_causal.peft_config.items():
                tract_specific_properties[f"peft_{k}_type"] = (
                    self.hf_model_causal.peft_config[k].peft_type.value
                )
                tract_specific_properties[f"peft_{k}_target_modules"] = (
                    ",".join(self.hf_model_causal.peft_config[k].target_modules)
                )
        self.export_model(
            export_dirpath,
            naming_scheme=naming_scheme,
            tract_specific_path=tract_specific_path,
            tract_specific_version=tract_specific_version,
            tract_specific_properties=tract_specific_properties,
            log_level=log_level,
            dump_with_tokenizer_and_conf=dump_with_tokenizer_and_conf,
            check_inference_modes=check_inference_modes,
            sample_generation_total_size=sample_generation_total_size,
            no_verify=no_verify,
            tract_check_io_tolerance=tract_check_io_tolerance,
        )


def find_subdir_with_filename_in(dirpath: Path, filename: str) -> Path:
    """Find a subdir with filename in it"""
    found_dirs = {p.parent for p in dirpath.glob(f"**/{filename}")}
    if not (0 < len(found_dirs) < 2):
        raise TorchToNNEFNotFoundFile(
            f"Found {len(found_dirs)} dirs for with '{filename}' file. "
            f"found_dirs={found_dirs}. "
            + (
                "Unable to decide which one should selected..."
                if len(found_dirs) > 1
                else "Is it a valid model directory ?"
            )
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


def _try_load_peft(dir_path, kwargs, exp):
    # pylint: disable-next=import-outside-toplevel
    from peft import PeftModel

    # likely an embedding issue with added tokens
    with (dir_path / "adapter_config.json").open("r", encoding="utf8") as fh:
        dic = json.load(fh)
    hf_model_causal = AutoModelForCausalLM.from_pretrained(
        dic["base_model_name_or_path"], **kwargs
    )
    msg = "Error(s) in loading state_dict for"
    if exp.args[0].startswith(msg) and "size mismatch for" in exp.args[0]:
        new_tokenizer_len = int(exp.args[0].split("[")[1].split(",")[0])
        hf_model_causal.resize_token_embeddings(new_tokenizer_len)
        print("new_tokenizer_len:", new_tokenizer_len)

    hf_model_causal = PeftModel.from_pretrained(hf_model_causal, dir_path)
    LOGGER.info("loaded a PEFT model with resized token embeddings")
    return hf_model_causal


def assert_model_safetensors_exists(dir_path):
    assert (
        "model" in p.name and p.name.endswith(".safetensors")
        for p in dir_path.iterdir()
    ), dir_path


def load_peft_model(local_dir, kwargs):
    """Load PEFT adapted models.

    Try to avoid direct reference to tokenizer object/config
    to limit dependencies of the function

    While also trying to be robust to 'wrong' key/values

    """
    dir_path = find_subdir_with_filename_in(local_dir, "adapter_config.json")
    assert dir_path.is_dir(), dir_path
    assert_model_safetensors_exists(dir_path)

    while True:
        try:
            hf_model_causal = AutoModelForCausalLM.from_pretrained(
                dir_path, **kwargs
            )
        except ValueError as exp:
            msg = "Should have a `model_type` key in its config.json,"
            if msg in exp.args[0]:
                return _try_load_peft(dir_path, kwargs, exp)
            raise exp
        except OSError as exp:
            if all(
                exp.args[0] in _
                for _ in ["OSError: Error no file named", "found in directory"]
            ):
                print("here")
                __import__("ipdb").set_trace()
                continue
            raise exp
        except RuntimeError as exp:
            msg = "Error(s) in loading state_dict for"
            if (
                exp.args[0].startswith(msg)
                and "size mismatch for" in exp.args[0]
            ):
                return _try_load_peft(dir_path, kwargs, exp)
            raise exp
        except TypeError as exp:
            msg = "__init__() got an unexpected keyword argument '"
            if exp.args[0].startswith(msg):
                with (dir_path / "adapter_config.json").open(
                    "r", encoding="utf8"
                ) as fh:
                    dic = json.load(fh)
                key = exp.args[0].split(msg)[-1][:-1]
                del dic[key]
                with (dir_path / "adapter_config.json").open(
                    "w", encoding="utf8"
                ) as fh:
                    json.dump(dic, fh, indent=2)
                continue
            raise exp
        return hf_model_causal


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
        try:
            dir_path = find_subdir_with_filename_in(local_dir, "config.json")
            assert dir_path.is_dir(), dir_path
            assert_model_safetensors_exists(dir_path)
            hf_model_causal = AutoModelForCausalLM.from_pretrained(
                dir_path, **kwargs
            )
            LOGGER.info(
                f"load '{hf_model_causal.config.model_type}' "
                f"from local directory: {dir_path}"
            )
        except TorchToNNEFNotFoundFile:
            hf_model_causal = load_peft_model(local_dir, kwargs)
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
        operating_dtype = torch.float32
        return torch.nn.functional.original_layer_norm(
            input.to(operating_dtype),
            normalized_shape=normalized_shape,
            weight=weight if weight is None else weight.to(operating_dtype),
            bias=bias if bias is None else bias.to(operating_dtype),
            eps=eps,
        ).to(input.dtype)


def dump_llm(
    model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    as_float16: bool = False,
    *args,
    **kwargs,
) -> T.Tuple[T.Union[Path, None], LLMExporter]:
    """Util to export LLM model"""
    exporter = LLMExporter.load(model_slug, local_dir, as_float16)
    if isinstance(kwargs.get("tract_check_io_tolerance"), str):
        kwargs["tract_check_io_tolerance"] = TractCheckTolerance(
            kwargs["tract_check_io_tolerance"]
        )
    exporter.dump(*args, **kwargs)
    export_path = kwargs.get("export_dirpath", args[0] if args else None)
    return (
        Path(export_path) if export_path else None,
        exporter,
    )
