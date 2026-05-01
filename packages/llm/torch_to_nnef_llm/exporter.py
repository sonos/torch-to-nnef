import json
import logging
import time
import typing as T
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_to_nnef.compress import (
    DEFAULT_COMPRESSION_REGISTRY,
    dynamic_load_registry,
)
from torch_to_nnef.exceptions import (
    T2NErrorConsistency,
    T2NErrorMisuse,
    T2NErrorNotImplemented,
    T2NErrorRuntime,
)
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    TractNNEF,
    build_io,
)
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import (
    INJECTED,
    SemanticVersion,
    T2NExtra,
    require_extra_decorator,
    torch_version,
)
from torch_to_nnef_llm._optional_types import (
    InjectedHuggingFaceHubModule,
    InjectedPeftModule,
    InjectedTransformersModule,
    InjectedTransformersUtilsModule,
    TransformersModule,
)
from torch_to_nnef_llm.config import (
    DtypeStr,
    ExportDirStruct,
    HFConfigHelper,
)
from torch_to_nnef_llm.loader import load_model, load_tokenizer
from torch_to_nnef_llm.models.base import (
    BaseCausal,
    update_forward_signature,
    use_dtype_dyn_cache,
)

LOGGER = logging.getLogger(__name__)

TYPE_OPTIONAL_DEVICE_MAP = T.Optional[
    T.Union[
        str,
        T.Dict[str, T.Union[int, str, torch.device]],
        int,
        torch.device,
    ]
]

LM_VAR_SCHEME = VariableNamingScheme.NATURAL_VERBOSE_CAMEL
LM_CHECK_TOLERANCE = TractCheckTolerance.APPROXIMATE

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
""".strip().replace("\n", " ")

HALF_TYPES = [torch.float16, torch.bfloat16]


def is_forced_half_precision_model(
    force_inputs_dtype: T.Optional[DtypeStr],
    force_module_dtype: T.Optional[DtypeStr],
) -> bool:
    return (
        force_inputs_dtype is not None
        and DtypeStr(force_inputs_dtype).torch_dtype in HALF_TYPES
    ) or (
        force_module_dtype is not None
        and DtypeStr(force_module_dtype).torch_dtype in HALF_TYPES
    )


def _normalize_dump_kwargs(kwargs: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    dump_kwargs = dict(kwargs)
    if isinstance(dump_kwargs.get("tract_check_io_tolerance"), str):
        dump_kwargs["tract_check_io_tolerance"] = TractCheckTolerance(
            dump_kwargs["tract_check_io_tolerance"]
        )
    return dump_kwargs


#: Default number of retries for transient Hugging Face download failures.
DEFAULT_HF_DOWNLOAD_N_RETRIES = 5

#: HTTP statuses worth retrying: rate limit + transient server errors. Auth or
#: missing (401/403/404) never recover on retry and are left to fail.
_TRANSIENT_HF_HTTP = frozenset({429, 500, 502, 503, 504})


def _hf_http_status(exc: BaseException) -> T.Optional[int]:
    """Find an HTTP status anywhere in the exception's cause chain.

    A rate-limited Hugging Face fetch surfaces here as an OSError whose cause
    chain holds the real HfHubHTTPError, so the status we must branch on (e.g.
    429) is not on the outermost exception.
    """
    seen: T.Set[int] = set()
    cur: T.Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(getattr(cur, "response", None), "status_code", None)
        if status is not None:
            return status
        cur = cur.__cause__ or cur.__context__
    return None


def _load_exporter_from(
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    force_module_dtype: T.Optional[DtypeStr] = None,
    force_inputs_dtype: T.Optional[DtypeStr] = None,
    num_logits_to_keep: T.Union[int, str] = 1,
    merge_peft: T.Optional[bool] = None,
    device_map: TYPE_OPTIONAL_DEVICE_MAP = None,
    trust_remote_code: bool = True,
    upcast_quant: T.Optional[T.Sequence[str]] = None,
):
    if (
        is_forced_half_precision_model(force_inputs_dtype, force_module_dtype)
        and torch_version() < "2.0.0"
    ):
        LOGGER.warning(
            "float16 with CPU backend is limited in PyTorch 1.X "
            "(if issues, try to use torch>2.0)"
        )
    local_dir = Path(local_dir) if local_dir else None
    assert hf_model_slug is not None or local_dir is not None
    hf_model_causal = load_model(
        hf_model_slug,
        local_dir,
        force_module_dtype=force_module_dtype,
        merge_peft=merge_peft,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        upcast_quant=upcast_quant,
    )
    tokenizer = load_tokenizer(
        hf_model_causal.config,
        hf_model_slug=hf_model_slug,
        local_dir=local_dir,
        trust_remote_code=trust_remote_code,
    )

    return LLMExporter(
        hf_model_causal,
        tokenizer,
        force_module_dtype=force_module_dtype,
        force_inputs_dtype=force_inputs_dtype,
        num_logits_to_keep=num_logits_to_keep,
        local_dir=local_dir,
    )


class LLMExporter:
    def __init__(
        self,
        hf_model_causal: nn.Module,
        tokenizer: TransformersModule.AutoTokenizer,
        local_dir: T.Optional[Path] = None,
        force_module_dtype: T.Optional[DtypeStr] = None,
        force_inputs_dtype: T.Optional[DtypeStr] = None,
        num_logits_to_keep: T.Union[int, str] = 1,
    ):
        """Init LLMExporter.

        Args:
            hf_model_causal:
                Any Causal model from `transformers` library
            tokenizer:
                Any tokenizer from `transformers` library
            local_dir:
                If set this is the local directory from where model was loaded.
            force_module_dtype:
                Force PyTorch dtype in parameters.
            force_inputs_dtype:
                Force PyTorch dtype in inputs of the models.
            num_logits_to_keep: number of token logits to keep. An int slices
                the last N positions at trace time (0 keeps all). The string
                "dynamic" instead exposes `logits_to_keep` as a runtime scalar
                input, so one export serves cheap prefill (pass 1) and
                speculative decode (pass k+1) without re-exporting.

        """
        self.hf_model_causal = hf_model_causal
        self.tokenizer = tokenizer
        self.local_dir = local_dir

        if hasattr(self.hf_model_causal.config, "torchscript"):
            LOGGER.debug(
                "change to config.torchscript=False and tie_weights again"
            )
            # avoid clone weight instead assign same parameters
            # to avoid duplicates
            self.hf_model_causal.config.torchscript = False
            # only effective if config set tie_word_embeddings=True
            # tie_encoder_decoder=True
        self.hf_model_causal.tie_weights()

        self.model_infos = HFConfigHelper(self.hf_model_causal.config)
        if self.model_infos.conf.model_type == "qwen3_vl":
            # Qwen3-VL currently hits SDPA masking issues during torch.jit
            # tracing. Force eager attention in export mode to keep the graph
            # traceable.
            self.hf_model_causal.config._attn_implementation = "eager"
            if hasattr(self.hf_model_causal, "model") and hasattr(
                self.hf_model_causal.model, "language_model"
            ):
                lang_config = (
                    self.hf_model_causal.model.language_model.config
                )
                lang_config._attn_implementation = "eager"
        self.wrapped_model = BaseCausal(
            self.hf_model_causal,
            handler=self.model_infos.handler,
            num_kv_tensors=self.model_infos.get_num_transformer_layers() * 2,
            with_dyn_cache=self.model_infos.handler.with_dyn_cache,
            num_logits_to_keep=num_logits_to_keep,
        )
        force_module_dtype = (
            DtypeStr(force_module_dtype) if force_module_dtype else None
        )
        force_inputs_dtype = (
            DtypeStr(force_inputs_dtype) if force_inputs_dtype else None
        )
        self.force_module_dtype = force_module_dtype
        if (
            force_module_dtype
            and force_inputs_dtype is None
            and force_module_dtype.torch_dtype in HALF_TYPES
        ):
            LOGGER.info(
                "request inputs aligned dtype: '%s'", force_module_dtype
            )
            force_inputs_dtype = DtypeStr.FLOAT16
        self.force_inputs_dtype = force_inputs_dtype

    @property
    def is_forced_half_precision_model(self) -> bool:
        return is_forced_half_precision_model(
            self.force_module_dtype, self.force_inputs_dtype
        )

    @property
    def main_weight_dtype(self) -> torch.dtype:
        ct: Counter = Counter()
        for p in self.wrapped_model.parameters():
            ct[p.dtype] += p.numel()
        return ct.most_common()[0][0]

    @property
    def is_mainly_weight_half_precision(self) -> bool:
        return self.main_weight_dtype in HALF_TYPES

    @property
    def inputs_dtype(self) -> torch.dtype:
        if self.force_inputs_dtype is None:
            if self.is_mainly_weight_half_precision:
                return torch.float16
            return torch.float32
        return self.force_inputs_dtype.torch_dtype

    @property
    def is_half_precision_model(self) -> bool:
        return (
            self.is_forced_half_precision_model
            or self.is_mainly_weight_half_precision
        )

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
        return sum(_.numel() for _ in self.hf_model_causal.parameters())

    def _build_model_input_spec(
        self,
        n_input_tokens: int = 1,
        n_past_input_tokens: int = 2,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
    ):
        input_spec = self.model_infos.handler.build_input_spec(
            tokenizer=self.tokenizer,
            config_helper=self.model_infos,
            inputs_dtype=self.inputs_dtype,
            sample_text=EN_SAMPLE_TEXT,
            n_input_tokens=n_input_tokens,
            n_past_input_tokens=n_past_input_tokens,
            real_kv_cache=real_kv_cache,
        )
        if getattr(self.wrapped_model, "dynamic_logits_to_keep", False):
            # extra trailing scalar input feeding the all-position logits
            # gather; example value 1 mirrors a regular single-token decode.
            input_spec.inputs = (
                *input_spec.inputs,
                torch.tensor(1, dtype=torch.int64),
            )
            input_spec.input_names = [*input_spec.input_names, "logits_to_keep"]
        return input_spec

    def _build_state_context(
        self,
        inputs: T.Tuple[torch.Tensor, ...],
    ):
        return self.model_infos.handler.build_forward_inputs(
            inputs=inputs,
            wrapper=self.wrapped_model,
        )

    def _export_layout_dirs(
        self,
        export_dirpath: Path,
        export_dir_struct: ExportDirStruct,
    ) -> T.Tuple[Path, Path]:
        if export_dir_struct == ExportDirStruct.DEEP:
            model_dir = export_dirpath / "model"
            tok_dir = export_dirpath / "tokenizer"
            model_dir.mkdir(parents=True, exist_ok=True)
            tok_dir.mkdir(parents=True, exist_ok=True)
            return model_dir, tok_dir
        if export_dir_struct == ExportDirStruct.FLAT:
            export_dirpath.mkdir(parents=True, exist_ok=True)
            return export_dirpath, export_dirpath
        raise T2NErrorNotImplemented()

    def _dump_modes_json(
        self,
        export_dirpath: Path,
        test_dir: Path,
        sample_generation_total_size: int,
    ) -> None:
        if sample_generation_total_size <= 0:
            LOGGER.info("'inference mode' evaluation skipped")
            return

        LOGGER.info(
            "'inference mode' evaluation started with "
            "sample_generation_total_size=%d",
            sample_generation_total_size,
        )
        pairs = self.dump_all_io_npz_kind(
            test_dir, size=sample_generation_total_size
        )
        modes = []
        for in_p, _ in pairs:
            base = in_p.stem
            for suff in ("_inputs", "_outputs", "_io"):
                if base.endswith(suff):
                    base = base[: -len(suff)]
                    break
            modes.append(base)
        with (export_dirpath / "modes.json").open("w", encoding="utf8") as fh:
            json.dump({"pytorch_supported_modes": modes}, fh)
        LOGGER.info("'inference mode' evaluation data generated")

    @staticmethod
    @require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="huggingface_hub")
    def load(
        model_slug: T.Optional[str] = None,
        local_dir: T.Optional[Path] = None,
        *,
        hf_download_n_retries: int = DEFAULT_HF_DOWNLOAD_N_RETRIES,
        huggingface_hub: InjectedHuggingFaceHubModule = INJECTED,
        **kwargs,
    ):
        """Load from either huggingface model slug hub or local_dir.

        ``hf_download_n_retries`` bounds how many times a *transient* Hugging
        Face failure (HTTP 429 rate limit or a 5xx) is retried with exponential
        backoff before giving up; set 0 to disable. Auth/missing errors are not
        retried, and a gated repo still triggers an interactive login.
        """
        with torch.no_grad():
            exporter_from_kwargs: T.Dict[str, T.Any] = {
                "hf_model_slug": model_slug,
                "local_dir": local_dir,
                **kwargs,
            }
            attempt = 0
            while True:
                try:
                    return _load_exporter_from(**exporter_from_kwargs)
                except OSError as exp:
                    msg = exp.args[0] if exp.args else ""
                    if isinstance(msg, str) and "gated repo" in msg:
                        print(msg)
                        huggingface_hub.login()
                        return _load_exporter_from(**exporter_from_kwargs)
                    status = _hf_http_status(exp)
                    if (
                        status in _TRANSIENT_HF_HTTP
                        and attempt < hf_download_n_retries
                    ):
                        attempt += 1
                        delay = min(2.0 * 2 ** (attempt - 1), 60.0)
                        LOGGER.warning(
                            "Hugging Face download failed with HTTP %s; "
                            "retry %d/%d in %.0fs",
                            status,
                            attempt,
                            hf_download_n_retries,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    raise T2NErrorRuntime(
                        "OSError while loading model"
                    ) from exp

    def check_wrapper_io(self):
        """Check the wrapper gives same outputs compared to vanilla model."""
        (
            inputs,
            _,
            out_cache_names,
            _,
        ) = self.generate_inputs_io_names_and_dynaxes()
        wrapped_outs = self.wrapped_model(*inputs)
        hf_inputs = inputs
        if getattr(self.wrapped_model, "dynamic_logits_to_keep", False):
            # drop the trailing logits_to_keep scalar; the HF model builds its
            # kv cache from input_ids + caches only.
            hf_inputs = inputs[:-1]
        state_context = self._build_state_context(hf_inputs)
        outs = self.model_infos.handler.call_model(
            model=self.hf_model_causal,
            state_context=state_context,
            wrapper=self.wrapped_model,
        )

        def err_check(output_name: str, ref: torch.Tensor, cand: torch.Tensor):
            ref = ref.float()
            cand = cand.float()
            if not torch.allclose(
                ref,
                cand,
                atol=1e-3 if self.is_half_precision_model else 1e-4,
            ):
                msg = (
                    f"Model: {self.hf_model_causal.__class__} wrapped "
                    f"with: {self.wrapped_model.__class__}, "
                    "give inconsistent results compared to "
                    f"vanilla in '{output_name}': "
                    f"avg diff: {(ref - cand).abs().mean():0.4f}. "
                    "Likely need a torch_to_nnef fix."
                )
                LOGGER.error(msg)
                raise T2NErrorConsistency(msg)

        if isinstance(self.wrapped_model, torch.fx.GraphModule):
            LOGGER.info(
                "skip checks wrapped_model vs hf_model_causal "
                "since use of GraphModule "
                "(which copied graph and could have been quantized in meantime)"
            )
        else:
            expected_outputs = self.model_infos.handler.build_forward_outputs(
                model=self.hf_model_causal,
                model_outputs=outs,
                state_context=state_context,
                num_logits_to_keep=self.wrapped_model.num_logits_to_keep,
            )
            if getattr(self.wrapped_model, "dynamic_logits_to_keep", False):
                # wrapped output gathered the last rows; HF emitted all of
                # them, so compare against the matching tail.
                kept = wrapped_outs[0].shape[1]
                expected_outputs[0] = expected_outputs[0][:, -kept:, :]
            for output_name, ref, cand in zip(
                out_cache_names,
                expected_outputs[1:],
                wrapped_outs[1:],
                strict=False,
            ):
                err_check(output_name, ref, cand)
            err_check("logits", expected_outputs[0], wrapped_outs[0])
            LOGGER.info(
                "In PyTorch wrapped_model:%s provide same results as %s",
                self.wrapped_model.__class__,
                self.hf_model_causal.__class__,
            )

    def generate_inputs_io_names_and_dynaxes(
        self,
        n_input_tokens: int = 1,
        n_past_input_tokens: int = 2,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
    ):
        input_spec = self._build_model_input_spec(
            n_input_tokens=n_input_tokens,
            n_past_input_tokens=n_past_input_tokens,
            real_kv_cache=real_kv_cache,
        )
        update_forward_signature(self.wrapped_model, input_spec)
        inputs = input_spec.inputs
        input_names = input_spec.input_names
        output_names = input_spec.output_names
        dynamic_axes = input_spec.dynamic_axes
        # dynamic logits_to_keep adds one extra input with no matching output
        n_extra_inputs = (
            1
            if getattr(self.wrapped_model, "dynamic_logits_to_keep", False)
            else 0
        )
        assert len(inputs) == len(input_names), (
            f"{len(inputs)} == {len(input_names)}"
        )
        assert len(inputs) - n_extra_inputs == len(output_names), (
            f"{len(inputs)} - {n_extra_inputs} == {len(output_names)}"
        )
        return (
            inputs,
            input_names,
            output_names,
            dynamic_axes,
        )

    def build_io_npz(
        self,
        io_npz_path: Path,
        *args,
        inputs_npz_path: T.Optional[Path] = None,
        outputs_npz_path: T.Optional[Path] = None,
        **kwargs,
    ):
        (
            inputs,
            input_names,
            output_names,
            _,
        ) = self.generate_inputs_io_names_and_dynaxes(*args, **kwargs)
        build_io(
            self.wrapped_model,
            inputs,
            input_bundle_path=inputs_npz_path,
            output_bundle_path=outputs_npz_path,
            input_names=input_names,
            output_names=output_names,
        )

    def dump_all_io_npz_kind(
        self, io_npz_dirpath: Path, size: int = 6
    ) -> T.List[Path]:
        """Realistic dump of IO's."""
        half = size // 2
        prompt_in_npz = io_npz_dirpath / "prompt_inputs.npz"
        prompt_out_npz = io_npz_dirpath / "prompt_outputs.npz"
        self.build_io_npz(
            io_npz_path=None,
            n_input_tokens=size,
            n_past_input_tokens=0,
            inputs_npz_path=prompt_in_npz,
            outputs_npz_path=prompt_out_npz,
        )
        res = {**np.load(prompt_in_npz), **np.load(prompt_out_npz)}
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
        prompt_with_past_in_npz = io_npz_dirpath / "prompt_with_past_inputs.npz"
        prompt_with_past_out_npz = (
            io_npz_dirpath / "prompt_with_past_outputs.npz"
        )
        try:
            self.build_io_npz(
                io_npz_path=None,
                n_input_tokens=half,
                n_past_input_tokens=half,
                real_kv_cache=real_kv_cache,
                inputs_npz_path=prompt_with_past_in_npz,
                outputs_npz_path=prompt_with_past_out_npz,
            )
        except (RuntimeError, ValueError, OSError) as exp:
            LOGGER.error(
                "Prompt with past, does not run in PyTorch "
                "(likely modeling limit): %s",
                exp,
            )
        text_gen_in_npz = io_npz_dirpath / "text_generation_inputs.npz"
        text_gen_out_npz = io_npz_dirpath / "text_generation_outputs.npz"
        self.build_io_npz(
            io_npz_path=None,
            n_input_tokens=1,
            n_past_input_tokens=size - 1,
            real_kv_cache=real_kv_cache,
            inputs_npz_path=text_gen_in_npz,
            outputs_npz_path=text_gen_out_npz,
        )
        return [
            (prompt_in_npz, prompt_out_npz),
            (prompt_with_past_in_npz, prompt_with_past_out_npz),
            (text_gen_in_npz, text_gen_out_npz),
        ]

    @require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
    def generate_test_text(
        self,
        prompt: str = "Alan Turing was",
        *,
        transformers: InjectedTransformersModule = INJECTED,
    ):
        LOGGER.info("start to generate testing text from loaded model:")
        generation_config = transformers.GenerationConfig(
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
        LOGGER.info("generated text: %s", text)

    def apply_half_precision_fixes(self):
        """Align float dtype arguments in few graph ops.

        Indeed all LLM are trained using GPU/TPU/CPU kernels
        related PyTorch backend support f16 dtype in some operators
        contrary to PyTorch CPU inference (@ 2024-09-09).

        To solve this issue we monkey patch in this cli few functional API.
        """
        if not isinstance(
            torch.nn.functional.layer_norm, StateLessF32LayerNorm
        ):
            torch.nn.functional.original_layer_norm = (
                torch.nn.functional.layer_norm
            )
            torch.nn.functional.layer_norm = StateLessF32LayerNorm()

    def reset_torch_fns(self):
        """Cleanup any torch behavior alterations."""
        if isinstance(torch.nn.functional.layer_norm, StateLessF32LayerNorm):
            torch.nn.functional.layer_norm = (
                torch.nn.functional.original_layer_norm
            )
            del torch.nn.functional.original_layer_norm

    @use_dtype_dyn_cache
    def prepare(  # pylint: disable=too-many-positional-arguments
        self,
        compression_method: T.Optional[str] = None,
        compression_registry: str = DEFAULT_COMPRESSION_REGISTRY,
        test_display_token_gens: bool = False,
        wrapper_io_check: bool = True,
        export_dirpath: T.Optional[Path] = None,
        log_level: int = logging.INFO,
    ):
        """Prepare model to export (f16/compression/checks...)."""
        logging.getLogger().setLevel(log_level)
        with torch.no_grad():
            if test_display_token_gens:
                self.generate_test_text()

        # compression method may sometime need
        # gradient optimization so avoid context manager no_grad
        if compression_method:
            LOGGER.info("start compresssion: %s", compression_method)
            registry = dynamic_load_registry(compression_registry)
            self.wrapped_model = registry[compression_method](
                self.wrapped_model,
                tokenizer=self.tokenizer,
                # may be usefull to dump compression evaluations results
                export_dirpath=export_dirpath,
                # may be usefull to perform internal evaluations
                # when more data than just llm torch is available
                local_dir=self.local_dir,
            )
            LOGGER.info(
                "successfully applied compression: %s", compression_method
            )

        with torch.no_grad():
            if test_display_token_gens and (
                compression_method or self.is_half_precision_model
            ):
                LOGGER.info(
                    "check testing text post compression/f16 conversion:"
                )
                self.generate_test_text()
            if wrapper_io_check:
                self.check_wrapper_io()

    @use_dtype_dyn_cache
    @require_extra_decorator(
        extra=T2NExtra.LLM_TRACT,
        module="transformers.utils",
        kw="transformers_utils",
    )
    def export_model(
        self,
        export_dirpath: Path,
        inference_target: TractNNEF,
        naming_scheme: VariableNamingScheme = LM_VAR_SCHEME,
        log_level=logging.INFO,
        dump_with_tokenizer_and_conf: bool = False,
        check_inference_modes: bool = True,
        sample_generation_total_size: int = 0,
        ignore_already_exist_dir: bool = False,
        export_dir_struct: ExportDirStruct = ExportDirStruct.DEEP,
        debug_bundle_path: T.Optional[Path] = None,
        *,
        transformers_utils: InjectedTransformersUtilsModule = INJECTED,
    ):
        """Export model has is currently in self.hf_model_causal.

        and dump some npz tests to check io latter-on
        """
        with torch.no_grad():
            if not ignore_already_exist_dir:
                assert not export_dirpath.exists(), export_dirpath
            assert sample_generation_total_size >= 2
            (
                inputs,
                input_names,
                output_names,
                dynamic_axes,
            ) = self.generate_inputs_io_names_and_dynaxes()

            LOGGER.info("start export with 'torch_to_nnef'")
            assert hasattr(inference_target, "dynamic_axes")
            inference_target.dynamic_axes = dynamic_axes

            # Add io.npz test in exproted dir for dbg purpose
            test_dir = export_dirpath / "tests"
            test_dir.mkdir(parents=True)

            if check_inference_modes:
                self._dump_modes_json(
                    export_dirpath, test_dir, sample_generation_total_size
                )
            else:
                LOGGER.info("'inference mode' evaluation skipped")

            model_dir, tok_dir = self._export_layout_dirs(
                export_dirpath, export_dir_struct
            )

            if dump_with_tokenizer_and_conf:
                # export_dir_struct
                self.hf_model_causal.config.to_json_file(
                    model_dir / transformers_utils.CONFIG_NAME, use_diff=False
                )
                self.tokenizer.save_pretrained(tok_dir)

            if self.is_half_precision_model:
                self.apply_half_precision_fixes()

            # Produce separate input/output bundles for Tract checks
            export_inputs_npz = test_dir / "export_inputs.npz"
            export_outputs_npz = test_dir / "export_outputs.npz"
            build_io(
                self.wrapped_model,
                inputs,
                input_bundle_path=export_inputs_npz,
                output_bundle_path=export_outputs_npz,
                input_names=input_names,
                output_names=output_names,
            )
            # No merged NPZ: keep inputs/outputs separate
            export_model_to_nnef(
                model=self.wrapped_model,
                args=inputs,
                inference_target=inference_target,
                file_path_export=model_dir / "model.nnef.tgz",
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
                debug_bundle_path=debug_bundle_path,
            )
            self.reset_torch_fns()

    def dump(self, **kwargs):
        """Prepare and export model to NNEF."""
        inference_target = self.build_inference_target(
            **{
                key: kwargs.pop(key)
                for key in [
                    "tract_specific_path",
                    "tract_specific_version",
                    "tract_specific_properties",
                    "no_verify",
                    "force_f32_attention",
                    "force_f32_linear_accumulator",
                    "force_f32_normalization",
                    "reify_sdpa_operator",
                    "tract_check_io_tolerance",
                ]
                if key in kwargs
            },
            compression_method=kwargs.get("compression_method"),
            compression_registry=kwargs.get("compression_registry"),
        )
        return self.dump_with_inference_target(
            inference_target=inference_target, **kwargs
        )

    def build_inference_target(
        self,
        tract_specific_path: T.Optional[Path] = None,
        tract_specific_version: T.Optional[str] = None,
        tract_specific_properties: T.Optional[T.Dict[str, str]] = None,
        no_verify: bool = False,
        force_f32_attention: T.Optional[bool] = None,
        force_f32_linear_accumulator: T.Optional[bool] = None,
        force_f32_normalization: T.Optional[bool] = None,
        reify_sdpa_operator: T.Optional[bool] = None,
        tract_check_io_tolerance: TractCheckTolerance = LM_CHECK_TOLERANCE,
        compression_method: T.Optional[str] = None,
        compression_registry: T.Optional[str] = None,
    ) -> TractNNEF:
        assert (  # mutualy exclusive arguments
            (tract_specific_path is None and tract_specific_version is None)
            or tract_specific_path is None
            or tract_specific_version is None
        )
        if tract_specific_version:
            assert tract_specific_path is None, "set either version or path"
            inference_target = TractNNEF(
                SemanticVersion.from_str(tract_specific_version)
                if isinstance(tract_specific_version, str)
                else tract_specific_version
            )
        elif tract_specific_path:
            tract_cli_path = Path(tract_specific_path)
            assert tract_cli_path.exists(), tract_cli_path
            tract_cli = TractCli(tract_cli_path)
            inference_target = TractNNEF(
                tract_cli.version,
                specific_tract_binary_path=tract_cli_path,
            )
        else:
            inference_target = TractNNEF.latest()
        inference_target.specific_properties = (
            self._get_tract_properties_from_prep(
                tract_specific_properties,
                compression_registry,
                compression_method,
            )
        )
        inference_target.check_io_tolerance = tract_check_io_tolerance

        if force_f32_attention is not None:
            inference_target.force_attention_inner_in_f32 = force_f32_attention
        if force_f32_linear_accumulator is not None:
            inference_target.force_linear_accumulation_in_f32 = (
                force_f32_linear_accumulator
            )
        if force_f32_normalization is not None:
            inference_target.force_norm_in_f32 = force_f32_normalization

        if reify_sdpa_operator is not None:
            inference_target.reify_sdpa_operator = reify_sdpa_operator

        if (
            self.is_half_precision_model
            and self.model_infos.conf.model_type == "qwen2"
        ):
            inference_target.force_attention_inner_in_f32 = True
            inference_target.force_linear_accumulation_in_f32 = True

        if no_verify:
            LOGGER.info(
                "tract inference is not checked because 'no_verify=True'"
            )
        inference_target.check_io = not no_verify
        return inference_target

    def _get_tract_properties_from_prep(
        self,
        tract_specific_properties,
        compression_registry,
        compression_method,
    ) -> T.Dict[str, str]:
        tract_specific_properties = tract_specific_properties or {}
        tract_specific_properties.update(
            {
                "hf_model_type": self.model_infos.conf.model_type,
                "n_parameters": str(self.model_n_params),
                "main_base_weight_dtype": DtypeStr.from_torch_dtype(
                    self.main_weight_dtype
                ).value,
                "forced_module_dtype": self.force_module_dtype.value
                if self.force_module_dtype
                else "",
                "as_float16": "1"
                if self.main_weight_dtype == torch.float16
                else "0",
                "inputs_dtype": DtypeStr.from_torch_dtype(
                    self.inputs_dtype
                ).value,
            }
        )
        if compression_method is not None:
            cprops = {
                "compression_register_key": compression_method,
                "compression_registry": compression_registry,
            }
            if "q4" in compression_method:
                cprops["compression_method"] = "min_max_q4_0_with_embeddings"
            tract_specific_properties.update(cprops)
        if not self.hf_model_causal.config._name_or_path.startswith("/tmp"):
            tract_specific_properties["name_or_path"] = (
                self.hf_model_causal.config._name_or_path
            )
        if hasattr(self.hf_model_causal, "peft_config"):
            # PEFT model need peft dependency to be in env
            tract_specific_properties["peft_merged"] = (
                "0" if self.is_peft_model(self.hf_model_causal) else "1"
            )
            for k, _conf in self.hf_model_causal.peft_config.items():
                tract_specific_properties[f"peft_{k}_type"] = (
                    self.hf_model_causal.peft_config[k].peft_type.value
                )
                tract_specific_properties[f"peft_{k}_target_modules"] = (
                    ",".join(self.hf_model_causal.peft_config[k].target_modules)
                )
        return tract_specific_properties

    @staticmethod
    @require_extra_decorator(extra=T2NExtra.PEFT, module="peft")
    def is_peft_model(model, *, peft: InjectedPeftModule = INJECTED) -> bool:
        """Check if the model is a PEFT model."""
        return isinstance(model, peft.PeftModel)

    def dump_with_inference_target(
        self,
        inference_target: TractNNEF,
        export_dirpath: T.Union[str, Path],
        compression_method: T.Optional[str] = None,
        compression_registry: str = DEFAULT_COMPRESSION_REGISTRY,
        test_display_token_gens: bool = False,
        naming_scheme: VariableNamingScheme = LM_VAR_SCHEME,
        dump_with_tokenizer_and_conf: bool = False,
        check_inference_modes: bool = True,
        wrapper_io_check: bool = True,
        log_level: int = logging.INFO,
        sample_generation_total_size: int = 6,
        no_verify: bool = False,
        ignore_already_exist_dir: bool = False,
        export_dir_struct: ExportDirStruct = ExportDirStruct.DEEP,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        export_dirpath = Path(export_dirpath)
        if no_verify and wrapper_io_check:
            LOGGER.info(
                "force disable 'wrapper_io_check' because 'no_verify=True'"
            )
            wrapper_io_check = False
        if no_verify and test_display_token_gens:
            LOGGER.info(
                "force disable 'test_display_token_gens' "
                "because 'no_verify=True'"
            )
            test_display_token_gens = False
        if export_dirpath.exists() and not ignore_already_exist_dir:
            raise T2NErrorMisuse(
                "'export_dirpath' should not exist but found: "
                f"'{export_dirpath}'"
            )

        self.prepare(
            compression_method=compression_method,
            compression_registry=compression_registry,
            test_display_token_gens=test_display_token_gens,
            wrapper_io_check=wrapper_io_check,
            export_dirpath=export_dirpath,
            log_level=log_level,
        )
        self.export_model(
            export_dirpath,
            naming_scheme=naming_scheme,
            inference_target=inference_target,
            log_level=log_level,
            dump_with_tokenizer_and_conf=dump_with_tokenizer_and_conf,
            check_inference_modes=check_inference_modes,
            sample_generation_total_size=sample_generation_total_size,
            ignore_already_exist_dir=ignore_already_exist_dir,
            export_dir_struct=export_dir_struct,
            debug_bundle_path=debug_bundle_path,
        )


class StateLessF32LayerNorm(nn.Module):
    def forward(  # pylint: disable=too-many-positional-arguments
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
    force_module_dtype: T.Optional[DtypeStr] = None,
    force_inputs_dtype: T.Optional[DtypeStr] = None,
    merge_peft: T.Optional[bool] = None,
    num_logits_to_keep: T.Union[int, str] = 1,
    device_map: TYPE_OPTIONAL_DEVICE_MAP = None,
    hf_download_n_retries: int = DEFAULT_HF_DOWNLOAD_N_RETRIES,
    trust_remote_code: bool = True,
    upcast_quant: T.Optional[T.Sequence[str]] = None,
    **kwargs,
) -> T.Tuple[T.Union[Path, None], LLMExporter]:
    """Util to export LLM model."""
    exporter = LLMExporter.load(
        model_slug,
        local_dir,
        force_module_dtype=force_module_dtype,
        force_inputs_dtype=force_inputs_dtype,
        merge_peft=merge_peft,
        num_logits_to_keep=num_logits_to_keep,
        device_map=device_map,
        hf_download_n_retries=hf_download_n_retries,
        trust_remote_code=trust_remote_code,
        upcast_quant=upcast_quant,
    )
    dump_kwargs = _normalize_dump_kwargs(kwargs)
    exporter.dump(**dump_kwargs)
    export_path = dump_kwargs.get("export_dirpath")
    return (
        Path(export_path) if export_path else None,
        exporter,
    )
