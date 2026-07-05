import json
import logging
import os
import tempfile
import typing as T
from pathlib import Path

import torch

from torch_to_nnef.exceptions import (
    T2NErrorMisuse,
    T2NErrorNotFoundFile,
    T2NErrorNotImplemented,
)
from torch_to_nnef.tensor.offload import (
    AUTO_DEVICE_MAP_KEY,
    ON_DISK_DEVICE_MAP_KEY,
    t2n_load_checkpoint_and_dispatch,
)
from torch_to_nnef.utils import (
    INJECTED,
    SemanticVersion,
    T2NExtra,
    init_empty_weights,
    require_extra_decorator,
)
from torch_to_nnef_llm._optional_types import (
    InjectedHuggingFaceHubModule,
    InjectedPeftModule,
    InjectedTransformersModule,
)
from torch_to_nnef_llm.config import (
    CUSTOM_CONFIGS,
    REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG,
    DtypeStr,
)
from torch_to_nnef_llm.models import handlers

LOGGER = logging.getLogger(__name__)

TYPE_OPTIONAL_DEVICE_MAP = T.Optional[
    T.Union[
        str,
        T.Dict[str, T.Union[int, str, torch.device]],
        int,
        torch.device,
    ]
]


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
def resolve_auto_model_class(
    model_type: str,
    *,
    transformers: InjectedTransformersModule = INJECTED,
):
    """Resolve the HF model class from the registered architecture handler."""
    handler_class = handlers.get_handler(model_type)
    return handler_class.get_auto_model_class(transformers)


# sentinel accepted by `upcast_quant` to dequantize whatever native format the
# model ships, without naming it.
UPCAST_ANY = "any"


def _as_quant_config(raw):
    """Normalize a model/config's ``quantization_config`` to a config *object*.

    ``AutoConfig`` may surface it as a plain dict; parse that into the proper
    ``*Config`` subclass (which carries ``quant_method`` and, for load-time
    dequant formats, a ``dequantize`` attribute). Returns ``None`` if absent.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        # pylint: disable-next=import-outside-toplevel
        from transformers.quantizers.auto import AutoQuantizationConfig

        return AutoQuantizationConfig.from_dict(raw)
    return raw


def _quant_method_of(quant_config) -> T.Optional[str]:
    """Lowercased ``quant_method`` of a quant config object, or ``None``."""
    if quant_config is None:
        return None
    qm = getattr(quant_config, "quant_method", None)
    if qm is None:
        return None
    return str(getattr(qm, "value", qm)).lower()


def _native_quant_method(model) -> T.Optional[str]:
    """Native quant method of a model, or ``None`` if (no longer) quantized."""
    raw = getattr(getattr(model, "config", None), "quantization_config", None)
    return _quant_method_of(_as_quant_config(raw))


def should_upcast(
    quant_method: T.Optional[str],
    requested: T.Optional[T.Sequence[str]],
) -> bool:
    """Whether a ``quant_method`` model should be dequantized for ``requested``.

    Pure decision (no model needed) so it is unit-testable. ``requested`` is the
    opt-in list of methods to up-cast; the ``"any"`` sentinel matches all.
    """
    if quant_method is None or not requested:
        return False
    wanted = {str(r).lower() for r in requested}
    return UPCAST_ANY in wanted or quant_method in wanted


def _normalize_upcast_request(
    requested: T.Optional[T.Sequence[str]],
) -> T.Optional[T.List[str]]:
    """Validate + normalize the requested up-cast methods, up-front.

    Each entry must be the ``"any"`` sentinel or a method transformers knows
    (validated through transformers' own ``QuantizationMethod`` enum, the single
    source of truth, so it stays in sync across versions). Returns the
    lowercased canonical values, or ``None`` when nothing was requested. Raises
    ``T2NErrorMisuse`` on an unknown method or a too-old transformers, before
    any download or model load.
    """
    if not requested:
        return None
    # a bare string is a common caller mistake; treat it as a single method
    # rather than iterating it per-character.
    if isinstance(requested, str):
        requested = [requested]
    # lazy import: transformers is an optional extra of this package
    # pylint: disable-next=import-outside-toplevel
    import transformers

    # up-cast relies on the quantizer dequantization API
    # (`AutoQuantizationConfig` / `model.dequantize()`), introduced with the
    # HfQuantizer refactor in transformers 4.38.0. Older versions lack
    # `transformers.quantizers`, so fail here with a clear message instead of a
    # cryptic ModuleNotFoundError later.
    if SemanticVersion.from_str(transformers.__version__) < "4.38.0":
        raise T2NErrorMisuse(
            "upcast_quant requires transformers >= 4.38.0 (quantizer "
            f"dequantization API); installed: {transformers.__version__}. "
            "Upgrade transformers, or drop upcast_quant."
        )
    # pylint: disable-next=import-outside-toplevel
    from transformers.utils.quantization_config import QuantizationMethod

    normalized: T.List[str] = []
    for raw in requested:
        value = str(raw).lower()
        if value == UPCAST_ANY:
            normalized.append(value)
            continue
        try:
            normalized.append(QuantizationMethod(value).value)
        except ValueError as exc:
            valid = [m.value for m in QuantizationMethod] + [UPCAST_ANY]
            raise T2NErrorMisuse(
                f"unknown upcast_quant method '{value}'; valid methods are "
                f"{valid}"
            ) from exc
    return normalized


# How a given quantizer can be dequantized in transformers:
#   - "load": only via the config's ``dequantize=True`` flag, applied during
#             ``from_pretrained`` (e.g. mxfp4, fp8, metal).
#   - "post": via ``model.dequantize()`` after load (e.g. bnb, higgs).
# A config object exposing a ``dequantize`` attribute is load-time; otherwise we
# attempt the post-load path (and verify the result is actually dense).
def plan_upcast(quant_config, requested: T.Optional[T.Sequence[str]]):
    """Decide *whether and how* to up-cast, before the real load. Pure.

    Returns one of:
      - ``("none", None)``: not quantized, or quantized but not requested
        (caller should warn in the latter case).
      - ``("load", quant_config)``: load-time dequant; ``quant_config`` has had
        ``dequantize=True`` set, to pass to ``from_pretrained``.
      - ``("post", method)``: dequantize via ``model.dequantize()`` after load.

    Raises ``T2NErrorMisuse`` if the model is quantized in a format the caller
    did *not* select (it would still break tract export, so fail loudly).
    """
    method = _quant_method_of(quant_config)
    if method is None:
        return ("none", None)
    if not requested:
        # opt-in; the caller warns that export will likely fail
        return ("none", None)
    if not should_upcast(method, requested):
        raise T2NErrorMisuse(
            f"model is natively quantized as '{method}', which tract cannot "
            f"export, but up-cast was only requested for {list(requested)}. "
            f"Add '{method}' (or 'any') to upcast_quant to dequantize it."
        )
    if hasattr(quant_config, "dequantize"):
        quant_config.dequantize = True
        return ("load", quant_config)
    return ("post", method)


def assert_upcast_dense(model, requested: T.Optional[T.Sequence[str]]) -> None:
    """Fail loudly if an up-casted model is still (even partially) quantized.

    Catches a format that only dequantized some weights here, rather than as an
    opaque tract error downstream. No-op when up-cast was not requested.
    """
    if not requested:
        return
    remaining = _native_quant_method(model)
    has_quantizer = getattr(model, "hf_quantizer", None) is not None
    if remaining is not None or has_quantizer:
        detail = (
            f"native '{remaining}' quantization"
            if remaining is not None
            else "an active quantizer"
        )
        raise T2NErrorMisuse(
            f"up-cast did not fully dequantize the model: it still reports "
            f"{detail}. tract cannot export a partially-quantized model; this "
            "format may need a different dequantization path."
        )


def _peek_quant_config(config_source, trust_remote_code, transformers):
    """Read a model's ``quantization_config`` (as a config object) before load.

    Lets a load-time dequant flag be injected into ``from_pretrained``. Returns
    ``None`` if the source has no quant config or can't be read (the caller then
    proceeds without load-time up-cast).
    """
    if config_source is None:
        return None
    # For a local dir, resolve the same subdir the loader will use (config.json
    # may be nested), so the peek doesn't miss the plan on those layouts.
    if isinstance(config_source, Path):
        try:
            config_source = find_subdir_with_filename_in(
                config_source, "config.json"
            )
        except T2NErrorNotFoundFile:
            return None
    # best-effort: any failure to read/parse the config (incl. a quant_method
    # this transformers version doesn't know, which makes `from_dict` raise)
    # just means "no load-time plan" rather than an error.
    try:
        cfg = transformers.AutoConfig.from_pretrained(
            config_source, trust_remote_code=trust_remote_code
        )
        return _as_quant_config(getattr(cfg, "quantization_config", None))
    except (OSError, ValueError, ImportError, KeyError) as exp:
        LOGGER.warning("could not read config to plan up-cast: %s", exp)
        return None


def _plan_and_inject_upcast(
    config_source, requested, kwargs, trust_remote_code, transformers
):
    """Plan a native-quant up-cast before load; inject the load-time flag.

    For load-time formats (mxfp4/fp8/metal) sets ``quantization_config`` (with
    ``dequantize=True``) in ``kwargs``. Returns the plan for the post-load step.
    """
    if not requested or config_source is None:
        return ("none", None)
    quant_config = _peek_quant_config(
        config_source, trust_remote_code, transformers
    )
    method = _quant_method_of(quant_config)
    if method is None:
        LOGGER.info("up-cast requested but model is not natively quantized")
        return ("none", None)
    plan = plan_upcast(quant_config, requested)
    if plan[0] == "load":
        LOGGER.info("up-casting native '%s' at load time", method)
        kwargs["quantization_config"] = plan[1]
    return plan


def _finish_upcast(model, plan, requested):
    """Complete the up-cast after load and verify the model is dense.

    "load" formats were already dequantized during ``from_pretrained``; "post"
    formats (bnb/higgs) dequantize here. A quantized model for which up-cast was
    not requested is left as-is with a warning (opt-in: never silent).
    """
    if plan[0] == "post":
        LOGGER.info("up-casting native '%s' post-load", plan[1])
        # Only some quantizers (bnb, higgs) implement post-load dequant; others
        # raise NotImplementedError, and transformers raises ValueError if the
        # model turned out not to carry a live quantizer. Surface either as a
        # clear T2NError instead of a cryptic crash from inside transformers.
        try:
            model = model.dequantize()
        except (NotImplementedError, ValueError) as exc:
            raise T2NErrorMisuse(
                f"native '{plan[1]}' quantization cannot be dequantized by "
                "transformers (no dequantize support), so it cannot be "
                "exported to tract. Use a checkpoint in a dequantizable format "
                "(e.g. mxfp4, fp8, bitsandbytes) or an unquantized model."
            ) from exc
    elif not requested and _native_quant_method(model) is not None:
        method = _native_quant_method(model)
        LOGGER.warning(
            "model ships native '%s' quantization but upcast_quant was not "
            "requested; tract export will likely fail. Pass upcast_quant=['%s']"
            " (or ['any']) to dequantize it to float first.",
            method,
            method,
        )
    assert_upcast_dense(model, requested)
    return model


def find_subdir_with_filename_in(dirpath: Path, filename: str) -> Path:
    """Find a subdir with filename in it."""
    found_dirs = {p.parent for p in dirpath.glob(f"**/{filename}")}
    if not 0 < len(found_dirs) < 2:
        raise T2NErrorNotFoundFile(
            f"Found {len(found_dirs)} dirs for with '{filename}' file. "
            f"found_dirs={found_dirs}. "
            + (
                "Unable to decide which one should selected..."
                if len(found_dirs) > 1
                else "Is it a valid model directory ?"
            )
        )
    return found_dirs.pop()


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
def load_tokenizer(
    config,
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    *,
    trust_remote_code: bool = True,
    transformers: InjectedTransformersModule = INJECTED,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_slug = REMAP_MODEL_TYPE_TO_TOKENIZER_SLUG.get(
        config.model_type, hf_model_slug
    )
    if tokenizer_slug is None:
        assert local_dir is not None
    if local_dir is not None:
        local_dir = find_subdir_with_filename_in(local_dir, "tokenizer.json")
    return transformers.AutoTokenizer.from_pretrained(
        local_dir or tokenizer_slug, trust_remote_code=trust_remote_code
    )


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
@require_extra_decorator(extra=T2NExtra.PEFT, module="peft")
def _try_load_peft(
    dir_path,
    kwargs,
    exp,
    *,
    transformers: InjectedTransformersModule = INJECTED,
    peft: InjectedPeftModule = INJECTED,
):
    # likely an embedding issue with added tokens
    with (dir_path / "adapter_config.json").open("r", encoding="utf8") as fh:
        dic = json.load(fh)
    hf_model_causal = transformers.AutoModelForCausalLM.from_pretrained(
        dic["base_model_name_or_path"], **kwargs
    )
    msg = "Error(s) in loading state_dict for"
    if exp.args[0].startswith(msg) and "size mismatch for" in exp.args[0]:
        new_tokenizer_len = int(exp.args[0].split("[")[1].split(",")[0])
        hf_model_causal.resize_token_embeddings(new_tokenizer_len)
        print("new_tokenizer_len:", new_tokenizer_len)

    hf_model_causal = peft.PeftModel.from_pretrained(hf_model_causal, dir_path)
    LOGGER.info("loaded a PEFT model with resized token embeddings")
    return hf_model_causal


def assert_model_safetensors_exists(dir_path):
    assert (
        "model" in p.name and p.name.endswith(".safetensors")
        for p in dir_path.iterdir()
    ), dir_path


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
def load_peft_model(
    local_dir,
    kwargs,
    *,
    transformers: InjectedTransformersModule = INJECTED,
):
    """Load PEFT adapted models."""
    dir_path = find_subdir_with_filename_in(local_dir, "adapter_config.json")
    assert dir_path.is_dir(), dir_path
    assert_model_safetensors_exists(dir_path)

    while True:
        try:
            hf_model_causal = transformers.AutoModelForCausalLM.from_pretrained(
                dir_path, **kwargs
            )
        except ValueError as exp:
            msg = "Should have a `model_type` key in its config.json,"
            if msg in exp.args[0]:
                return _try_load_peft(dir_path, kwargs, exp)
            raise T2NErrorMisuse(msg) from exp
        except RuntimeError as exp:
            msg = "Error(s) in loading state_dict for"
            if (
                exp.args[0].startswith(msg)
                and "size mismatch for" in exp.args[0]
            ):
                return _try_load_peft(dir_path, kwargs, exp)
            raise T2NErrorMisuse(msg) from exp
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
            raise T2NErrorMisuse(msg) from exp
        return hf_model_causal


def _resolve_snapshot_dir(slug: str, huggingface_hub) -> str:
    """Return the local snapshot dir for ``slug``, cache-first.

    ``list_repo_files`` has no cache mode and hits the rate-limited ``/tree``
    endpoint even for an already-cached model, so prefer ``snapshot_download``
    with ``local_files_only=True`` (no hub call when cached) and fall back to a
    networked ``snapshot_download`` only on a genuine cache miss (whose
    transient failures the LLMExporter.load retry then covers).
    """
    # local import: huggingface_hub is an optional extra (injected), so it is
    # only importable inside this call path which requires it.
    # pylint: disable-next=import-outside-toplevel
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return huggingface_hub.snapshot_download(slug, local_files_only=True)
    except LocalEntryNotFoundError:
        return huggingface_hub.snapshot_download(slug)


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="huggingface_hub")
@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
def _from_pretrained(
    slug_or_dir: T.Union[str, Path],
    auto_model_class,
    *,
    huggingface_hub: InjectedHuggingFaceHubModule = INJECTED,
    transformers: InjectedTransformersModule = INJECTED,
    **kwargs,
):
    if "device_map" in kwargs and kwargs["device_map"] is not None:
        device_map = kwargs.pop("device_map")
        if Path(slug_or_dir).exists():
            weights_location = Path(slug_or_dir)
        else:
            weights_location = Path(
                _resolve_snapshot_dir(str(slug_or_dir), huggingface_hub)
            )

        with init_empty_weights():
            model = auto_model_class.from_pretrained(slug_or_dir, **kwargs)

        if device_map == "auto":
            # pylint: disable-next=import-outside-toplevel
            import accelerate

            device_map = accelerate.infer_auto_device_map(model)
            LOGGER.info("device map selected: %s", device_map)
        if any(
            _ in device_map
            for _ in [AUTO_DEVICE_MAP_KEY, ON_DISK_DEVICE_MAP_KEY]
        ):
            t2n_load_checkpoint_and_dispatch(
                model,
                weights_location,
                device_map=device_map,
                offload_dir=Path(tempfile.mkdtemp(suffix="offload_t2n")),
            )
        elif device_map:
            # pylint: disable-next=import-outside-toplevel
            import accelerate

            model = accelerate.load_checkpoint_and_dispatch(
                model,
                weights_location,
                device_map=device_map,
                offload_folder=tempfile.mkdtemp(suffix="offload_accelerate"),
            )
        return model
    return auto_model_class.from_pretrained(slug_or_dir, **kwargs)


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
def load_model(
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    force_module_dtype: T.Optional[DtypeStr] = None,
    merge_peft: T.Optional[bool] = None,
    device_map: TYPE_OPTIONAL_DEVICE_MAP = None,
    trust_remote_code: bool = True,
    upcast_quant: T.Optional[T.Sequence[str]] = None,
    *,
    transformers: InjectedTransformersModule = INJECTED,
):
    """Load a model from a slug, local checkpoint, or custom config.

    ``trust_remote_code`` is forwarded to transformers. When True (the default,
    needed by models whose architecture ships custom code on the Hub), loading
    a model **executes arbitrary Python from its repository**: only export
    models you trust, or pass ``trust_remote_code=False`` (CLI
    ``--no-trust-remote-code``) to refuse it.
    """
    # validate requested up-cast methods up-front, before any download/load
    upcast_quant = _normalize_upcast_request(upcast_quant)
    # accept a str path from direct callers (the dump_llm path coerces upstream)
    if local_dir is not None:
        local_dir = Path(local_dir)
    if trust_remote_code:
        LOGGER.warning(
            "trust_remote_code is enabled: loading '%s' may execute arbitrary "
            "code from the model repository. Pass --no-trust-remote-code to "
            "refuse it (standard architectures load fine without it).",
            hf_model_slug or local_dir,
        )
    kwargs: T.Dict[str, T.Any] = {"trust_remote_code": trust_remote_code}
    # transformers 5.x defaults to a fused SDPA attention path that (a) passes
    # mixed-dtype q/k/v (bf16 query vs float key/value) which the SDPA kernel
    # rejects during the export forward, and (b) exports to the fused
    # tract_transformers_sdpa op, which diverges from PyTorch on tract. Eager
    # attention decomposes into core ops that export cleanly and track much
    # closer (28%-of-elements divergence -> f32 noise on SmolLM).
    if SemanticVersion.from_str(transformers.__version__) >= "5.0.0":
        kwargs["attn_implementation"] = "eager"
    if force_module_dtype is not None:
        key = "torch_dtype"
        if SemanticVersion.from_str(transformers.__version__) >= "4.57.0":
            key = "dtype"
        kwargs[key] = DtypeStr(force_module_dtype).torch_dtype

    if device_map is not None:
        kwargs["device_map"] = device_map

    custom_config = CUSTOM_CONFIGS.get(hf_model_slug or "")

    # Plan native-quant up-cast (fp8/fp4/mxfp4/... to dense float) before load,
    # and inject the load-time dequant flag into kwargs when applicable.
    config_source = local_dir if local_dir is not None else hf_model_slug
    upcast_plan = _plan_and_inject_upcast(
        config_source if custom_config is None else None,
        upcast_quant,
        kwargs,
        trust_remote_code,
        transformers,
    )

    if custom_config is not None:
        auto_model_class = resolve_auto_model_class(custom_config.model_type)
        hf_model_causal = auto_model_class.from_config(custom_config, **kwargs)
        LOGGER.info(
            "load custom config: '%s', un-initialized weights", hf_model_slug
        )
    elif local_dir:
        try:
            dir_path = find_subdir_with_filename_in(local_dir, "config.json")
            assert dir_path.is_dir(), dir_path
            assert_model_safetensors_exists(dir_path)
            config = transformers.AutoConfig.from_pretrained(
                dir_path, trust_remote_code=trust_remote_code
            )
            auto_model_class = resolve_auto_model_class(config.model_type)
            hf_model_causal = _from_pretrained(
                dir_path, auto_model_class, **kwargs
            )
            LOGGER.info(
                "load '%s' from local directory: %s",
                hf_model_causal.config.model_type,
                dir_path,
            )
        except (T2NErrorNotFoundFile, OSError):
            hf_model_causal = load_peft_model(local_dir, kwargs)
    elif hf_model_slug is not None:
        config = transformers.AutoConfig.from_pretrained(
            hf_model_slug, trust_remote_code=trust_remote_code
        )
        auto_model_class = resolve_auto_model_class(config.model_type)
        hf_model_causal = _from_pretrained(
            hf_model_slug, auto_model_class, **kwargs
        )
        LOGGER.info(
            "load default trained model from huggingface: '%s'", hf_model_slug
        )
    else:
        raise T2NErrorNotImplemented(
            "No local nor Huggingface slug, nor custom conf ?"
        )

    if merge_peft:
        # pylint: disable-next=import-outside-toplevel
        from peft import PeftModel

        if isinstance(hf_model_causal, PeftModel):
            hf_model_causal = hf_model_causal.merge_and_unload()
        else:
            LOGGER.warning(
                "no 'Peft' model found: %s (so no merge applied)",
                hf_model_causal.__class__,
            )

    # Finish the up-cast (post-load dequant + dense verification) before the
    # dtype cast below, since `.to(dtype)` does not dequantize quantized params.
    hf_model_causal = _finish_upcast(hf_model_causal, upcast_plan, upcast_quant)

    if force_module_dtype is not None:
        force_dtype = DtypeStr(force_module_dtype).torch_dtype
        hf_model_causal = hf_model_causal.to(force_dtype)
        LOGGER.info("force casted model internals to: '%s'", force_module_dtype)
    return hf_model_causal
