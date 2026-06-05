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

LOGGER = logging.getLogger(__name__)

TYPE_OPTIONAL_DEVICE_MAP = T.Optional[
    T.Union[
        str,
        T.Dict[str, T.Union[int, str, torch.device]],
        int,
        torch.device,
    ]
]


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
        local_dir or tokenizer_slug, trust_remote_code=True
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
            model = transformers.AutoModelForCausalLM.from_pretrained(
                slug_or_dir, **kwargs
            )

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
    return transformers.AutoModelForCausalLM.from_pretrained(
        slug_or_dir, **kwargs
    )


@require_extra_decorator(extra=T2NExtra.LLM_TRACT, module="transformers")
def load_model(
    hf_model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    force_module_dtype: T.Optional[DtypeStr] = None,
    merge_peft: T.Optional[bool] = None,
    device_map: TYPE_OPTIONAL_DEVICE_MAP = None,
    *,
    transformers: InjectedTransformersModule = INJECTED,
):
    """Load a model from a slug, local checkpoint, or custom config."""
    kwargs: T.Dict[str, T.Any] = {"trust_remote_code": True}
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
    if custom_config is not None:
        hf_model_causal = transformers.AutoModelForCausalLM.from_config(
            custom_config, **kwargs
        )
        LOGGER.info(
            "load custom config: '%s', un-initialized weights", hf_model_slug
        )
    elif local_dir:
        try:
            dir_path = find_subdir_with_filename_in(local_dir, "config.json")
            assert dir_path.is_dir(), dir_path
            assert_model_safetensors_exists(dir_path)
            hf_model_causal = _from_pretrained(dir_path, **kwargs)
            LOGGER.info(
                "load '%s' from local directory: %s",
                hf_model_causal.config.model_type,
                dir_path,
            )
        except (T2NErrorNotFoundFile, OSError):
            hf_model_causal = load_peft_model(local_dir, kwargs)
    elif hf_model_slug is not None:
        hf_model_causal = _from_pretrained(hf_model_slug, **kwargs)
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

    if force_module_dtype is not None:
        force_dtype = DtypeStr(force_module_dtype).torch_dtype
        hf_model_causal = hf_model_causal.to(force_dtype)
        LOGGER.info("force casted model internals to: '%s'", force_module_dtype)
    return hf_model_causal
