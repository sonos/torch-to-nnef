import datetime
import logging
import sys
import typing as T
from dataclasses import dataclass
from pathlib import Path

import torch

from torch_to_nnef._optional_types import (
    HuggingFaceHubModule,
    InjectedHuggingFaceHubModule,
    InjectedNemoModule,
    InjectedQuestionaryModule,
)
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

LOGGER = logging.getLogger(__name__)


# Popular slugs (kept for compatibility/tests)
PARAKEET_V3_SLUG = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_110M_SLUG = "parakeet-tdt_ctc-110m"
NEMOTRON_0_6B = "nvidia/nemotron-speech-streaming-en-0.6b"


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="questionary")
def ask_model_selector(
    pretrained_model_info_list,
    *,
    questionary: InjectedQuestionaryModule = INJECTED,
    **kwargs,
):
    question = questionary.select(
        " ∵ What model do you want to export ? "
        "(those starting with nvidia/ are from 🤗Hub)",
        qmark="",
        choices=[
            questionary.Choice(
                title=pretrained_model_info.pretrained_model_name,
                description=pretrained_model_info.description,
            )
            for pretrained_model_info in pretrained_model_info_list
        ],
        use_jk_keys=False,
        use_search_filter=True,
        **kwargs,
    )
    return question.ask()


def nemo_asr_hg_list(huggingface_hub: HuggingFaceHubModule):
    """Return the list of available NeMo ASR models from HuggingFace."""
    hugging_face_hub_model_list = []

    models = huggingface_hub.list_models(
        author="nvidia",
        task="automatic-speech-recognition",
        library="nemo",
        full=True,
    )

    for m in models:
        if getattr(m, "gated", False):
            continue
        tags = set(m.tags or [])
        if "automatic-speech-recognition" not in tags:
            continue
        jtags = "tags:" + ", ".join(sorted(tags))

        def last_modified_str(dt: datetime.datetime) -> str:
            fmt = "%Y-%m-%d"
            return f"last modified: {dt.strftime(fmt)}"

        hugging_face_hub_model_list.append(
            HGModelInfo(
                pretrained_model_name=m.modelId,
                organization="nvidia",
                description=(
                    "HuggingFace (compatibility only guessed): "
                    + (m.cardData.get("description") if m.cardData else jtags)
                    + last_modified_str(m.lastModified)
                ),
                tags=sorted(tags),
                pipeline_tag=m.pipeline_tag,
                library_name=m.library_name,
                sha=m.sha,
                last_modified=m.lastModified,
            )
        )

    hugging_face_hub_model_list.sort(
        key=lambda x: x.last_modified, reverse=True
    )
    return hugging_face_hub_model_list


@dataclass(frozen=True)
class HGModelInfo:
    pretrained_model_name: str
    organization: str
    description: str
    tags: T.List[str]
    pipeline_tag: str
    library_name: str
    sha: str
    last_modified: datetime.datetime


@require_extra_decorator(
    extra=T2NExtra.NEMO_TRACT, module="nemo.collections.asr", kw="nemo_asr"
)
@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="huggingface_hub")
def load_asr_model_from_nemo_slug(
    model_slug: str,
    *,
    nemo_asr: InjectedNemoModule = INJECTED,
    huggingface_hub: InjectedHuggingFaceHubModule = INJECTED,
):
    """Load a NeMo ASR model from a given model slug."""
    # pylint: disable=import-outside-toplevel
    from huggingface_hub import errors

    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_slug, map_location=torch.device("cpu")
        )
    except (errors.RepositoryNotFoundError, FileNotFoundError):
        LOGGER.error("Could not find model with slug: %s", model_slug)
        if model_slug != "*":
            while True:
                resp = (
                    input("Do you want to list available models? (y/n): ")
                    .strip()
                    .lower()
                )
                if resp in ("y", "n"):
                    break
            if resp == "n":
                LOGGER.info("User chose not to list available models. Exiting.")
                sys.exit(1)

        available_models = (
            nemo_asr_hg_list(huggingface_hub)
            + nemo_asr.models.ASRModel.list_available_models()
        )
        model_slug = ask_model_selector(available_models)
        LOGGER.info("selected model slug: %s", model_slug)
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_slug,
            map_location=torch.device("cpu"),
        )
    return asr_model


@require_extra_decorator(
    extra=T2NExtra.NEMO_TRACT, module="nemo.collections.asr", kw="nemo_asr"
)
def load_asr_model_from_path(
    model_path: Path,
    *,
    nemo_asr: InjectedNemoModule = INJECTED,
):
    return nemo_asr.models.ASRModel.restore_from(
        restore_path=model_path, map_location=torch.device("cpu")
    )
