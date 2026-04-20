"""Manually maintained registry of per-slug, per-subnet extensions.

Some NeMo models have implicit dimensionality constraints (e.g., max
receptive field of a conv stack) that are not exposed by NeMo but are
required by tract for correct pulsification.  This registry maps known
pretrained model IDs to the extensions that encode those constraints.

Structure::

    SLUG_EXTENSIONS = {
        "<pretrained-id>": {
            "<subnet>": ["<extension-string>", ...],
        },
    }

For local ``.nemo`` files (finetunes), the HuggingFace slug is not stored
inside the archive.  A parallel ``slug_fingerprints.json`` file maps each
known slug to an ``EncoderFingerprint`` derived from its pretrained
``model_config.yaml``.  Finetuning does not alter the architecture fields
in the fingerprint, so a finetune of a known pretrained will match the
same fingerprint and inherit its extensions.

Maintainer flow: add a slug to ``SLUG_EXTENSIONS`` then run
``python -m torch_to_nnef_nemo.tools.refresh_slug_fingerprints`` to
regenerate the JSON from ``ASRModel.from_pretrained``.
"""

import json
import logging
import typing as T
from dataclasses import dataclass, fields
from pathlib import Path

from torch_to_nnef_nemo.model_loader import (
    NEMOTRON_0_6B,
    PARAKEET_V3_SLUG,
)

LOGGER = logging.getLogger(__name__)

# -- registry ----------------------------------------------------------------
# Extend this mapping as new models are validated for pulsified export.

_ENCODER_TIME_ASSERT_39993 = ["tract_assert AUDIO_SIGNAL__TIME<=39993"]

SLUG_EXTENSIONS: T.Dict[str, T.Dict[str, T.List[str]]] = {
    NEMOTRON_0_6B: {
        "encoder": _ENCODER_TIME_ASSERT_39993,
    },
    PARAKEET_V3_SLUG: {
        "encoder": _ENCODER_TIME_ASSERT_39993,
    },
}


def get_extensions_for_slug(
    slug: str,
) -> T.Dict[str, T.List[str]]:
    """Return per-subnet extensions for a known pretrained model slug.

    Returns an empty dict when the slug has no registered extensions.
    """
    return SLUG_EXTENSIONS.get(slug, {})


# -- fingerprints ------------------------------------------------------------
# Auto-filled by ``tools/refresh_slug_fingerprints.py``.  Do not edit by hand.

FINGERPRINTS_PATH = Path(__file__).with_name("slug_fingerprints.json")


@dataclass(frozen=True)
class EncoderFingerprint:
    """Architecture-level identity of a NeMo ASR encoder.

    Compared for exact equality during slug resolution.  Fields are
    chosen to be stable across finetuning (training only touches
    weights, not these config entries) while discriminative across the
    pretrained slugs we track.
    """

    model_class: str
    encoder_class: str
    feat_in: int
    d_model: int
    n_layers: int
    subsampling: str
    subsampling_factor: int
    conv_kernel_size: int
    self_attention_model: str
    att_context_size: T.Tuple[T.Tuple[int, int], ...]
    num_tdt_durations: T.Optional[int]

    @classmethod
    def from_dict(cls, d: T.Mapping[str, T.Any]) -> "EncoderFingerprint":
        return cls(
            model_class=str(d["model_class"]),
            encoder_class=str(d["encoder_class"]),
            feat_in=int(d["feat_in"]),
            d_model=int(d["d_model"]),
            n_layers=int(d["n_layers"]),
            subsampling=str(d["subsampling"]),
            subsampling_factor=int(d["subsampling_factor"]),
            conv_kernel_size=int(d["conv_kernel_size"]),
            self_attention_model=str(d["self_attention_model"]),
            att_context_size=_normalize_att_context_size(d["att_context_size"]),
            num_tdt_durations=(
                int(d["num_tdt_durations"])
                if d.get("num_tdt_durations") is not None
                else None
            ),
        )

    def to_dict(self) -> T.Dict[str, T.Any]:
        return {
            "model_class": self.model_class,
            "encoder_class": self.encoder_class,
            "feat_in": self.feat_in,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "subsampling": self.subsampling,
            "subsampling_factor": self.subsampling_factor,
            "conv_kernel_size": self.conv_kernel_size,
            "self_attention_model": self.self_attention_model,
            "att_context_size": [list(p) for p in self.att_context_size],
            "num_tdt_durations": self.num_tdt_durations,
        }


def _normalize_att_context_size(
    value: T.Any,
) -> T.Tuple[T.Tuple[int, int], ...]:
    """Coerce a yaml ``att_context_size`` into a tuple-of-pairs.

    NeMo encodes this field two ways: as a flat ``[left, right]`` pair
    for offline models, or as a list of pairs for multi-context
    streaming models.  We normalize both to the same shape so the
    fingerprint dataclass has a single type.
    """
    seq = list(value)
    if not seq:
        return ()
    # Flat pair ``[left, right]`` (offline) vs list of pairs (streaming).
    # OmegaConf's ``ListConfig`` is not a ``list``, so detect the leaf
    # shape by checking whether the first element is an integer.
    if isinstance(seq[0], int):
        return ((int(seq[0]), int(seq[1])),)
    return tuple((int(p[0]), int(p[1])) for p in seq)


def fingerprint_from_model_cfg(
    model_cfg: T.Any,
    model_class: str,
    encoder_class: str,
) -> EncoderFingerprint:
    """Build a fingerprint from an OmegaConf ``asr_model.cfg``.

    ``model_class`` and ``encoder_class`` are passed in separately
    because we read them from the runtime class rather than from the
    cfg's ``_target_`` strings -- ``target`` is stripped from
    ``asr_model.cfg`` after instantiation, and we want the same values
    whether we ran through ``from_pretrained`` or ``restore_from``.
    """
    enc = model_cfg.encoder
    md = (
        model_cfg.get("model_defaults", None)
        if hasattr(model_cfg, "get")
        else None
    )
    num_tdt: T.Optional[int] = None
    if md is not None:
        raw = md.get("num_tdt_durations", None) if hasattr(md, "get") else None
        if raw is not None:
            num_tdt = int(raw)
    return EncoderFingerprint(
        model_class=model_class,
        encoder_class=encoder_class,
        feat_in=int(enc.feat_in),
        d_model=int(enc.d_model),
        n_layers=int(enc.n_layers),
        subsampling=str(enc.subsampling),
        subsampling_factor=int(enc.subsampling_factor),
        conv_kernel_size=int(enc.conv_kernel_size),
        self_attention_model=str(enc.self_attention_model),
        att_context_size=_normalize_att_context_size(enc.att_context_size),
        num_tdt_durations=num_tdt,
    )


def fingerprint_from_asr_model(asr_model: T.Any) -> EncoderFingerprint:
    """Build a fingerprint from a loaded NeMo ``ASRModel``."""
    return fingerprint_from_model_cfg(
        asr_model.cfg,
        model_class=type(asr_model).__name__,
        encoder_class=type(asr_model.encoder).__name__,
    )


def load_slug_fingerprints() -> T.Dict[str, EncoderFingerprint]:
    """Load the committed slug -> fingerprint map.

    Returns an empty dict when the JSON is missing (fresh clone that
    hasn't run the refresh script yet).
    """
    if not FINGERPRINTS_PATH.exists():
        return {}
    raw = json.loads(FINGERPRINTS_PATH.read_text())
    return {slug: EncoderFingerprint.from_dict(v) for slug, v in raw.items()}


def resolve_slug_from_asr_model(asr_model: T.Any) -> T.Optional[str]:
    """Reverse-lookup: arch fingerprint -> known pretrained slug.

    Returns ``None`` if no known slug matches.  Logs the computed
    fingerprint at DEBUG level so maintainers can paste it into a
    regen when adding a new slug.
    """
    fp = fingerprint_from_asr_model(asr_model)
    for slug, known in load_slug_fingerprints().items():
        if known == fp:
            return slug
    LOGGER.debug(
        "no slug fingerprint matched; computed=%s",
        {f.name: getattr(fp, f.name) for f in fields(fp)},
    )
    return None
