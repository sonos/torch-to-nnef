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
"""

import typing as T

from torch_to_nnef.nemo_tract.model_loader import (
    NEMOTRON_0_6B,
    PARAKEET_V3_SLUG,
)

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
