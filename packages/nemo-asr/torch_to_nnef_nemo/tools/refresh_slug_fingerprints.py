"""Regenerate ``slug_fingerprints.json`` from the slugs in SLUG_EXTENSIONS.

Run this after adding a new slug to ``SLUG_EXTENSIONS`` (in
``slug_extensions.py``) so that local ``.nemo`` finetunes of that slug
can be auto-resolved by architecture fingerprint.

Each slug is loaded via ``ASRModel.from_pretrained`` (downloading from
HuggingFace on first use; subsequent runs reuse the cached ``.nemo``)
and introspected to produce a fingerprint.  The JSON file is then
rewritten in place.

Usage::

    python -m torch_to_nnef_nemo.tools.refresh_slug_fingerprints
"""

from __future__ import annotations

import json
import logging
import sys

from torch_to_nnef_nemo.model_loader import load_asr_model_from_nemo_slug
from torch_to_nnef_nemo.slug_extensions import (
    FINGERPRINTS_PATH,
    SLUG_EXTENSIONS,
    fingerprint_from_asr_model,
)

LOGGER = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out: dict = {}
    for slug in sorted(SLUG_EXTENSIONS):
        LOGGER.info("loading %s", slug)
        asr_model = load_asr_model_from_nemo_slug(slug)
        fp = fingerprint_from_asr_model(asr_model)
        LOGGER.info("  -> %s", fp)
        out[slug] = fp.to_dict()
    FINGERPRINTS_PATH.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("wrote %s", FINGERPRINTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
