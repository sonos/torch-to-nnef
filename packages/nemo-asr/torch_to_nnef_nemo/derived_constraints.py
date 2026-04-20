"""Architecture-derived tract extensions for NeMo ASR encoders.

Instead of hand-maintaining per-slug ``tract_assert ...`` strings in
``slug_extensions.py``, derive them from the encoder's architecture
config.  The same derivation applies unchanged to any finetune of a
known pretrained, since finetuning does not alter ``pos_emb_max_len``,
``subsampling``, ``subsampling_factor`` or ``self_attention_model``.

Design: the slug registry in ``slug_extensions.py`` remains as an
overrides/quirks path for constraints we can't (yet) derive principled.
At merge time we apply both; deduplication keeps the union.
"""

from __future__ import annotations

import logging
import typing as T

LOGGER = logging.getLogger(__name__)

# Attention variants whose positional encoding table is sized by
# ``pos_emb_max_len``.  For ``rel_pos_local_attn`` the PE table is
# actually sized by the local window (``left + right + 1``) rather than
# by the sequence length -- so in principle the bound does not apply.
# We keep it in the bounded set for parity with the committed registry
# (see NEMOTRON_0_6B in slug_extensions.SLUG_EXTENSIONS); revisit if
# empirical tract runs show the bound is too tight for streaming.
_REL_POS_ATTENTIONS = frozenset({"rel_pos", "rel_pos_local_attn", "abs_pos"})

# Conv-based subsampling flavors whose preimage follows the same
# ``(enc_max - 1) * factor + 1`` recurrence (kernel=3 or compatible,
# stride=2, symmetric padding, non-causal).  ``stacking`` and friends
# use different math and are not included.
_CONV_SUBSAMPLINGS = frozenset(
    {"dw_striding", "striding", "vggnet", "striding_conv1d"}
)


def _encoder_frame_bound(encoder_cfg: T.Any) -> T.Optional[int]:
    """Max encoder-internal frame count imposed by the positional table.

    NeMo's ``RelPositionalEncoding`` builds a PE table of size
    ``pos_emb_max_len`` and extends it at runtime if the sequence is
    longer -- but a dynamic extension is not representable as a static
    NNEF graph, so tract requires ``encoder_frames <= pos_emb_max_len``.
    Returns ``None`` if the attention variant is not in the bounded set.
    """
    sa = str(encoder_cfg.self_attention_model)
    if sa not in _REL_POS_ATTENTIONS:
        return None
    return int(encoder_cfg.pos_emb_max_len)


def _subsampling_preimage(
    encoder_max: int,
    subsampling: str,
    subsampling_factor: int,
) -> T.Optional[int]:
    """Max pre-subsampling mel frames given encoder-frame upper bound.

    Uses the naive ``(enc_max - 1) * factor + 1`` inversion, which is
    up to ``factor - 1`` frames tighter than NeMo's ``calc_length``
    true preimage.  Matches the committed hardcodes and is strictly
    safe (never exceeds the true max).  Returns ``None`` for
    subsamplings we have not validated.
    """
    if subsampling not in _CONV_SUBSAMPLINGS:
        return None
    return (encoder_max - 1) * subsampling_factor + 1


def derive_encoder_time_bound(encoder_cfg: T.Any) -> T.Optional[int]:
    """Compute the mel-frame upper bound for this encoder, or ``None``.

    ``None`` when either the attention variant or the subsampling
    flavor is not in the supported set -- in that case the caller
    should fall back to the slug override registry.
    """
    enc_max = _encoder_frame_bound(encoder_cfg)
    if enc_max is None:
        return None
    return _subsampling_preimage(
        enc_max,
        str(encoder_cfg.subsampling),
        int(encoder_cfg.subsampling_factor),
    )


def derive_extensions(asr_model: T.Any) -> T.Dict[str, T.List[str]]:
    """Per-subnet tract extensions derived from a loaded ``ASRModel``.

    Empty dict when the encoder architecture has no derivable bounds.
    """
    bound = derive_encoder_time_bound(asr_model.cfg.encoder)
    if bound is None:
        return {}
    return {"encoder": [f"tract_assert AUDIO_SIGNAL__TIME<={bound}"]}
