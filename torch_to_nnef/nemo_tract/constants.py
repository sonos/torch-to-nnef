"""Constants and small helpers for NeMo Tract wrappers.

This module centralizes shared string constants and tiny predicates used across
wrappers to keep the main logic files focused.
"""

import re
from typing import Any, Set

# Default time dimension used when fabricating audio-like examples
DEFAULT_TIME: int = 16000


# Common NeMo input names that represent lengths
LENGTH_INPUT_NAMES: Set[str] = {
    "length",
    "target_length",
    "processed_length",
    "audio_signal_length",
}

# Common NeMo output names that represent lengths
LENGTH_OUTPUT_NAMES: Set[str] = {
    "encoded_lengths",
    "prednet_lengths",
    "length",
    "processed_length",
    "audio_signal_length",
    "input_length",
}


# Names related to internal state tensors
STATE_INPUT_NAMES = ("input_states_1", "input_states_2", "states")
INPUT_STATE_TUPLE_NAME = "states"
OUT_STATE_NAME = "out_states"


def axis_kind_to_symbol(ax: Any) -> str:
    """Return a normalized symbol for an axis kind using full names.

    Newer Tract versions accept multi-letter symbols, so prefer clear names
    (e.g., BATCH, TIME, STREAM) instead of single letters.
    """
    kind = getattr(ax, "kind", ax)
    s = str(kind)
    # Strip enum-like prefixes (e.g., AxisKind.Batch -> Batch)
    if "." in s:
        s = s.rsplit(".", maxsplit=1)[-1]
    s = s.strip()
    return (s or "D").upper()


_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_]")


def _sanitize_name(name: str) -> str:
    return _SANITIZE_RE.sub("_", name).upper()


def make_axis_symbol(input_name: str, axis_kind: Any, axis_index: int) -> str:
    """Produce a readable axis symbol with intended sharing semantics.

    - Batch axis: always "BATCH" (shared globally across tensors).
    - Known kinds (e.g., TIME, STREAM, LENGTH): use the full kind name to
      intentionally share across tensors when they represent the same concept.
    - Unknown kinds: namespace by input name to avoid false collisions,
      using e.g. "AUDIO_SIGNAL_DIM0".
    """
    sym = axis_kind_to_symbol(axis_kind)
    up = sym.upper()
    if up == "B" or "BATCH" in up:
        return "BATCH"
    base = up if up and up not in {"?", "D", "DIM"} else f"DIM{axis_index}"
    return f"{_sanitize_name(input_name)}__{base}"


def is_length_name(name: str) -> bool:
    nl = name.lower()
    return (
        nl in LENGTH_INPUT_NAMES
        or nl in LENGTH_OUTPUT_NAMES
        or ("length" in nl)
    )
