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
    """Produce a readable, namespaced axis symbol.

    - Batch axis: always namespaced as ``INPUT__BATCH`` (per-input),
      not a global ``BATCH``.
    - Known kinds (e.g., TIME, STREAM, LENGTH): use the full kind name and
      namespace by input (e.g., ``AUDIO_SIGNAL__TIME``) to avoid accidental
      cross-input collisions.
    - Unknown kinds: fall back to a generic numbered dim also namespaced by
      input (e.g., ``AUDIO_SIGNAL__DIM0``).
    """
    sym = axis_kind_to_symbol(axis_kind)
    up = sym.upper()
    if up == "BATCH":
        # Always namespace batch by input to keep interfaces consistent
        return f"{_sanitize_name(input_name)}__BATCH"
    # Heuristic: NeMo RNN-T decoders expose recurrent states as
    #   states: [L, B, H] (or expanded to states_0/states_1)
    # Their axis-annotations may not always label the batch axis.
    # To keep axis sharing consistent across tuple elements and match Tract's
    # expectations, force axis 1 of any state input to be BATCH.
    in_up = _sanitize_name(input_name)
    if (
        in_up == "STATES"
        or in_up.startswith("STATES_")
        or in_up.startswith("INPUT_STATES_")
    ) and axis_index == 1:
        return f"{_sanitize_name(input_name)}__BATCH"
    base = up if up and up not in {"?", "D", "DIM"} else f"DIM{axis_index}"
    return f"{_sanitize_name(input_name)}__{base}"


def is_length_name(name: str) -> bool:
    nl = name.lower()
    return (
        nl in LENGTH_INPUT_NAMES
        or nl in LENGTH_OUTPUT_NAMES
        or ("length" in nl)
    )
