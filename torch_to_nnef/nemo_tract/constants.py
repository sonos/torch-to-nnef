"""Constants and small helpers for NeMo Tract wrappers.

This module centralizes shared string constants and tiny predicates used across
wrappers to keep the main logic files focused.
"""

from typing import Set

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


# Mapping from NeMo axis kinds to symbolic names used by exporters
AXIS_KIND_TO_SYMBOL = {
    "batch": "B",
    "time": "T",
    "stream": "S",
}


def is_length_name(name: str) -> bool:
    nl = name.lower()
    return (
        nl in LENGTH_INPUT_NAMES
        or nl in LENGTH_OUTPUT_NAMES
        or ("length" in nl)
    )
