from ctypes import c_size_t

from .nemo_asr import (
    NemoAsrConfig,
    NemoAsrModel,
    Transcript,
    TranscriptItem,
    Transcripts,
    RuntimeConfig,
    load_config_from_dir,
)
from .utils import check_ffi_error, lib

__version__ = "0.1.0"


def init_env_logger(verbosity: int = 0):
    exit_code = lib.nemo_asr_init_env_logger(c_size_t(int(verbosity)))
    check_ffi_error(
        exit_code, "Something went wrong when initializing env logger"
    )


__all__ = [
    "init_env_logger",
    "NemoAsrConfig",
    "NemoAsrModel",
    "Transcript",
    "TranscriptItem",
    "Transcripts",
    "load_config_from_dir",
    "RuntimeConfig",
]
