from .utils import lib, check_ffi_error
from .nemo_asr import *  # NOQA
from ctypes import c_size_t


def init_env_logger(verbosity: int = 0):
    exit_code = lib.nemo_asr_init_env_logger(c_size_t(int(verbosity)))
    check_ffi_error(
        exit_code, "Something went wrong when initializing env logger"
    )
