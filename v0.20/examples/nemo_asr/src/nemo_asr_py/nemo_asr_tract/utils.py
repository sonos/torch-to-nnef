import logging
import platform
from _ctypes import POINTER, Structure, byref
from contextlib import contextmanager
from ctypes import c_char_p, c_float, c_int32, c_uint, cdll, string_at
from pathlib import Path
from typing import Optional

python_version = "".join(platform.python_version_tuple()[:2])
dylib_dir = Path(__file__).parent / "dylib"
pattern = f"libnemo_asr.cpython-{python_version}*"
try:
    dylib_path = next(iter(dylib_dir.glob(pattern)))
except StopIteration as ex:
    raise FileNotFoundError(
        f"Could not find dylib with pattern {pattern} in {dylib_dir}"
    ) from ex
lib = cdll.LoadLibrary(str(dylib_path))


class CStringArray(Structure):
    _fields_ = [("data", POINTER(c_char_p)), ("size", c_int32)]

    def to_pylist(self):
        return [self.data[i].decode("utf8") for i in range(self.size)]


@contextmanager
def string_pointer(ptr):
    try:
        yield ptr
    finally:
        lib.nemo_asr_destroy_string(ptr)


def check_ffi_error(exit_code, error_context_msg):
    if exit_code != 0:
        with string_pointer(c_char_p()) as ptr:
            if lib.nemo_asr_get_last_error(byref(ptr)) == 0:
                ffi_error_message = string_at(ptr).decode("utf8")
            else:
                ffi_error_message = "see stderr"
        raise ValueError(f"{error_context_msg}: {ffi_error_message}")


def opt_float_to_c(value: Optional[float]) -> Optional[c_float]:
    return byref(c_float(float(value))) if value is not None else None


def opt_uint_to_c(value: Optional[int]) -> Optional[c_uint]:
    return byref(c_uint(int(value))) if value is not None else None


def init_logfile(filepath: Path, verbose: bool = False) -> int:
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG

    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler: logging.Handler = logging.FileHandler(filepath)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s,%(msecs)d %(levelname)-8s "
            "[%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d:%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    try:
        # pylint: disable-next=import-outside-toplevel
        from rich.logging import RichHandler

        handler = RichHandler(show_path=False)
    except ImportError:
        handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(message)s",
            "%Y-%m-%d:%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return log_level
