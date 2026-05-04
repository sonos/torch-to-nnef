from ctypes import byref, c_char_p, c_size_t, c_void_p
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel as PydanticModel
from pydantic import RootModel

from .utils import check_ffi_error, lib


class TranscriptItem(PydanticModel):
    token: str
    logit: float
    emitted_at_encoder_timestep: int
    emitted_at_encoder_timestep_iteration: int


class Transcript(PydanticModel):
    text: str
    items: List[TranscriptItem]


class Transcripts(RootModel):
    root: List[Transcript]


class NemoAsrConfig(PydanticModel):
    pretrained_name: str
    nemo_version: str
    target: str
    sample_rate: int
    labels: List[str]


class RuntimeConfig(PydanticModel):
    max_n_tokens_per_step: Optional[int] = None
    force_cpu: bool = False
    encoder_per_batch: bool = True
    dump_intermediate_io_path: Optional[Path] = None


MODEL_CONFIG_FILENAME = "model_config.json"


class NemoAsrModel:
    """Class encapsulating Rust NemoAsrModel.

    Attrs:
        ptr:
            Pointer to Rust NemoAsrModel instance.
    """

    def __init__(self, rs_asr_model, path: Union[str, Path]):
        if not isinstance(rs_asr_model, type(c_void_p())):
            raise TypeError("Expected a rs_asr_model as argument to __init__. ")
        self.ptr = rs_asr_model
        self.path = Path(path)

    @classmethod
    def from_dir(cls, path: Union[str, Path]):
        assert Path(path).is_dir(), f"Expected directory path, got: {path}"
        assert (Path(path) / MODEL_CONFIG_FILENAME).is_file(), (
            f"Expected '{MODEL_CONFIG_FILENAME}' file in directory: {path}"
        )
        ptr = c_void_p()
        check_ffi_error(
            lib.nemo_asr_from_dir(
                byref(ptr),
                c_char_p(str(Path(path).absolute()).encode("utf-8")),
            ),
            "Error while creating NemoAsrModel",
        )
        return cls(ptr, path)

    @classmethod
    def from_dir_with_runtime_config(
        cls, path: Union[str, Path], config: RuntimeConfig
    ):
        assert Path(path).is_dir(), f"Expected directory path, got: {path}"
        assert (Path(path) / MODEL_CONFIG_FILENAME).is_file(), (
            f"Expected '{MODEL_CONFIG_FILENAME}' file in directory: {path}"
        )
        assert isinstance(config, RuntimeConfig), (
            f"Expected RuntimeConfig instance, got: {type(config)}"
        )
        ptr = c_void_p()
        check_ffi_error(
            lib.nemo_asr_from_dir_with_runtime_config(
                byref(ptr),
                c_char_p(str(Path(path).absolute()).encode("utf-8")),
                c_char_p(config.json().encode("utf-8")),
            ),
            "Error while creating NemoAsrModel",
        )
        return cls(ptr, path)

    def infer_from_wav_paths(self, wavs: List[Union[str, Path]]) -> str:
        for in_wav in wavs:
            assert Path(in_wav).is_file(), f"Expected file path, got: {in_wav}"
        ptr = c_char_p()

        def clean_ptr():
            check_ffi_error(
                lib.nemo_asr_destroy_string(ptr),
                "Error while destroying default Transcripts string",
            )

        # Python strings → bytes (C strings)
        c_string_wavs = [
            str(Path(path).absolute()).encode("utf-8") for path in wavs
        ]
        # Build array type
        ArrayType = c_char_p * len(c_string_wavs)
        # Instantiate array
        c_array = ArrayType(*c_string_wavs)

        check_ffi_error(
            lib.infer_from_wav_paths(
                self.ptr,
                c_array,
                c_size_t(len(wavs)),
                byref(ptr),
            ),
            "Error while extracting default Transcripts",
        )
        if ptr.value is None:
            clean_ptr()
            raise ValueError(
                "unexpected empty pointer should be filled with "
                "json Transcripts"
            )
        loading_config = Transcripts.model_validate_json(
            ptr.value.decode("utf-8")
        )
        clean_ptr()
        return loading_config.root

    @property
    def config(self) -> NemoAsrConfig:
        return load_config_from_dir(self.path)

    def __del__(self):
        check_ffi_error(
            lib.nemo_asr_model_destroy(self.ptr),
            "Error while destroying NemoAsrModel",
        )


def load_config_from_dir(path: Union[str, Path]) -> NemoAsrConfig:
    """Loads the Nemo ASR model config from a given directory."""
    with (
        Path(path)
        .joinpath(MODEL_CONFIG_FILENAME)
        .open("r", encoding="utf-8") as f
    ):
        return NemoAsrConfig.model_validate_json(f.read())
