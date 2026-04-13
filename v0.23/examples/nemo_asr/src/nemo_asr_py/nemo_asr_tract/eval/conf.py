from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

from nemo_asr_tract.dataset import DatasetConfig

__all__ = ["DecodingStragegy", "EvalConfig", "EvalResult"]


class DecodingStragegy(str, Enum):
    """For now only greedy is supported.

    To avoid unfair engine comparison,
    we enforce greedy decoding across all engines.
    """

    GREEDY = "greedy"


@dataclass(frozen=True)
class EvalConfig:
    output_dir: Union[str, Path]
    model_dir: Union[str, Path]
    model_runner_class: str
    dataset: DatasetConfig
    device_id: int = -1
    decoding_stragegy: DecodingStragegy = (
        DecodingStragegy.GREEDY
    )  # ensure every engine is on same page
    warmup: int = 0


@dataclass(frozen=True)
class EvalResult:
    wer: float
    rtfx: float
    total_audio_seconds: float
    total_transcription_seconds: float
    manifest_path: str
    num_samples: int
