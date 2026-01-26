"""Public eval API datatypes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class EvalConfig:
    output_dir: Union[str, Path]
    model_dir: Union[str, Path]
    model_runner_class: str
    dataset_path: str
    dataset: str
    split: str = "test"
    device_id: int = -1
    batch_size: int = 32
    max_eval_samples: int | None = None
    streaming: bool = True
    warmup: int = 0


@dataclass(frozen=True)
class EvalResult:
    wer: float
    rtfx: float
    total_audio_seconds: float
    total_transcription_seconds: float
    manifest_path: str
    num_samples: int
