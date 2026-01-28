"""Basic calibration utilities for quantization tasks."""

from typing import Iterable, List

from nemo_asr_tract.dataset import (
    AUDIO_FILEPATHS_KEY,
    REFERENCES_KEY,
    DatasetConfig,
    collect_dataset,
    ensure_cache_dir,
    prepare_dataset,
    sort_by_duration,
)
from nemo_asr_tract.utils import chunks


def iter_calibration_data(cfg: DatasetConfig) -> Iterable[List[str]]:
    """Prepare calibration data for quantization of Nemo models."""
    cache_dir = ensure_cache_dir(cfg.name, cfg.split)
    dataset = prepare_dataset(cfg, cache_dir)
    all_data = sort_by_duration(collect_dataset(dataset))

    for batch in chunks(all_data[AUDIO_FILEPATHS_KEY], cfg.batch_size):
        yield batch


LIBRISPEECH_CLEAN_512_TRAIN_CONFIG = DatasetConfig(
    name="clean",
    split="train.100",
    hg_path="openslr/librispeech_asr",
    batch_size=8,
    max_eval_samples=512,
    streaming=True,
    remap={"audio": AUDIO_FILEPATHS_KEY, "text": REFERENCES_KEY},
)

LIBRISPEECH_CLEAN_1024_TRAIN_CONFIG = DatasetConfig(
    name="clean",
    split="train.100",
    hg_path="openslr/librispeech_asr",
    batch_size=8,
    max_eval_samples=1024,
    streaming=True,
    remap={"audio": AUDIO_FILEPATHS_KEY, "text": REFERENCES_KEY},
)
