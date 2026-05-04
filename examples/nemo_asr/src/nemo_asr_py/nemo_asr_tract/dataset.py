"""Dataset / audio for evaluation/calib.

This is not generic enough to support all datasets in HG.
It's tailored to work with ESB datasets.

"""

import io
import os
from dataclasses import dataclass

import numpy as np
import soundfile
from datasets.features._torchcodec import AudioDecoder
from tqdm import tqdm

from nemo_asr_tract.normalizer import data_utils

DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")

AUDIO_FILEPATHS_KEY = "audio_filepaths"
REFERENCES_KEY = "references"
DURATION_KEY = "durations"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for evaluation/calibration dataset."""

    hg_path: str
    name: str
    split: str = "test"
    max_eval_samples: int | None = None
    batch_size: int = 32
    streaming: bool = True
    remap: dict[str, str] | None = None

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.split}"


def ensure_cache_dir(dataset: str, split: str) -> str:
    cache_dir = os.path.join(DATA_CACHE_DIR, dataset, split)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_audio(sample):
    if isinstance(sample, AudioDecoder):
        full_sample = sample.get_all_samples()
        data = full_sample.data.float().numpy()
        assert data.shape[0] == 1, "Expected mono audio"
        assert len(data.shape) == 2, "Expected 2D audio array"
        return data[0], full_sample.sample_rate
    if "array" in sample:
        return np.float32(sample["array"]), 16000

    if "bytes" in sample:
        with io.BytesIO(sample["bytes"]) as f:
            return soundfile.read(f, dtype="float32")

    raise ValueError("Invalid audio sample format")


def download_audio_files_factory(cache_dir: str):
    def fn(batch):
        audio_paths, durations = [], []

        for sample_id, sample in zip(batch["id"], batch["audio"], strict=False):
            sample_id = sample_id.replace("/", "_").removesuffix(".wav")
            audio_path = os.path.join(cache_dir, f"{sample_id}.wav")

            audio, sr = load_audio(sample)

            if not os.path.exists(audio_path):
                soundfile.write(audio_path, audio, sr)

            audio_paths.append(audio_path)
            durations.append(len(audio) / sr)

        batch[AUDIO_FILEPATHS_KEY] = audio_paths
        batch[DURATION_KEY] = durations
        batch[REFERENCES_KEY] = batch["norm_text"]
        return batch

    return fn


def prepare_dataset(cfg: DatasetConfig, cache_dir: str):
    ds = data_utils.load_data(cfg)

    if cfg.max_eval_samples:
        ds = ds.take(cfg.max_eval_samples)

    ds = data_utils.prepare_data(ds)

    return ds.map(
        download_audio_files_factory(cache_dir),
        batched=True,
        batch_size=cfg.batch_size,
        remove_columns=["audio"],
    )


def collect_dataset(dataset, remap_keys: dict | None = None):
    all_data = {AUDIO_FILEPATHS_KEY: [], DURATION_KEY: [], REFERENCES_KEY: []}

    for sample in tqdm(iter(dataset), desc="Preparing samples"):
        for k in all_data:
            if remap_keys is None:
                all_data[k].append(sample[k])
            else:
                all_data[k].append(sample[remap_keys.get(k, k)])

    return all_data


def sort_by_duration(all_data):
    order = sorted(
        range(len(all_data[DURATION_KEY])),
        key=lambda i: all_data[DURATION_KEY][i],
        reverse=True,
    )

    for k in all_data:
        all_data[k] = [all_data[k][i] for i in order]

    return all_data
