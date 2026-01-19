"""
This script evaluates an ASR model from NVIDIA NeMo on a specified dataset split,
calculating Word Error Rate (WER) and Real-Time Factor (RTF).

Based on the evaluation script from the Open ASR Leaderboard:
https://github.com/huggingface/open_asr_leaderboard/blob/main/nemo_asr/run_eval.py
ASR evaluation script (NeMo / exported models)

Features:
- Callable Python API
- Clean CLI wrapper
- Typed config & results

"""

import argparse
import io
import os
import time
from dataclasses import dataclass
from typing import Union, List
from functools import partial

import evaluate
import numpy as np
import soundfile
import torch
from nemo.collections.asr.models import ASRModel
from datasets.features._torchcodec import AudioDecoder
from tqdm import tqdm

from nemo_asr_tract.nemo_asr import NemoAsrModel, load_config_from_dir
from nemo_asr_tract.normalizer import data_utils


# =============================================================================
# Constants
# =============================================================================

WER_METRIC = evaluate.load("wer")
DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")


# =============================================================================
# Public API datatypes
# =============================================================================


@dataclass(frozen=True)
class EvalConfig:
    exported_dir: str
    dataset_path: str
    dataset: str
    use_original_model: bool = False
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


# =============================================================================
# Device / model
# =============================================================================


def setup_device(device_id: int):
    if device_id >= 0:
        return torch.device(f"cuda:{device_id}"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def load_asr_model(
    exported_dir: str,
    use_original_model: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Union[ASRModel, NemoAsrModel]:
    conf = load_config_from_dir(exported_dir)

    if not use_original_model:
        return NemoAsrModel.from_dir(exported_dir)

    model_id = conf.pretrained_name
    if model_id.endswith(".nemo"):
        model = ASRModel.restore_from(model_id, map_location=device)
    else:
        model = ASRModel.from_pretrained(model_id, map_location=device)

    model.to(dtype)
    model.eval()
    return model


def ensure_decoding_strategy(model: ASRModel):
    if model.cfg.decoding.strategy != "beam":
        model.cfg.decoding.strategy = "greedy_batch"
        model.change_decoding_strategy(model.cfg.decoding)


# =============================================================================
# Dataset / audio
# =============================================================================


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

        for sample_id, sample in zip(batch["id"], batch["audio"]):
            sample_id = sample_id.replace("/", "_").removesuffix(".wav")
            audio_path = os.path.join(cache_dir, f"{sample_id}.wav")

            audio, sr = load_audio(sample)

            if not os.path.exists(audio_path):
                soundfile.write(audio_path, audio, sr)

            audio_paths.append(audio_path)
            durations.append(len(audio) / sr)

        batch["audio_filepaths"] = audio_paths
        batch["durations"] = durations
        batch["references"] = batch["norm_text"]
        return batch

    return fn


def prepare_dataset(cfg: EvalConfig, cache_dir: str):
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


def collect_dataset(dataset):
    all_data = {"audio_filepaths": [], "durations": [], "references": []}

    for sample in tqdm(iter(dataset), desc="Preparing samples"):
        for k in all_data:
            all_data[k].append(sample[k])

    return all_data


def sort_by_duration(all_data):
    order = sorted(
        range(len(all_data["durations"])),
        key=lambda i: all_data["durations"][i],
        reverse=True,
    )

    for k in all_data:
        all_data[k] = [all_data[k][i] for i in order]

    return all_data


# =============================================================================
# Transcription
# =============================================================================


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def transcribe(model, audio_files: List[str], cfg: EvalConfig):
    transcripts = []
    desc_suffix = ""
    if isinstance(model, NemoAsrModel):
        infer_from_wav_paths = model.infer_from_wav_paths
        desc_suffix = " with exported model"
    elif "canary" in cfg.exported_dir:
        pnc = "nopnc" if "v2" not in cfg.exported_dir else "pnc"
        infer_from_wav_paths = partial(
            model.transcribe,
            batch_size=cfg.batch_size,
            verbose=False,
            pnc=pnc,
            num_workers=0,
        )
        desc_suffix = "with nemo canary"
    else:
        infer_from_wav_paths = partial(
            model.transcribe,
            batch_size=cfg.batch_size,
            verbose=False,
            num_workers=0,
        )
        desc_suffix = "with nemo"
    for batch in tqdm(
        chunks(audio_files, cfg.batch_size),
        desc="Transcribing " + desc_suffix,
        total=len(audio_files) // cfg.batch_size,
    ):
        transcripts.extend(infer_from_wav_paths(batch))
    return transcripts


def measure_transcription_time(
    model,
    audio_files: List[str],
    cfg: EvalConfig,
):
    transcriptions = []
    total_time = 0.0

    for i in range(cfg.warmup + 1):
        files = (
            audio_files[: cfg.batch_size * 4]
            if i == 0 and cfg.warmup > 0
            else audio_files
        )

        start = time.time()
        with torch.inference_mode():
            transcriptions = transcribe(model, files, cfg)
        end = time.time()

        if i == cfg.warmup:
            total_time = end - start

    return total_time, transcriptions


# =============================================================================
# Metrics / output
# =============================================================================


def normalize_predictions(transcriptions):
    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]
    return [data_utils.normalizer(t.text) for t in transcriptions]


def write_results(all_data, predictions, cfg: EvalConfig, avg_time: float):
    model_id = cfg.exported_dir
    if not cfg.use_original_model:
        model_id += "_exported_rust_inference"
    return data_utils.write_manifest(
        references=all_data["references"],
        transcriptions=predictions,
        model_id=model_id,
        dataset_path=cfg.dataset_path,
        dataset_name=cfg.dataset,
        split=cfg.split,
        audio_length=all_data["durations"],
        transcription_time=[avg_time] * len(predictions),
    )


def compute_metrics(all_data, predictions, total_time: float):
    wer = 100 * WER_METRIC.compute(
        references=all_data["references"],
        predictions=predictions,
    )

    audio_seconds = sum(all_data["durations"])
    rtfx = audio_seconds / total_time

    return round(wer, 2), round(rtfx, 2), audio_seconds


# =============================================================================
# Public API
# =============================================================================


def run_asr_evaluation(cfg: EvalConfig) -> EvalResult:
    cache_dir = ensure_cache_dir(cfg.dataset, cfg.split)

    device, dtype = setup_device(cfg.device_id)
    model = load_asr_model(
        cfg.exported_dir,
        use_original_model=cfg.use_original_model,
        device=device,
        dtype=dtype,
    )

    if isinstance(model, ASRModel):
        ensure_decoding_strategy(model)

    dataset = prepare_dataset(cfg, cache_dir)
    all_data = sort_by_duration(collect_dataset(dataset))

    total_time, transcriptions = measure_transcription_time(
        model,
        all_data["audio_filepaths"],
        cfg,
    )

    predictions = normalize_predictions(transcriptions)
    avg_time = total_time / len(predictions)

    manifest_path = write_results(all_data, predictions, cfg, avg_time)

    wer, rtfx, audio_seconds = compute_metrics(
        all_data, predictions, total_time
    )

    return EvalResult(
        wer=wer,
        rtfx=rtfx,
        total_audio_seconds=audio_seconds,
        total_transcription_seconds=total_time,
        manifest_path=os.path.abspath(manifest_path),
        num_samples=len(predictions),
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--exported_dir",
        required=True,
        help=(
            "Path to the exported ASR model directory. "
            "This must be a NeMo-Tract-exported model ."
        ),
    )

    p.add_argument(
        "--dataset_path",
        default="esb/datasets",
        help=(
            "Root path or Hugging Face namespace containing the ESB datasets "
            "(e.g. 'hf-audio/esb-datasets-test-only-sorted')."
        ),
    )

    p.add_argument(
        "--dataset",
        required=True,
        help=(
            "Name of the dataset to evaluate. "
            "Examples: ami, earnings22, gigaspeech, librispeech, "
            "spgispeech, tedlium, voxpopuli."
        ),
    )

    p.add_argument(
        "--split",
        default="test",
        help=(
            "Dataset split to evaluate. Examples: test, test.clean, test.other."
        ),
    )

    p.add_argument(
        "--device",
        type=int,
        default=-1,
        help=(
            "CUDA device index to use for evaluation. "
            "Use -1 to run on CPU, or a non-negative integer for a specific GPU."
        ),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "Number of audio samples processed per batch during transcription. "
            "Larger values improve throughput but increase memory usage."
        ),
    )

    p.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples to evaluate. "
            "If unset or negative, the full dataset split is evaluated."
        ),
    )

    p.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help=(
            "Disable dataset streaming. "
            "When set, the full dataset is downloaded locally before evaluation."
        ),
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help=(
            "Number of warmup transcription runs performed before timing. "
            "Warmup runs are not included in runtime measurements."
        ),
    )

    p.set_defaults(streaming=True)
    a = p.parse_args()

    return EvalConfig(
        exported_dir=a.exported_dir,
        dataset_path=a.dataset_path,
        dataset=a.dataset,
        split=a.split,
        device_id=a.device,
        batch_size=a.batch_size,
        max_eval_samples=a.max_eval_samples,
        streaming=a.streaming,
        warmup=a.warmup,
    )


def main():
    cfg = parse_args()
    result = run_asr_evaluation(cfg)

    print("Manifest:", result.manifest_path)
    print("Samples:", result.num_samples)
    print("WER:", result.wer, "%")
    print("RTFX:", result.rtfx)


if __name__ == "__main__":
    main()
