"""'Generic' evaluation of ASR models.

used to evaluated  NVIDIA NeMo on a specified dataset split
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
import logging
import os
from pathlib import Path
from typing import Optional

import evaluate
from nemo_asr_tract.dataset import (
    AUDIO_FILEPATHS_KEY,
    DURATION_KEY,
    REFERENCES_KEY,
    DatasetConfig,
    collect_dataset,
    ensure_cache_dir,
    prepare_dataset,
    sort_by_duration,
)
from nemo_asr_tract.eval.conf import EvalConfig, EvalResult
from nemo_asr_tract.eval.manifest import write_manifest
from nemo_asr_tract.eval.runner import (
    AsrRunner,
    load_runner_from_config,
    measure_transcription_time,
)
from nemo_asr_tract.normalizer import data_utils

__all__ = ["run_asr_evaluation"]


WER_METRIC = evaluate.load("wer")


def normalize_predictions(transcriptions):
    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]
    return [data_utils.normalizer(t.text) for t in transcriptions]


def write_results(
    all_data, predictions, cfg: EvalConfig, model_id: str, avg_time: float
):
    out_dir = (
        Path(cfg.output_dir) / cfg.dataset.name / cfg.dataset.split / model_id
    )

    return write_manifest(
        audio_filepaths=all_data[AUDIO_FILEPATHS_KEY],
        references=all_data[REFERENCES_KEY],
        transcriptions=predictions,
        model_id=model_id,
        dataset_path=cfg.dataset.hg_path,
        dataset_name=cfg.dataset.name,
        split=cfg.dataset.split,
        audio_length=all_data[DURATION_KEY],
        transcription_time=[avg_time] * len(predictions),
        basedir=out_dir,
    )


def compute_metrics(all_data, predictions, total_time: float):
    wer = 100 * WER_METRIC.compute(
        references=all_data[REFERENCES_KEY],
        predictions=predictions,
    )

    audio_seconds = sum(all_data[DURATION_KEY])
    rtfx = audio_seconds / total_time

    return round(wer, 2), round(rtfx, 2), audio_seconds


# =============================================================================
# Public API
# =============================================================================


def run_asr_evaluation(
    cfg: EvalConfig, runner: Optional[AsrRunner] = None
) -> EvalResult:
    cache_dir = ensure_cache_dir(cfg.dataset.name, cfg.dataset.split)
    dataset = prepare_dataset(cfg.dataset, cache_dir)
    all_data = sort_by_duration(collect_dataset(dataset))

    runner = runner or load_runner_from_config(cfg)

    logging.info(
        "Transcribing %d audio files of '%s'",
        len(all_data[AUDIO_FILEPATHS_KEY]),
        cfg.dataset.full_name,
    )
    total_time, transcriptions = measure_transcription_time(
        runner,
        all_data[AUDIO_FILEPATHS_KEY],
        cfg,
    )
    logging.info("Transcribed all '%s'", cfg.dataset.full_name)

    predictions = normalize_predictions(transcriptions)
    avg_time = total_time / len(predictions)

    model_id = runner.name()
    manifest_path = write_results(
        all_data, predictions, cfg, model_id=model_id, avg_time=avg_time
    )
    wer, rtfx, audio_seconds = compute_metrics(
        all_data, predictions, total_time
    )

    logging.info(
        "Evaluation on %s completed: WER=%s %%, RTFx=%s",
        cfg.dataset.full_name,
        wer,
        rtfx,
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
        "--model_runner_class",
        required=True,
        default="nemo_asr_tract.eval.runner.ExportedNemoRunner",
        help="Name of the model runner to use.",
    )

    p.add_argument(
        "--hg_path",
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
            "CUDA/MPS device index to use for evaluation. "
            "Use -1 to run on CPU, or a non-negative integer "
            "for a specific GPU."
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
            "When set, the full dataset is downloaded "
            "locally before evaluation."
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

    p.add_argument(
        "--output_dir",
        type=str,
        default="asr_evaluation_results",
        help=(
            "Directory where evaluation results and manifests will be saved. "
            "Defaults to 'asr_evaluation_results'."
        ),
    )

    p.set_defaults(streaming=True)
    a = p.parse_args()

    return EvalConfig(
        model_dir=a.exported_dir,
        model_runner_class=a.model_runner_class,
        dataset=DatasetConfig(
            hg_path=a.hg_path,
            name=a.dataset,
            split=a.split,
            batch_size=a.batch_size,
            max_eval_samples=a.max_eval_samples,
            streaming=a.streaming,
        ),
        device_id=a.device,
        warmup=a.warmup,
        output_dir=a.output_dir,
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
