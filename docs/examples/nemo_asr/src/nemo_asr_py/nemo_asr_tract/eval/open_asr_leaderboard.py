"""Evaluate a NeMo ASR model exported with t2n.

Programmatic ESB evaluation runner.

- Reproduces Open ASR Leaderboard evals
- Compares original NeMo vs exported model
- Calls manifest.score_results directly
"""

import argparse
import logging as log
from pathlib import Path
from typing import List, Tuple

from nemo_asr_tract import init_env_logger
from nemo_asr_tract.dataset import DatasetConfig
from nemo_asr_tract.eval.base import (
    EvalConfig,
    load_runner_from_config,
    run_asr_evaluation,
)
from nemo_asr_tract.eval.manifest import score_results
from nemo_asr_tract.nemo_asr import load_config_from_dir

# =============================================================================
# Dataset matrix (faithful to the original shell script)
# =============================================================================

HF_ESB_SLUG = "hf-audio/esb-datasets-test-only-sorted"
ESB_DATASETS: List[Tuple[str, str]] = [
    ("ami", "test"),
    ("earnings22", "test"),
    ("gigaspeech", "test"),
    ("librispeech", "test.clean"),
    ("librispeech", "test.other"),
    ("spgispeech", "test"),
    ("tedlium", "test"),
    ("voxpopuli", "test"),
]


# =============================================================================
# Evaluation runner
# =============================================================================


def run_eval(
    *,
    model_id: str,
    model_runner_class: str,
    exported_dir: str,
    hg_path: str,
    dataset: str,
    split: str,
    device_id: int,
    batch_size: int,
    results_dir: Path,
):
    cfg = EvalConfig(
        model_dir=exported_dir,
        model_runner_class=model_runner_class,
        dataset=DatasetConfig(
            hg_path=hg_path,
            name=dataset,
            split=split,
            batch_size=batch_size,
            max_eval_samples=None,
            streaming=True,
        ),
        device_id=device_id,
        warmup=0,
        output_dir=results_dir,
    )
    runner = load_runner_from_config(cfg)
    tag = runner.name()
    out_dir = results_dir / dataset / split / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {tag} | {dataset}:{split} ===", flush=True)

    result = run_asr_evaluation(cfg, runner=runner)

    # Persist a small summary (debug / CI friendly)
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"runner_name: {tag}\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"split: {split}\n")
        f.write(f"WER: {result.wer}\n")
        f.write(f"RTFX: {result.rtfx}\n")
        f.write(f"samples: {result.num_samples}\n")
        f.write(f"manifest: {result.manifest_path}\n")

    return result


def init_log(verbosity: int):
    _stream_log = log.StreamHandler()
    try:
        # use rich handler if availlable
        # pylint: disable-next=import-outside-toplevel
        from rich.logging import RichHandler

        _stream_log = RichHandler()
    except ImportError:
        # If rich is not installed, fall back to the default stream handler.
        pass

    if verbosity > 2:
        raise ValueError("verbosity level should be between 0 and 2")

    log.basicConfig(
        format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level={
            -1: log.ERROR,
            0: log.INFO,
            1: log.DEBUG,
            2: log.DEBUG,
        }[verbosity],
        handlers=[_stream_log],
    )
    if verbosity > -1:
        init_env_logger(verbosity)
    return log


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exported_dir", required=True)
    parser.add_argument(
        "-d",
        "--dataset",
        default="*",
        help="Datasets to evaluate if '*' all open Leaderboard used.",
    )
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-r", "--results_dir", required=True)
    parser.add_argument(
        "-c",
        "--model_runner_class",
        required=False,
        default=[
            "nemo_asr_tract.eval.runner.ExportedNemoRunner",
            "nemo_asr_tract.eval.runner.NemoRunner",
        ],
        nargs="+",
        help="Model runner classes to use (you can implement your own).",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=0,
        help="Logging verbosity level: -1 (ERROR), 0 (INFO), 1 (DEBUG).",
    )

    args = parser.parse_args()

    init_log(verbosity=args.verbosity)
    results_dir = Path(args.results_dir).absolute()
    results_dir.mkdir(parents=True, exist_ok=True)

    conf = load_config_from_dir(args.exported_dir)

    # -------------------------------------------------------------------------
    # Run evaluations
    # -------------------------------------------------------------------------

    for dataset, split in ESB_DATASETS:
        if args.dataset != "*" and dataset not in args.dataset:
            print(f"\n=== Skipping {dataset}:{split} ===")
            continue
        for model_runner_class in args.model_runner_class:
            run_eval(
                model_id=conf.pretrained_name,
                model_runner_class=model_runner_class,
                exported_dir=args.exported_dir,
                hg_path=HF_ESB_SLUG,
                dataset=dataset,
                split=split,
                device_id=args.device,
                batch_size=args.batch_size,
                results_dir=results_dir,
            )

    # -------------------------------------------------------------------------
    # Final scoring (direct function call, no subprocess)
    # -------------------------------------------------------------------------

    print("\n=== Scoring results ===", flush=True)
    score_results(
        str(results_dir),
        conf.pretrained_name,
    )


if __name__ == "__main__":
    main()
