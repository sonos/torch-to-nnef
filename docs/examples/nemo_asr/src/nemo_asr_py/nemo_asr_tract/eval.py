"""Evaluate a NeMo ASR model exported with t2n.

Programmatic ESB evaluation runner.

- Reproduces Open ASR Leaderboard evals
- Compares original NeMo vs exported model
- Calls eval_utils.score_results directly
"""

from pathlib import Path
from typing import List, Tuple
import argparse
import logging as log


from nemo_asr_tract import init_env_logger
from nemo_asr_tract.nemo_asr import load_config_from_dir
from nemo_asr_tract.open_asr_leaderboard_eval import (
    EvalConfig,
    run_asr_evaluation,
)

from nemo_asr_tract.normalizer import eval_utils


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
    exported_dir: str,
    dataset_path: str,
    dataset: str,
    split: str,
    device_id: int,
    batch_size: int,
    results_dir: Path,
    use_original_model: bool,
):
    tag = "original" if use_original_model else "exported"
    out_dir = results_dir / tag / dataset / split
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(
        exported_dir=exported_dir,
        use_original_model=use_original_model,
        dataset_path=dataset_path,
        dataset=dataset,
        split=split,
        device_id=device_id,
        batch_size=batch_size,
        max_eval_samples=None,
        streaming=True,
        warmup=0,
    )

    print(f"\n=== {tag.upper()} | {dataset}:{split} ===", flush=True)

    result = run_asr_evaluation(cfg)

    # Persist a small summary (debug / CI friendly)
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"variant: {tag}\n")
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
        "-s",
        "--skip_dataset",
        nargs="+",
        default=[],
        help="Datasets to skip during evaluation.",
    )
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-r", "--results_dir", required=True)
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
        if dataset in args.skip_dataset:
            print(f"\n=== Skipping {dataset}:{split} ===")
            continue
        for use_original_model in (False, True):
            run_eval(
                model_id=conf.pretrained_name,
                exported_dir=args.exported_dir,
                dataset_path=HF_ESB_SLUG,
                dataset=dataset,
                split=split,
                device_id=args.device,
                batch_size=args.batch_size,
                results_dir=results_dir,
                use_original_model=use_original_model,
            )

    # -------------------------------------------------------------------------
    # Final scoring (direct function call, no subprocess)
    # -------------------------------------------------------------------------

    print("\n=== Scoring results ===", flush=True)
    eval_utils.score_results(
        str(results_dir),
        args.model_id,
    )


if __name__ == "__main__":
    main()
