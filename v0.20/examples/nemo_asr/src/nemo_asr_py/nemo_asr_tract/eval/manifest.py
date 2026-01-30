"""Reading/writing manifest files and scoring ASR from them.

based on:
https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/eval_utils.py
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import evaluate
from nemo_asr_tract.utils import clean_name


def read_manifest(manifest_path: str):
    """Reads a manifest file (jsonl format) returns a list of dict samples."""
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 0:
                datum = json.loads(line)
                data.append(datum)
    return data


def write_manifest(
    audio_filepaths: list,
    references: list,
    transcriptions: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    audio_length: list = None,
    transcription_time: list = None,
    basedir: Union[str, Path] = "./results_manifests",
):
    """Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        audio_filepaths: List of audio file paths.
        references: Ground truth reference texts.
        transcriptions: Model predicted transcriptions.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.
        audio_length: Length of each audio sample in seconds.
        transcription_time: Transcription time of each sample in seconds.
        basedir: Base directory to save the manifest file.

    Returns:
        Path to the manifest file.
    """
    model_id = model_id.replace("/", "-")
    dataset_path = dataset_path.replace("/", "-")
    dataset_name = dataset_name.replace("/", "-")

    if len(references) != len(transcriptions):
        raise ValueError(
            f"The number of samples in `references` ({len(references)}) "
            f"must match `transcriptions` ({len(transcriptions)})."
        )

    if audio_length is not None and len(audio_length) != len(references):
        raise ValueError(
            f"The number of samples in `audio_length` ({len(audio_length)}) "
            f"must match `references` ({len(references)})."
        )
    if transcription_time is not None and len(transcription_time) != len(
        references
    ):
        raise ValueError(
            "The number of samples in `transcription_time` "
            f"({len(transcription_time)}) "
            f"must match `references` ({len(references)})."
        )
    if len(audio_filepaths) != len(references):
        raise ValueError(
            "The number of samples in `audio_filepaths` "
            f"({len(audio_filepaths)}) "
            f"must match `references` ({len(references)})."
        )

    audio_length = (
        audio_length if audio_length is not None else len(references) * [None]
    )
    transcription_time = (
        transcription_time
        if transcription_time is not None
        else len(references) * [None]
    )

    basedir = Path(basedir)
    if not basedir.exists():
        os.makedirs(basedir)

    manifest_path = basedir / (
        f"MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}.jsonl"
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (
            audio_filepath,
            text,
            transcript,
            audio_length_item,
            transcription_time_item,
        ) in enumerate(
            zip(
                audio_filepaths,
                references,
                transcriptions,
                audio_length,
                transcription_time,
            )
        ):
            datum = {
                # dummy value for Speech Data Processor
                "id": idx,
                "audio_filepath": Path(audio_filepath).name,
                "duration": audio_length_item,
                "time": transcription_time_item,
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path


def _try_import_rich():
    try:
        from rich.console import Console
        from rich.table import Table

        return Console(), Table
    except Exception:
        return None, None


def trim_model_id(model_id: str) -> str:
    """Trim model id to fit in table display."""
    if "/" in model_id:
        model_id = model_id.split("/")[-1]
    return model_id


def trim_dataset_id(dataset_id: str) -> str:
    """Trim dataset id to fit in table display."""
    rm_dataset_id = "hf-audio-esb-datasets-test-only-sorted_"
    return dataset_id.replace(rm_dataset_id, "esb/")


def display_results_plain(results: dict):
    print("*" * 80)
    print("Results per dataset:")
    print("*" * 80)

    for k, v in results.items():
        metrics = f"{trim_dataset_id(k)}: WER = {v['wer']:0.2f} %"
        if v["rtfx"] is not None:
            metrics += f", RTFx = {v['rtfx']:0.2f}"
        print(metrics)


def display_composite_plain(
    composite_wer,
    composite_audio_length,
    composite_inference_time,
    count_entries,
):
    print()
    print("*" * 80)
    print("Composite Results:")
    print("*" * 80)

    for k, v in composite_wer.items():
        wer = v / count_entries[k]
        print(f"{trim_model_id(k)}: WER = {wer:0.2f} %")

    for k in composite_audio_length:
        if composite_audio_length[k] is not None:
            rtfx = composite_audio_length[k] / composite_inference_time[k]
            print(f"{trim_model_id(k)}: RTFx = {rtfx:0.2f}")

    print("*" * 80)


def display_results_rich(results: dict, console, Table):
    table = Table(title="Results per Dataset", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Dataset")
    table.add_column("WER (%)", justify="right")
    table.add_column("RTFx", justify="right")

    for k, v in results.items():
        model, dataset = [x.strip() for x in k.split("|", 1)]
        table.add_row(
            trim_model_id(model),
            trim_dataset_id(dataset),
            f"{v['wer']:.2f}",
            f"{v['rtfx']:.2f}" if v["rtfx"] is not None else "—",
        )

    console.print(table)


def display_composite_rich(
    composite_wer,
    composite_audio_length,
    composite_inference_time,
    count_entries,
    console,
    Table,
):
    table = Table(title="Composite Results", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("WER (%)", justify="right")
    table.add_column("RTFx", justify="right")

    for k, v in composite_wer.items():
        wer = v / count_entries[k]
        if composite_audio_length[k] is not None:
            rtfx = composite_audio_length[k] / composite_inference_time[k]
            rtfx_str = f"{rtfx:.2f}"
        else:
            rtfx_str = "—"

        table.add_row(trim_model_id(k), f"{wer:.2f}", rtfx_str)

    console.print(table)


def compute_composite(results: dict):
    composite_wer = defaultdict(float)
    composite_audio_length = defaultdict(float)
    composite_inference_time = defaultdict(float)
    count_entries = defaultdict(int)

    for k, v in results.items():
        key = k.split("|")[0].strip()
        composite_wer[key] += v["wer"]

        if v["rtfx"] is not None:
            composite_audio_length[key] += v["audio_length"]
            composite_inference_time[key] += v["inference_time"]
        else:
            composite_audio_length[key] = composite_inference_time[key] = None

        count_entries[key] += 1

    return (
        composite_wer,
        composite_audio_length,
        composite_inference_time,
        count_entries,
    )


def score_results(directory: str, model_id: str = None):
    """Scores all result files in a directory and returns a composite score.

    over all evaluated datasets

    Args:
        directory: Path to the result directory,
            containing one or more jsonl files.
        model_id: Optional, model name to filter out
            result files based on model name.

    Returns:
        Composite score over all evaluated datasets and
        a dictionary of all results.
    """
    # Strip trailing slash
    if directory.endswith(os.pathsep):
        directory = directory[:-1]

    result_files = list(
        sorted(glob.glob(f"{directory}/**/*.jsonl", recursive=True))
    )

    if model_id:
        print("Filtering models by id:", model_id)
        model_id = clean_name(model_id)
        result_files = [fp for fp in result_files if model_id in fp]

    if not result_files:
        raise ValueError(f"No result files found in {directory}")

    def parse_filepath(fp: str):
        model_index = fp.find("MODEL_")
        fp = fp[model_index:]
        ds_index = fp.find("DATASET_")
        model_id = fp[:ds_index].replace("MODEL_", "").rstrip("_")
        author_index = model_id.find("-")
        model_id = model_id[:author_index] + "/" + model_id[author_index + 1 :]
        dataset_id = fp[ds_index:].replace("DATASET_", "").rstrip(".jsonl")
        return model_id, dataset_id

    results = {}
    wer_metric = evaluate.load("wer")

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        references = [d["text"] for d in manifest]
        predictions = [d["pred_text"] for d in manifest]

        time = [d["time"] for d in manifest]
        duration = [d["duration"] for d in manifest]
        compute_rtfx = all(time) and all(duration)

        wer = round(
            100
            * wer_metric.compute(
                references=references,
                predictions=predictions,
            ),
            2,
        )

        if compute_rtfx:
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(audio_length / inference_time, 4)
        else:
            audio_length = inference_time = rtfx = None

        results[f"{model_id_of_file} | {dataset_id}"] = {
            "wer": wer,
            "audio_length": audio_length,
            "inference_time": inference_time,
            "rtfx": rtfx,
        }

    console, Table = _try_import_rich()

    if console:
        display_results_rich(results, console, Table)
    else:
        display_results_plain(results)

    (
        composite_wer,
        composite_audio_length,
        composite_inference_time,
        count_entries,
    ) = compute_composite(results)

    if console:
        display_composite_rich(
            composite_wer,
            composite_audio_length,
            composite_inference_time,
            count_entries,
            console,
            Table,
        )
    else:
        display_composite_plain(
            composite_wer,
            composite_audio_length,
            composite_inference_time,
            count_entries,
        )

    return composite_wer, results


def parser_args():
    parser = argparse.ArgumentParser(
        description="Score ASR results stored in JSONL manifest files."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the result directory, containing one or more "
        "jsonl files.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional, model name to filter out result files "
        "based on model name.",
    )

    return parser.parse_args()


def main():
    args = parser_args()
    score_results(args.directory, args.model_id)


if __name__ == "__main__":
    main()
