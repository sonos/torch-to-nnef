import argparse
import glob
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import evaluate
from rich.console import Console
from rich.text import Text


def wer_alignment(
    ref_tokens: List[str],
    hyp_tokens: List[str],
) -> List[Tuple[str, str | None, str | None]]:
    """Returns a list of edit operations.

    (op, ref_token, hyp_token)

    op ∈ {"ok", "sub", "del", "ins"}
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "ok"
            else:
                choices = {
                    "sub": dp[i - 1][j - 1] + 1,
                    "del": dp[i - 1][j] + 1,
                    "ins": dp[i][j - 1] + 1,
                }
                back[i][j], dp[i][j] = min(choices.items(), key=lambda x: x[1])

    # Backtrace
    i, j = n, m
    aligned = []

    while i > 0 or j > 0:
        op = back[i][j]
        if op == "ok":
            aligned.append(("ok", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif op == "sub":
            aligned.append(("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            aligned.append(("del", ref_tokens[i - 1], None))
            i -= 1
        elif op == "ins":
            aligned.append(("ins", None, hyp_tokens[j - 1]))
            j -= 1

    return aligned[::-1]


def utterance_key(row: dict, fallback_index: int):
    """Build a unique utterance identifier w/ audio path when available."""
    uid = row.get("id", fallback_index)
    audio = (
        row.get("audio")
        or row.get("audio_filepath")
        or row.get("wav")
        or row.get("path")
    )

    if audio:
        return f"{uid} | {audio}"
    return str(uid)


def render_alignment(ref: str, hyp: str) -> Text:
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    aligned = wer_alignment(ref_toks, hyp_toks)

    text = Text()

    for op, ref_tok, hyp_tok in aligned:
        if op == "ok":
            text.append(hyp_tok + " ")
        elif op == "sub":
            text.append(hyp_tok + " ", style="yellow bold")
        elif op == "del":
            text.append(f"[{ref_tok}] ", style="red bold")
        elif op == "ins":
            text.append(hyp_tok + " ", style="green bold")

    return text


def read_manifest(fp: str) -> List[dict]:
    with open(fp, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_filepath(fp: str) -> Tuple[str, str]:
    """Extract (model_id, dataset_id) from filepath.

    Mirrors logic from score_results()
    """
    model_index = fp.find("MODEL_")
    fp = fp[model_index:]
    ds_index = fp.find("DATASET_")

    model_id = fp[:ds_index].replace("MODEL_", "").rstrip("_")
    author_index = model_id.find("-")
    if author_index != -1:
        model_id = model_id[:author_index] + "/" + model_id[author_index + 1 :]

    dataset_id = fp[ds_index:].replace("DATASET_", "").rstrip(".jsonl")
    return model_id, dataset_id


def build_dataset_index(
    result_files: List[str],
    model_filter: str | None,
) -> Dict[str, Dict[str, List[dict]]]:
    """dataset_id -> model_id -> manifest."""
    datasets = defaultdict(dict)

    for fp in result_files:
        model_id, dataset_id = parse_filepath(fp)

        if model_filter and model_filter not in model_id:
            continue

        datasets[dataset_id][model_id] = read_manifest(fp)

    return datasets


def compute_wer(ref: str, hyp: str, wer_metric) -> float:
    return round(
        100 * wer_metric.compute(references=[ref], predictions=[hyp]), 2
    )


def compute_ranked_diffs(
    datasets: dict,
    wer_metric,
    *,
    dataset_filter: str | None = None,
    min_wer_delta: float = 0.0,
    show_all: bool = False,
) -> list[dict]:
    ranked = []

    for dataset_id, models in datasets.items():
        if dataset_filter and dataset_id != dataset_filter:
            continue
        if len(models) < 2:
            continue

        aligned = {}

        for model_id, manifest in models.items():
            for idx, row in enumerate(manifest):
                uid = utterance_key(row, idx)
                aligned.setdefault(uid, {})["ref"] = row["text"]
                aligned[uid][model_id] = row["pred_text"]

        for uid, entry in aligned.items():
            ref = entry.get("ref")
            preds = {k: v for k, v in entry.items() if k != "ref"}

            if len(preds) < 2:
                continue
            if not show_all and len(set(preds.values())) == 1:
                continue

            wers = {
                m: 100 * wer_metric.compute(references=[ref], predictions=[p])
                for m, p in preds.items()
            }

            wer_gap = max(wers.values()) - min(wers.values())
            if wer_gap < min_wer_delta:
                continue

            ranked.append(
                {
                    "dataset": dataset_id,
                    "utterance_id": uid,
                    "ref": ref,
                    "preds": preds,
                    "wers": wers,
                    "wer_gap": wer_gap,
                }
            )

    return sorted(ranked, key=lambda x: x["wer_gap"], reverse=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank prediction discrepancies across runners "
        "by absolute WER gap"
    )

    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root directory containing MODEL_*/DATASET_*.jsonl files",
    )

    parser.add_argument(
        "--dataset",
        help="Only compare a specific dataset id",
    )

    parser.add_argument(
        "--model-filter",
        help="Substring filter applied to model ids",
    )

    parser.add_argument(
        "--min-wer-delta",
        type=float,
        default=0.0,
        help="Minimum absolute WER difference to keep an item",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="Maximum number of ranked discrepancies to display",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Include identical predictions",
    )

    return parser


def main():
    args = build_argparser().parse_args()
    result_files = sorted(
        glob.glob(f"{args.results_dir}/**/*.jsonl", recursive=True)
    )

    datasets = build_dataset_index(result_files, args.model_filter)
    wer_metric = evaluate.load("wer")

    ranked = compute_ranked_diffs(
        datasets,
        wer_metric,
        dataset_filter=args.dataset,
        min_wer_delta=args.min_wer_delta,
        show_all=args.show_all,
    )

    console = Console()

    for rank, item in enumerate(ranked[: args.max_items], 1):
        console.rule(f"RANK #{rank} | WER GAP {item['wer_gap']:.2f}%")
        console.print(f"[bold]Dataset:[/] {item['dataset']}")
        console.print(f"[bold]Utterance ID:[/] {item['utterance_id']}")
        console.print(f"[bold]REF:[/] {item['ref']}")

        for model, pred in sorted(item["preds"].items()):
            console.print(f"\n[cyan]{model}[/]  WER={item['wers'][model]:.2f}%")
            console.print(render_alignment(item["ref"], pred))


if __name__ == "__main__":
    main()
