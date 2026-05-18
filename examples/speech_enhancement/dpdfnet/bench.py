"""Per-frame inference bench for DPDFNet-2 on tract.

Profiles the streaming NNEF artifact via `tract dump --profile --json`
and partitions the per-op latency by node-name prefix
(`inner__enc__`, `inner__erb_dec__`, `inner__df_dec__` = NN; everything
else = DSP) so we can compare apples-to-apples against the official
ONNX bundle (which is NN-only: libDF handles STFT/iSTFT externally).

Per-frame budget at 16 kHz with hop=160 is **10 ms** (160 / 16000).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

from torch_to_nnef.inference_target.tract import TractCli, TractNNEF
from torch_to_nnef.utils import SemanticVersion

HERE = Path(__file__).resolve().parent
NN_PREFIXES = ("inner__enc__", "inner__erb_dec__", "inner__df_dec__")


def load_manifest(nnef_path: Path) -> dict:
    """Read the sidecar JSON manifest written by `export.py`.

    The manifest carries the per-variant audio params (sample_rate,
    n_fft, hop_size, state_size) so this script doesn't have to
    hard-code shapes per checkpoint. Looks for a `.json` next to
    `nnef_path` with the multi-suffix stem stripped (e.g.
    `dpdfnet2.nnef.tgz` -> `dpdfnet2.json`), then for plain stem
    fallbacks.
    """
    stem = nnef_path.name
    for suffix in (".tgz", ".nnef", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    candidates = [
        nnef_path.parent / f"{stem}.json",
        nnef_path.parent / (nnef_path.name + ".json"),
        nnef_path.with_suffix(".json"),
    ]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    raise SystemExit(
        f"missing manifest next to {nnef_path}; expected one of {candidates}"
    )


def make_input_bundle(out_dir: Path, manifest: dict) -> Path:
    """Build an NPZ bundle matching the per-frame NNEF export's 4 inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"{manifest['variant']}_inputs.npz"
    if bundle_path.exists():
        return bundle_path
    rng = np.random.default_rng(0)
    hop_size = manifest["hop_size"]
    n_fft = manifest["n_fft"]
    np.savez(
        bundle_path,
        audio_frame=(rng.standard_normal(hop_size, dtype=np.float32) * 0.05),
        stft_buf=np.zeros(n_fft, dtype=np.float32),
        nn_state=np.zeros(manifest["state_size"], dtype=np.float32),
        ola_buf=np.zeros(n_fft, dtype=np.float32),
    )
    return bundle_path


def tract_profile_cmd(
    tract_cli: TractCli, model_path: Path, bundle_path: Path
) -> list[str]:
    cmd: list[str] = [str(tract_cli.tract_path), str(model_path)]
    cmd += ["--nnef-tract-core", "--nnef-tract-pulse"]
    if tract_cli.version >= "0.20.20":
        cmd += ["--nnef-tract-extra"]
    if tract_cli.version >= "0.22.0":
        cmd += ["--nnef-tract-transformers"]
    cmd += [
        "-O",
        "dump",
        "--profile",
        "--json",
        "--input-from-bundle",
        str(bundle_path),
    ]
    return cmd


def profile_one(
    tract_cli: TractCli, model_path: Path, bundle_path: Path, n_trials: int
) -> dict:
    cmd = tract_profile_cmd(tract_cli, model_path, bundle_path)
    per_trial_totals: list[float] = []
    per_op_samples: dict[str, list[float]] = {}
    last_n_ops = 0
    for trial in range(n_trials):
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tract profile failed on {model_path.name} (trial {trial})"
                f":\nstderr:\n{result.stderr[-2000:]}"
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"tract profile JSON parse failed: {exc}\n"
                f"first 2KB of stdout:\n{result.stdout[:2000]}"
            ) from exc
        nodes = data["nodes"]
        last_n_ops = len(nodes)
        per_trial_totals.append(sum(n["secs_per_iter"] for n in nodes))
        for n in nodes:
            label = f"{n['op_name']} ({n['node_name']})"
            per_op_samples.setdefault(label, []).append(n["secs_per_iter"])

    op_summary = sorted(
        ((lbl, statistics.median(s)) for lbl, s in per_op_samples.items()),
        key=lambda kv: -kv[1],
    )
    return {
        "median_total_s": statistics.median(per_trial_totals),
        "min_total_s": min(per_trial_totals),
        "p90_total_s": sorted(per_trial_totals)[
            min(int(0.9 * len(per_trial_totals)), len(per_trial_totals) - 1)
        ],
        "n_ops": last_n_ops,
        "all_ops": op_summary,
        "top_ops": op_summary[:10],
    }


def split_nn_dsp(
    stats: dict,
) -> tuple[float, float, list, list]:
    """Sum NN- vs DSP-scoped op latencies by node-name prefix."""
    nn_ops: list[tuple[str, float]] = []
    dsp_ops: list[tuple[str, float]] = []
    for label, secs in stats["all_ops"]:
        node_name = label.rsplit("(", 1)[-1].rstrip(")")
        if node_name.startswith(NN_PREFIXES):
            nn_ops.append((label, secs))
        else:
            dsp_ops.append((label, secs))
    return (
        sum(s for _, s in nn_ops),
        sum(s for _, s in dsp_ops),
        nn_ops,
        dsp_ops,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnef", type=Path, default=HERE / "dpdfnet2.nnef.tgz")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument("--work-dir", type=Path, default=HERE / "bench_work")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    latest = TractNNEF.latest_version()
    tract_version = (
        SemanticVersion.from_str(args.tract_version)
        if args.tract_version
        else latest
    )
    tract_cli = TractCli.download(tract_version)
    print(f"tract: {tract_cli.tract_path} (v{tract_cli.version})")

    manifest = load_manifest(args.nnef)
    bundle = make_input_bundle(args.work_dir, manifest)
    budget_ms = 1000.0 * manifest["hop_size"] / manifest["sample_rate"]
    print(
        f"variant {manifest['variant']}: {manifest['sample_rate']} Hz, "
        f"hop={manifest['hop_size']}, n_fft={manifest['n_fft']}, "
        f"state_size={manifest['state_size']}"
    )
    print(
        f"per-frame budget: {budget_ms:.2f} ms "
        f"({manifest['hop_size']} samples @ "
        f"{manifest['sample_rate'] // 1000} kHz)"
    )
    print(f"profiling {args.nnef.name} ...")
    stats = profile_one(tract_cli, args.nnef, bundle, args.n_trials)
    nn_secs, dsp_secs, nn_ops, dsp_ops = split_nn_dsp(stats)

    median_ms = stats["median_total_s"] * 1000
    min_ms = stats["min_total_s"] * 1000
    p90_ms = stats["p90_total_s"] * 1000
    rtfx = budget_ms / median_ms if median_ms > 0 else float("inf")
    nn_ms = nn_secs * 1000
    dsp_ms = dsp_secs * 1000
    print()
    print(
        f"  median total : {median_ms:7.3f} ms  "
        f"({stats['n_ops']} ops, {rtfx:.2f}x real-time)"
    )
    print(f"  min          : {min_ms:7.3f} ms")
    print(f"  p90          : {p90_ms:7.3f} ms")
    print(f"  NN-only      : {nn_ms:7.3f} ms  ({len(nn_ops)} ops)")
    print(f"  DSP-only     : {dsp_ms:7.3f} ms  ({len(dsp_ops)} ops)")

    if args.top_k > 0:
        print()
        print(f"Top {args.top_k} hottest NN ops:")
        for label, secs in nn_ops[: args.top_k]:
            print(f"  {secs * 1000:>7.3f} ms  {label}")
        print()
        print(f"Top {args.top_k} hottest DSP ops:")
        for label, secs in dsp_ops[: args.top_k]:
            print(f"  {secs * 1000:>7.3f} ms  {label}")

    out = args.work_dir / "bench.json"
    out.write_text(
        json.dumps(
            {
                "median_total_s": stats["median_total_s"],
                "min_total_s": stats["min_total_s"],
                "p90_total_s": stats["p90_total_s"],
                "n_ops": stats["n_ops"],
                "nn_total_s": nn_secs,
                "dsp_total_s": dsp_secs,
                "n_nn_ops": len(nn_ops),
                "n_dsp_ops": len(dsp_ops),
                "top_ops": stats["top_ops"],
            },
            indent=2,
        )
    )
    print(f"\nfull results: {out}")


if __name__ == "__main__":
    main()
