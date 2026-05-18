"""Per-frame inference bench for DeepFilterNet 3 on tract.

Uses ``tract dump --profile --json`` so the timings are inference-only
(no graph-load overhead, no Python wrapper cost). Each entry in tract's
JSON is one op with `secs_per_iter`; summing gives total per-frame
latency. The script also reports the top-k hottest ops so you can see
where time is spent.

Three artifact shapes are compared if all are provided:

- **NNEF variant A**: t2n export with matmul-iFFT (grazder verbatim);
  ships full pipeline (STFT/ERB/NN/iSTFT) in one artifact.
- **NNEF variant B**: t2n export with native `torch.fft.irfft` on
  synthesis; same single-artifact shape as A.
- **Official ONNX (3 components)**: the upstream DeepFilterNet
  `models/DeepFilterNet3_onnx.tar.gz` ships `enc.onnx`,
  `erb_dec.onnx`, `df_dec.onnx` as three frequency-domain components
  (no FFT in graph). This is what DfTract / czoli1976 actually run --
  libDF (Rust) handles STFT/iSTFT/ERB feature extraction outside the
  graph, the per-frame NN cost is the sum of the three tract calls.
  We feed each component via `--input-bundle` after pre-baking concrete
  shapes with `onnxsim` (tract's CLI can't resolve the symbolic dims
  baked into the ONNX: the DfTract Rust binary does it via the API).

DFN3's per-frame budget for real-time at 48 kHz is 480 / 48000 =
**10 ms per frame**. Anything below that is real-time-capable; below 1 ms
is 10x real-time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TORCH_DF_PATH = _HERE / "_torchDF_clone" / "torchDF"
if not _TORCH_DF_PATH.exists():
    raise SystemExit(f"missing {_TORCH_DF_PATH}; run ./bootstrap.sh first")
sys.path.insert(0, str(_TORCH_DF_PATH))


def _patch_torchaudio_audio_meta_data() -> None:
    # pylint: disable=import-outside-toplevel
    import torchaudio

    if "AudioMetaData" in dir(torchaudio):
        return
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(self, *args, **kwargs) -> None:
            pass

    common.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
    backend.common = common  # type: ignore[attr-defined]
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules.setdefault("torchaudio.backend.common", common)


_patch_torchaudio_audio_meta_data()

from torch_to_nnef.inference_target.tract import (  # noqa: E402
    TractCli,
    TractNNEF,
)


def make_input_bundle(out_dir: Path) -> Path:
    """Build an NPZ bundle matching the per-frame NNEF export's 13 inputs.

    Reuses the cached bundle if `inputs.npz` exists: avoids re-importing
    the upstream `df` package (which pins to torch 2.11; the test suite
    may have re-locked the venv to torch 2.9).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "inputs.npz"
    if bundle_path.exists():
        return bundle_path
    # pylint: disable-next=import-outside-toplevel
    from torch_df_streaming_minimal import (
        TorchDFMinimalPipeline,  # noqa: PLC0415
    )

    pipeline = TorchDFMinimalPipeline().eval()
    states = tuple(s.detach().cpu().numpy() for s in pipeline.states)
    state_names = list(pipeline.input_names[1:])
    rng = np.random.default_rng(0)
    input_frame = (
        rng.standard_normal(pipeline.hop_size, dtype=np.float32) * 0.05
    )
    bundle = {"input_frame": input_frame}
    for name, state in zip(state_names, states, strict=True):
        bundle[name] = state
    np.savez(bundle_path, **bundle)
    return bundle_path


# Per-frame shapes for the official 3-component ONNX. nb_erb=32,
# nb_df=96, conv_ch=64, emb_hidden=512, S=1 (one frame).
_OFFICIAL_ONNX_SHAPES = {
    "enc": {
        "feat_erb": [1, 1, 1, 32],
        "feat_spec": [1, 2, 1, 96],
    },
    "erb_dec": {
        "emb": [1, 1, 512],
        "e3": [1, 64, 1, 8],
        "e2": [1, 64, 1, 8],
        "e1": [1, 64, 1, 16],
        "e0": [1, 64, 1, 32],
    },
    "df_dec": {
        "emb": [1, 1, 512],
        "c0": [1, 64, 1, 96],
    },
}


def prepare_official_onnx(
    onnx_dir: Path, work_dir: Path
) -> dict[str, tuple[Path, Path]]:
    """Pre-simplify the 3 official ONNX components + write input bundles.

    The official `DeepFilterNet3_onnx.tar.gz` ONNXs carry symbolic
    dimension parameters (`Relue0_dim_0`, `Relue0_dim_3`, ...) that
    tract's CLI cannot resolve from `--input-bundle` alone: DfTract's
    Rust API sets them on the model before analyse. We work around it
    by running `onnxsim` first with concrete input shapes; that bakes
    every symbolic dim down to the trace-time value. Returns a dict
    `{component: (simplified_onnx_path, input_bundle_path)}`.
    """
    # pylint: disable=import-outside-toplevel
    import onnx  # noqa: PLC0415
    import onnxsim  # noqa: PLC0415

    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[Path, Path]] = {}
    for name, shapes in _OFFICIAL_ONNX_SHAPES.items():
        src = onnx_dir / f"{name}.onnx"
        if not src.exists():
            raise SystemExit(
                f"missing {src}; expected the official 3-component ONNX "
                f"(unpack DeepFilterNet3_onnx.tar.gz into {onnx_dir})"
            )
        model = onnx.load(str(src))
        simplified, ok = onnxsim.simplify(model, overwrite_input_shapes=shapes)
        if not ok:
            raise RuntimeError(f"onnxsim failed on {src}")
        simp_path = work_dir / f"{name}_simplified.onnx"
        onnx.save(simplified, str(simp_path))
        bundle_path = work_dir / f"{name}_inputs.npz"
        np.savez(
            bundle_path,
            **{n: np.zeros(s, dtype=np.float32) for n, s in shapes.items()},
        )
        out[name] = (simp_path, bundle_path)
    return out


def tract_profile_cmd(
    tract_cli: TractCli,
    model_path: Path,
    bundle_path: Path,
    iters: int,
) -> list[str]:
    """`tract <model> ... dump --profile --json --input-from-bundle <npz>`.

    The `--profile` flag runs the graph inside tract's internal bench
    loop (iter count controlled by tract's defaults; we pass `--iters`
    via `dump`'s sister flag if available). Output is per-op
    `secs_per_iter` in JSON, which we sum to get total inference time.
    """
    is_onnx = model_path.suffix == ".onnx"
    cmd: list[str] = [str(tract_cli.tract_path), str(model_path)]
    if not is_onnx:
        cmd += ["--nnef-tract-core", "--nnef-tract-pulse"]
        if tract_cli.version >= "0.20.20":
            cmd += ["--nnef-tract-extra"]
        if tract_cli.version >= "0.22.0":
            cmd += ["--nnef-tract-transformers"]
    cmd += ["-O"]
    cmd += [
        "dump",
        "--profile",
        "--json",
        "--input-from-bundle",
        str(bundle_path),
    ]
    return cmd


def profile_one(
    tract_cli: TractCli,
    model_path: Path,
    bundle_path: Path,
    n_trials: int,
) -> dict:
    """Run tract's profiler `n_trials` times; aggregate medians per op."""
    cmd = tract_profile_cmd(tract_cli, model_path, bundle_path, iters=1)
    per_trial_totals: list[float] = []
    per_op_samples: dict[str, list[float]] = {}
    last_raw: list[dict] = []
    for trial in range(n_trials):
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tract profile failed on {model_path.name} (trial {trial}):\n"
                f"cmd: {' '.join(cmd)}\nstderr:\n{result.stderr[-2000:]}"
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"tract profile JSON parse failed on {model_path.name} "
                f"(trial {trial}): {exc}\nfirst 2KB of stdout:\n"
                f"{result.stdout[:2000]}"
            ) from exc
        nodes = data["nodes"]
        last_raw = nodes
        total = sum(n["secs_per_iter"] for n in nodes)
        per_trial_totals.append(total)
        for n in nodes:
            label = f"{n['op_name']} ({n['node_name']})"
            per_op_samples.setdefault(label, []).append(n["secs_per_iter"])

    op_summary: list[tuple[str, float]] = []
    for label, samples in per_op_samples.items():
        op_summary.append((label, statistics.median(samples)))
    op_summary.sort(key=lambda kv: -kv[1])

    return {
        "median_total_s": statistics.median(per_trial_totals),
        "min_total_s": min(per_trial_totals),
        "p90_total_s": sorted(per_trial_totals)[
            min(int(0.9 * len(per_trial_totals)), len(per_trial_totals) - 1)
        ],
        "trials": n_trials,
        "n_ops": len(last_raw),
        "all_ops": op_summary,
        "top_ops": op_summary[:10],
    }


# torch_streaming_model submodules that match the official ONNX bundle's
# 3 NN components. Every NNEF node whose qualified name starts with one
# of these prefixes is "NN" work; the rest (FFT, windowing, ERB feature
# extract, frame synthesis) is the DSP that libDF performs outside the
# graph in the official deploy.
_NN_PREFIXES = ("enc__", "erb_dec__", "df_dec__")


def _split_nn_dsp(stats: dict) -> tuple[float, float, list, list]:
    """Sum NN-scoped vs DSP-scoped op latencies from a `profile_one` result.

    Returns `(nn_secs, dsp_secs, nn_ops, dsp_ops)` where the *_ops lists
    are `(label, secs)` pairs sorted slowest-first.
    """
    nn_ops: list[tuple[str, float]] = []
    dsp_ops: list[tuple[str, float]] = []
    for label, secs in stats["all_ops"]:
        # label is `OpName (node_name)`: pull `node_name` to match prefix.
        node_name = label.rsplit("(", 1)[-1].rstrip(")")
        if node_name.startswith(_NN_PREFIXES):
            nn_ops.append((label, secs))
        else:
            dsp_ops.append((label, secs))
    return (
        sum(s for _, s in nn_ops),
        sum(s for _, s in dsp_ops),
        nn_ops,
        dsp_ops,
    )


def main() -> None:  # pylint: disable=too-many-statements
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nnef-a", type=Path, default=None, help="variant A NNEF (matmul-iFFT)"
    )
    parser.add_argument(
        "--nnef-b",
        type=Path,
        default=None,
        help="variant B NNEF (torch.fft.irfft)",
    )
    parser.add_argument(
        "--onnx-official-dir",
        type=Path,
        default=None,
        help="Directory containing the upstream `enc.onnx`, `erb_dec.onnx`, "
        "`df_dec.onnx` (from `DeepFilterNet3_onnx.tar.gz`). The bench "
        "pre-simplifies them with onnxsim, profiles each via tract, and "
        "sums to get NN-only per-frame cost (no FFT/ERB DSP, which "
        "lives in libDF in the official deploy).",
    )
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("./bench_work"))
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Show top-k hottest ops per variant (default: 5).",
    )
    args = parser.parse_args()

    if (
        args.nnef_a is None
        and args.nnef_b is None
        and args.onnx_official_dir is None
    ):
        raise SystemExit(
            "specify at least one of --nnef-a / --nnef-b / --onnx-official-dir"
        )

    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.utils import SemanticVersion

    latest = TractNNEF.latest_version()
    tract_version = (
        SemanticVersion.from_str(args.tract_version)
        if args.tract_version
        else latest
    )
    tract_cli = TractCli.download(tract_version)
    print(f"tract: {tract_cli.tract_path} (v{tract_cli.version})")

    nnef_bundle = make_input_bundle(args.work_dir)
    print(f"NNEF bundle: {nnef_bundle} (single audio frame + 12 states)")
    print("per-frame real-time budget: 10.00ms (480 samples @ 48 kHz)")
    print()

    results: dict[str, dict] = {}

    if args.nnef_a is not None and args.nnef_a.exists():
        print(f"profiling NNEF matmul-iFFT: {args.nnef_a.name} ...")
        results["NNEF matmul-iFFT"] = profile_one(
            tract_cli, args.nnef_a, nnef_bundle, args.n_trials
        )

    if args.nnef_b is not None and args.nnef_b.exists():
        print(f"profiling NNEF torch.fft.irfft: {args.nnef_b.name} ...")
        results["NNEF torch.fft.irfft"] = profile_one(
            tract_cli, args.nnef_b, nnef_bundle, args.n_trials
        )

    if args.onnx_official_dir is not None:
        print(
            "preparing official 3-component ONNX (onnxsim + concrete shapes)..."
        )
        prepared = prepare_official_onnx(
            args.onnx_official_dir, args.work_dir / "onnx_simplified"
        )
        components_total_s = 0.0
        components_min_s = 0.0
        components_p90_s = 0.0
        total_ops = 0
        top_ops_combined: list[tuple[str, float]] = []
        for name in ("enc", "erb_dec", "df_dec"):
            onnx_path, onnx_bundle = prepared[name]
            print(f"  profiling {name}.onnx ({onnx_path.name}) ...")
            r = profile_one(tract_cli, onnx_path, onnx_bundle, args.n_trials)
            components_total_s += r["median_total_s"]
            components_min_s += r["min_total_s"]
            components_p90_s += r["p90_total_s"]
            total_ops += r["n_ops"]
            top_ops_combined.extend(
                (f"[{name}] {label}", secs) for label, secs in r["top_ops"]
            )
            print(
                f"    -> {r['median_total_s'] * 1000:.3f}ms median "
                f"({r['n_ops']} ops)"
            )
        top_ops_combined.sort(key=lambda kv: -kv[1])
        results["ONNX official (3-comp, NN-only)"] = {
            "median_total_s": components_total_s,
            "min_total_s": components_min_s,
            "p90_total_s": components_p90_s,
            "trials": args.n_trials,
            "n_ops": total_ops,
            "top_ops": top_ops_combined,
        }

    # Add NN-only rows for NNEF variants (apples-to-apples with the
    # official ONNX 3-comp NN-only number: drop FFT / windowing / ERB
    # feature / synthesis ops that the upstream deploy runs in libDF).
    nn_only_results: dict[str, dict] = {}
    for label, stats in results.items():
        if "all_ops" not in stats:
            continue
        if not label.startswith("NNEF"):
            continue
        nn_secs, dsp_secs, nn_ops, dsp_ops = _split_nn_dsp(stats)
        nn_only_results[f"{label} (NN-only)"] = {
            "median_total_s": nn_secs,
            "dsp_total_s": dsp_secs,
            "n_ops": len(nn_ops),
            "n_dsp_ops": len(dsp_ops),
            "top_ops": nn_ops[:10],
            "dsp_top_ops": dsp_ops[:10],
        }

    print()
    header = (
        f"{'Variant':<40} {'median':>10} {'min':>10} "
        f"{'p90':>10} {'ops':>6} {'RTFx':>7}"
    )
    print(header)
    print("-" * 88)
    for label, stats in results.items():
        median_ms = stats["median_total_s"] * 1000
        min_ms = stats["min_total_s"] * 1000
        p90_ms = stats["p90_total_s"] * 1000
        rtfx = 10.0 / median_ms if median_ms > 0 else float("inf")
        print(
            f"{label:<40} {median_ms:>7.3f}ms {min_ms:>7.3f}ms "
            f"{p90_ms:>7.3f}ms {stats['n_ops']:>6} {rtfx:>6.2f}x"
        )
    for label, stats in nn_only_results.items():
        median_ms = stats["median_total_s"] * 1000
        rtfx = 10.0 / median_ms if median_ms > 0 else float("inf")
        dsp_ms = stats["dsp_total_s"] * 1000
        suffix = f"  (DSP excluded: {dsp_ms:.3f}ms / {stats['n_dsp_ops']} ops)"
        print(
            f"{label:<40} {median_ms:>7.3f}ms {'-':>9} {'-':>9} "
            f"{stats['n_ops']:>6} {rtfx:>6.2f}x{suffix}"
        )

    if args.top_k > 0:
        for label, stats in results.items():
            print()
            print(f"Top {args.top_k} hottest ops for {label}:")
            for op_label, secs in stats["top_ops"][: args.top_k]:
                print(f"  {secs * 1000:>7.3f}ms  {op_label}")

    out_json = args.work_dir / "bench.json"
    combined = dict(results)
    combined.update(nn_only_results)
    out_json.write_text(json.dumps(combined, indent=2))
    print(f"\nfull results: {out_json}")


if __name__ == "__main__":
    main()
