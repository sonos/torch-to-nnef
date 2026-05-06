"""NaN/Inf-aware comparator for hypothesis-driven primitive tests.

The default tract IO check (`--assert-output-bundle`) does not expose an
`equal_nan` flag, so when both PyTorch and tract produce NaN at the same
position the byte-level comparison fails. Hypothesis routinely generates
inputs that yield NaN, so we bypass tract's strict assert and compare NPZs in
Python with `np.testing.assert_allclose(equal_nan=True)`.

Flow:
  1. Clone the inference target with `check_io=False` so
     `export_model_to_nnef` skips its internal assert.
  2. Dump reference inputs + outputs via `build_io` (PyTorch side).
  3. Run tract with `run --save-outputs-npz` to capture the runtime outputs.
  4. Load both NPZs and compare per output, with dtype-aware tolerance.

Tract is the only supported target: the Khronos reference interpreter has
an embryonic op coverage that does not span the breadth of the proptest
spec catalog.
"""

import subprocess
import tempfile
import typing as T
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    build_io,
)
from torch_to_nnef.log import log

from .dtypes import is_float_dtype, lookup_tol


class ProptestComparatorError(AssertionError):
    """Raised when tract output diverges from the PyTorch reference."""


def _make_no_check_target(target: TractNNEF) -> TractNNEF:
    """Return a copy of `target` with `check_io=False`.

    We disable the built-in `post_export` assert so the proptest comparator
    owns the comparison.
    """
    twin = deepcopy(target)
    twin.check_io = False
    return twin


def _build_tract_run_cmd(
    target: TractNNEF,
    nnef_path: Path,
    inputs_npz: Path,
    outputs_actual_npz: Path,
) -> T.List[str]:
    """Build the tract CLI invocation that dumps actual outputs to NPZ.

    Mirrors the flag layout used by `TractCli.assert_io_cmd_str` but
    substitutes `--save-outputs-npz` for `--assert-output-bundle`.
    Both 0.21.15 and 0.22.1 expose `--save-outputs-npz`.
    """
    extra: T.List[str] = []
    if target.version >= "0.20.20":
        extra.append("--nnef-tract-extra")
    if target.version >= "0.22.0":
        extra.append("--nnef-tract-transformers")
    cmd: T.List[str] = (
        [
            str(target.tract_cli.tract_path),
            str(nnef_path),
            "--nnef-tract-core",
            "--nnef-tract-pulse",
        ]
        + extra
        + [
            "-O",
            "run",
            "--input-from-bundle",
            str(inputs_npz),
            "--save-outputs-npz",
            str(outputs_actual_npz),
            "--allow-float-casts",
        ]
    )
    return cmd


def _run_tract(cmd: T.List[str]) -> None:
    """Run tract and surface stderr if it fails."""
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf8", errors="replace")
        cmd_str = " ".join(cmd)
        raise ProptestComparatorError(
            f"tract CLI failed (rc={proc.returncode})\n"
            f"cmd: {cmd_str}\n"
            f"stderr:\n{stderr}"
        )


def _torch_dtype_from_numpy(np_dtype: np.dtype) -> torch.dtype:
    """Best-effort numpy -> torch dtype mapping for tolerance lookup."""
    mapping = {
        np.dtype("float32"): torch.float32,
        np.dtype("float16"): torch.float16,
        np.dtype("float64"): torch.float64,
        np.dtype("int64"): torch.int64,
        np.dtype("int32"): torch.int32,
        np.dtype("int16"): torch.int16,
        np.dtype("int8"): torch.int8,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool_"): torch.bool,
    }
    return mapping[np.dtype(np_dtype)]


def _compare_npz(
    reference_npz: Path,
    actual_npz: Path,
    output_names: T.Sequence[str],
    tolerance: TractCheckTolerance,
    input_dtypes: T.Sequence[torch.dtype] = (),
) -> None:
    """Compare two NPZ files per-output with NaN/Inf-aware semantics.

    `input_dtypes` is the list of dtypes of the original PyTorch inputs.
    f16/bf16 outputs are cast to f32 during NPZ serialization (see
    `model_wrapper.py:write_output_npz`), which would otherwise cause
    the comparator to look up the f32 (strict) tolerance for what was
    really an f16 computation. We work around that by using the loosest
    tolerance among (NPZ ref dtype) and (input dtypes).
    """
    ref_bundle = np.load(reference_npz)
    act_bundle = np.load(actual_npz)
    missing = set(output_names) - set(act_bundle.files)
    if missing:
        got = sorted(act_bundle.files)
        raise ProptestComparatorError(
            f"tract output NPZ is missing keys: {sorted(missing)} (got {got})"
        )
    for name in output_names:
        ref = ref_bundle[name]
        act = act_bundle[name]
        if ref.shape != act.shape:
            raise ProptestComparatorError(
                f"shape mismatch on output {name!r}: "
                f"ref={ref.shape} vs tract={act.shape}"
            )
        npz_dtype = _torch_dtype_from_numpy(ref.dtype)
        if not is_float_dtype(npz_dtype):
            if not np.array_equal(ref, act):
                raise ProptestComparatorError(
                    f"non-float output {name!r} differs (dtype={ref.dtype})\n"
                    f"ref={ref}\ntract={act}"
                )
            continue
        # Pick the loosest tolerance among the NPZ dtype and any float
        # input dtypes (f16 -> 100x looser than f32).
        candidates = [npz_dtype] + [
            d for d in input_dtypes if is_float_dtype(d)
        ]
        candidate_tols = [lookup_tol(d, tolerance) for d in candidates]
        tol = max(candidate_tols, key=lambda t: max(t.rtol, t.atol))
        try:
            np.testing.assert_allclose(
                act, ref, rtol=tol.rtol, atol=tol.atol, equal_nan=True
            )
        except AssertionError as exc:
            raise ProptestComparatorError(
                f"output {name!r} diverges "
                f"(dtype={ref.dtype}, rtol={tol.rtol:g}, atol={tol.atol:g})\n"
                f"{exc}"
            ) from exc


def assert_outputs_close_nan_aware(
    model: nn.Module,
    inputs: T.Tuple[torch.Tensor, ...],
    inference_target: TractNNEF,
    tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE,
) -> None:
    """Assert that tract's outputs match PyTorch's reference, NaN-aware.

    Args:
        model: an `nn.Module` whose forward returns one or more tensors.
        inputs: positional inputs forwarded to `model(*inputs)`.
        inference_target: a `TractNNEF` instance. A clone with
            `check_io=False` is used internally.
        tolerance: tolerance level for the numeric comparison. Mapped to
            (rtol, atol) per dtype via :mod:`tests.proptest.dtypes`.

    Raises:
        T2NErrorInvalidArgument: when `inference_target` is not a
            `TractNNEF` instance (the only supported target).
        ProptestComparatorError: on any divergence (shape mismatch, missing
            output, non-float bit-exact mismatch, or float values outside
            tolerance).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorInvalidArgument(
            "proptest comparator is tract-only; got "
            f"{type(inference_target).__name__}"
        )
    target = _make_no_check_target(inference_target)
    model = model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nnef_path = tmp / "model.nnef"
        inputs_npz = tmp / "inputs.npz"
        outputs_ref_npz = tmp / "outputs_ref.npz"
        outputs_act_npz = tmp / "outputs_act.npz"

        input_names, output_names = build_io(
            model,
            inputs,
            input_bundle_path=inputs_npz,
            output_bundle_path=outputs_ref_npz,
        )
        exported = export_model_to_nnef(
            model=model,
            args=inputs,
            file_path_export=nnef_path,
            compression_level=0,
            input_names=input_names,
            output_names=output_names,
            log_level=log.WARNING,
            inference_target=target,
            allow_same_io_names=True,
        )
        _run_tract(
            _build_tract_run_cmd(target, exported, inputs_npz, outputs_act_npz)
        )
        input_dtypes = tuple(
            t.dtype for t in inputs if isinstance(t, torch.Tensor)
        )
        _compare_npz(
            outputs_ref_npz,
            outputs_act_npz,
            output_names,
            tolerance,
            input_dtypes=input_dtypes,
        )
