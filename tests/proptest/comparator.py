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

`compare_arrays` / `resolve_tol` are the tolerance policy, kept public
here because `onnx_backend.py` measures a different exporter but must
judge numeric agreement by the same rules. `run_tract` is public for the
same reason: `nnef_gap.py` has to invoke tract exactly as this module
does, since "tract refuses the graph" is one of the failure stages a
declared gap can name.
"""

import subprocess
import tempfile
import typing as T
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_to_nnef.dtypes import (
    NUMPY_TO_TORCH_DTYPE,
    dtype_is_floating_point,
)
from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    build_io,
)
from torch_to_nnef.log import log

from .dtypes import Tol, lookup_tol


class ProptestComparatorError(AssertionError):
    """Raised when tract output diverges from the PyTorch reference."""


def _torch_dtype_from_numpy(np_dtype: np.dtype) -> torch.dtype:
    """Numpy -> torch dtype lookup for tolerance dispatch."""
    return NUMPY_TO_TORCH_DTYPE[np.dtype(np_dtype).type]


def resolve_tol(
    npz_dtype: torch.dtype,
    tolerance: TractCheckTolerance,
    input_dtypes: T.Sequence[torch.dtype] = (),
) -> Tol:
    """Pick the loosest tolerance among the output and float input dtypes.

    f16/bf16 outputs are cast to f32 during NPZ serialization (see
    `model_wrapper.py:write_output_npz`), which would otherwise have the
    caller look up the f32 (strict) tolerance for what was really an f16
    computation.
    """
    candidates = [npz_dtype] + [
        d for d in input_dtypes if dtype_is_floating_point(d)
    ]
    candidate_tols = [lookup_tol(d, tolerance) for d in candidates]
    return max(candidate_tols, key=lambda t: max(t.rtol, t.atol))


def compare_arrays(
    reference: np.ndarray,
    actual: np.ndarray,
    name: str,
    tolerance: TractCheckTolerance,
    input_dtypes: T.Sequence[torch.dtype] = (),
) -> None:
    """Compare one output array against its reference, NaN/Inf-aware.

    Shared by every proptest backend so they all apply one tolerance
    policy: shape must match exactly, non-float dtypes must be bit-exact,
    and float dtypes go through `assert_allclose(equal_nan=True)` at the
    dtype-resolved tolerance.

    Raises:
        ProptestComparatorError: on shape mismatch, non-float mismatch, or
            float values outside tolerance.
    """
    if reference.shape != actual.shape:
        raise ProptestComparatorError(
            f"shape mismatch on output {name!r}: "
            f"ref={reference.shape} vs actual={actual.shape}"
        )
    npz_dtype = _torch_dtype_from_numpy(reference.dtype)
    if not dtype_is_floating_point(npz_dtype):
        if not np.array_equal(reference, actual):
            raise ProptestComparatorError(
                f"non-float output {name!r} differs "
                f"(dtype={reference.dtype})\n"
                f"ref={reference}\nactual={actual}"
            )
        return
    tol = resolve_tol(npz_dtype, tolerance, input_dtypes)
    try:
        np.testing.assert_allclose(
            actual, reference, rtol=tol.rtol, atol=tol.atol, equal_nan=True
        )
    except AssertionError as exc:
        raise ProptestComparatorError(
            f"output {name!r} diverges "
            f"(dtype={reference.dtype}, rtol={tol.rtol:g}, "
            f"atol={tol.atol:g})\n{exc}"
        ) from exc


def make_no_check_target(target: TractNNEF) -> TractNNEF:
    """Return a copy of `target` with `check_io=False`.

    We disable the built-in `post_export` assert so the caller owns the
    comparison. Public because `nnef_gap.py` needs it for a different
    reason: with `check_io` on, a numeric divergence surfaces as a
    `T2NError` from *inside* the export, which that module would then
    misread as an export failure.
    """
    twin = deepcopy(target)
    twin.check_io = False
    return twin


def run_tract(cmd: T.List[str]) -> None:
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


def _compare_npz(
    reference_npz: Path,
    actual_npz: Path,
    output_names: T.Sequence[str],
    tolerance: TractCheckTolerance,
    input_dtypes: T.Sequence[torch.dtype] = (),
) -> None:
    """Compare two NPZ files per-output with NaN/Inf-aware semantics.

    `input_dtypes` is the list of dtypes of the original PyTorch inputs;
    see `resolve_tol` for why they matter. The per-array comparison lives
    in `compare_arrays` so the ONNX backend applies the same policy.
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
        compare_arrays(
            ref_bundle[name],
            act_bundle[name],
            name=name,
            tolerance=tolerance,
            input_dtypes=input_dtypes,
        )


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
    target = make_no_check_target(inference_target)
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
        run_tract(
            target.tract_cli.run_save_outputs_cmd_str(
                exported, inputs_npz, outputs_act_npz
            )
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
