"""Measure what PyTorch's ONNX exporter does with each proptest sample.

Why this exists: the ONNX column of `docs/contributing/supported_operators.md`
used to be scraped from `onnx_torchscript_supported_aten_ops.html`, a page
PyTorch stopped publishing after torch 2.8 when the TorchScript exporter was
retired. That left the column describing an exporter that no longer exists.
The spec catalog in `op_specs/` already builds real modules for 300+ aten
ops, so we can measure the current exporter instead of guessing.

Unlike `comparator.py`, nothing here raises on an export failure: a failure
*is* the measurement. Three axes are recorded independently, because they
fail for different reasons and conflating them misattributes blame:

  A. export:    did `torch.onnx.export(dynamo=True)` produce a graph
  B. runtime:   can onnxruntime load and run that graph
  C. numerics:  do its outputs match PyTorch

Only axis A is a statement about ONNX operator coverage. Axis B is
runtime coverage, and axis C is usually a precision property of whichever
kernel ran, not evidence that the op is unsupported.

Axis A distinguishes *where* export died, which matters because
`dynamo=True` runs `torch.export` first: a capture failure says nothing
about ONNX and must not be reported as a missing ONNX op.
"""

import contextlib
import io
import re
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from .comparator import ProptestComparatorError, compare_arrays

#: Default ONNX opset. 18 is the floor that covers the spec catalog's
#: reductions and `Resize` attributes; recorded in the report so a bump
#: invalidates cached grades rather than silently changing meaning.
DEFAULT_OPSET = 18

# ---------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------

#: Axis A values.
EXPORT_OK = "ok"
#: `torch.export` could not capture the module: not an ONNX verdict.
EXPORT_CAPTURE_FAILED = "capture_failed"
#: The exporter has no ONNX function for an op in the graph. This is the
#: one outcome that actually means "ONNX does not support this".
EXPORT_NO_ONNX_FUNCTION = "no_onnx_function"
#: Lowering reached ONNX but broke for another reason.
EXPORT_CONVERSION_FAILED = "conversion_failed"
#: Anything we did not anticipate; kept distinct so it can be triaged
#: rather than silently counted as an ONNX gap.
EXPORT_UNKNOWN_ERROR = "unknown_error"

#: Axis B / C values.
RUNTIME_OK = "ok"
RUNTIME_LOAD_FAILED = "load_failed"
RUNTIME_RUN_FAILED = "run_failed"
NUMERICS_MATCH = "match"
NUMERICS_DIVERGE = "diverge"
#: Not reached because an earlier axis failed.
NOT_REACHED = "not_reached"

#: Export outcomes that mean "the exporter refused this graph". A capture
#: failure is deliberately excluded: it is a `torch.export` limitation.
ONNX_GAP_OUTCOMES = frozenset(
    {
        EXPORT_NO_ONNX_FUNCTION,
        EXPORT_CONVERSION_FAILED,
        EXPORT_UNKNOWN_ERROR,
    }
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
#: The exporter names the op it could not lower like
#: ``<OpOverload(op='prims.digamma', overload='default')>``.
_BLOCKING_OP_RE = re.compile(r"op='([A-Za-z0-9_.]+)'")
#: ``<class 'torch.onnx....DispatchError'>: No ONNX function found for ...``
_EXC_SUMMARY_RE = re.compile(r"<class '[^']+'>:\s*(.+)")


@dataclass
class ExampleOutcome:
    """What one drawn example did on all three axes."""

    export: str
    runtime: str = NOT_REACHED
    numerics: str = NOT_REACHED
    #: Exception class name, for triage of `unknown_error`.
    error_type: T.Optional[str] = None
    #: First meaningful line of the failure, ANSI-stripped.
    error_head: T.Optional[str] = None
    #: The op the exporter could not lower, when it named one.
    blocking_op: T.Optional[str] = None
    #: Drawn input shapes/dtypes, so a `partial` grade carries the
    #: signature of what broke it rather than just a count.
    shapes: T.Tuple[T.Tuple[int, ...], ...] = ()
    dtypes: T.Tuple[str, ...] = ()

    @property
    def is_onnx_gap(self) -> bool:
        return self.export in ONNX_GAP_OUTCOMES

    def as_dict(self) -> T.Dict[str, T.Any]:
        out: T.Dict[str, T.Any] = {
            "export": self.export,
            "runtime": self.runtime,
            "numerics": self.numerics,
            "shapes": [list(s) for s in self.shapes],
            "dtypes": list(self.dtypes),
        }
        for key, value in (
            ("error_type", self.error_type),
            ("error_head", self.error_head),
            ("blocking_op", self.blocking_op),
        ):
            if value is not None:
                out[key] = value
        return out


@dataclass
class OnnxRunConfig:
    """Environment knobs that change what a measurement means."""

    opset: int = DEFAULT_OPSET
    #: Recorded in the report; only `dynamo` exists from torch 2.9 on.
    path: str = "dynamo"
    #: Compare axis C at all. Off makes the sweep roughly 2x cheaper.
    check_numerics: bool = True
    tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE
    onnxruntime_providers: T.Tuple[str, ...] = ("CPUExecutionProvider",)
    extras: T.Dict[str, str] = field(default_factory=dict)


def _clean(message: str) -> str:
    """Extract the most diagnostic single line from an exporter error.

    The exporter's `str(exc)` leads with boilerplate ("Failed to convert
    the exported program... This is step 3/3... Next steps: ...") and puts
    the actual cause further down under `## Exception summary`, formatted
    as ``<class '...DispatchError'>: No ONNX function found for ...``.
    Reporting the first line would make every failure look identical, so
    prefer the summarized exception when one is present.
    """
    flat = _ANSI_RE.sub("", message).strip()
    match = _EXC_SUMMARY_RE.search(flat)
    if match:
        return match.group(1).strip()[:400]
    for line in flat.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:400]
    return ""


def _classify_export_error(exc: BaseException) -> T.Tuple[str, T.Optional[str]]:
    """Map an exporter exception to an axis-A outcome + blocking op.

    The exporter's own exception hierarchy already draws the line we care
    about (see `torch.onnx._internal.exporter._errors`):
      - `TorchExportError`      -> capture, upstream of ONNX
      - `DispatchError`         -> no ONNX function for some op
      - other `ConversionError` -> lowering broke otherwise
    It is imported lazily and defensively: it is private API, so a rename
    upstream must degrade to `unknown_error`, not crash the sweep.
    """
    text = _ANSI_RE.sub("", f"{exc}")
    blocking = None
    match = _BLOCKING_OP_RE.search(text)
    if match:
        blocking = match.group(1)
    try:
        from torch.onnx._internal.exporter import (  # noqa: PLC0415
            _errors as onnx_errors,
        )
    except ImportError:
        return EXPORT_UNKNOWN_ERROR, blocking
    if isinstance(exc, getattr(onnx_errors, "TorchExportError", ())):
        return EXPORT_CAPTURE_FAILED, blocking
    if isinstance(exc, getattr(onnx_errors, "DispatchError", ())):
        return EXPORT_NO_ONNX_FUNCTION, blocking
    if isinstance(exc, getattr(onnx_errors, "ConversionError", ())):
        # A dispatch failure is often re-raised as a plain ConversionError
        # with the DispatchError kept in the message body.
        if "No ONNX function found" in text or "DispatchError" in text:
            return EXPORT_NO_ONNX_FUNCTION, blocking
        return EXPORT_CONVERSION_FAILED, blocking
    return EXPORT_UNKNOWN_ERROR, blocking


def _reference_outputs(
    model: nn.Module, inputs: T.Sequence[torch.Tensor]
) -> T.List[np.ndarray]:
    """Run the module in torch and flatten its outputs to numpy."""
    with torch.no_grad():
        out = model(*inputs)
    tensors: T.List[torch.Tensor] = []
    if isinstance(out, torch.Tensor):
        tensors = [out]
    else:
        for item in out:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
    return [_to_numpy(t) for t in tensors]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach to numpy, upcasting the dtypes numpy cannot hold natively."""
    detached = tensor.detach()
    if detached.dtype in (torch.bfloat16,):
        detached = detached.to(torch.float32)
    return detached.cpu().numpy()


def measure_example(
    model: nn.Module,
    inputs: T.Sequence[torch.Tensor],
    workdir: Path,
    config: T.Optional[OnnxRunConfig] = None,
) -> ExampleOutcome:
    """Export one sample to ONNX and report all three axes.

    Never raises for an ONNX-side problem: the returned `ExampleOutcome`
    is the result. Only a bug in this function itself should propagate.
    """
    config = config or OnnxRunConfig()
    model = model.eval()
    inputs = tuple(inputs)
    outcome_meta: T.Dict[str, T.Any] = {
        "shapes": tuple(tuple(t.shape) for t in inputs),
        "dtypes": tuple(str(t.dtype) for t in inputs),
    }

    onnx_path = workdir / "model.onnx"
    # The exporter narrates progress on stdout and emits warnings for
    # graphs it had to massage; both are noise over 371 specs x N examples.
    sink = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            torch.onnx.export(
                model,
                inputs,
                str(onnx_path),
                dynamo=config.path == "dynamo",
                opset_version=config.opset,
                verbose=False,
            )
    # Deliberately broad: this is a measurement harness, and an exporter
    # that raises something unforeseen must be recorded, not crash the run.
    except Exception as exc:  # noqa: BLE001
        export_outcome, blocking = _classify_export_error(exc)
        return ExampleOutcome(
            export=export_outcome,
            error_type=type(exc).__name__,
            error_head=_clean(f"{exc}"),
            blocking_op=blocking,
            **outcome_meta,
        )

    reference = _reference_outputs(model, inputs)
    return _measure_runtime(onnx_path, inputs, reference, config, outcome_meta)


def _quiet_session_options(ort):
    """Session options that keep onnxruntime's own logger quiet.

    onnxruntime logs from C++ straight to the process file descriptors, so
    `contextlib.redirect_stdout` cannot intercept it: over 371 specs x N
    examples its per-graph notices ("Removing initializer ...", constant
    folding notes) bury the actual pytest output. Severity 3 is
    error-and-above, and a real load failure still surfaces as the
    raised exception we classify, not as a log line.
    """
    # Also covers anything logged before per-session options apply.
    setter = getattr(ort, "set_default_logger_severity", None)
    if setter is not None:
        setter(3)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return options


def _measure_runtime(
    onnx_path: Path,
    inputs: T.Sequence[torch.Tensor],
    reference: T.Sequence[np.ndarray],
    config: OnnxRunConfig,
    outcome_meta: T.Dict[str, T.Any],
) -> ExampleOutcome:
    """Axes B and C: load the graph in onnxruntime, run it, compare."""
    import onnxruntime as ort  # noqa: PLC0415

    sink = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=_quiet_session_options(ort),
                providers=list(config.onnxruntime_providers),
            )
    except Exception as exc:  # noqa: BLE001
        return ExampleOutcome(
            export=EXPORT_OK,
            runtime=RUNTIME_LOAD_FAILED,
            error_type=type(exc).__name__,
            error_head=_clean(f"{exc}"),
            **outcome_meta,
        )

    try:
        feed = {
            spec.name: _to_numpy(inputs[idx])
            for idx, spec in enumerate(session.get_inputs())
        }
        with (
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            actual = session.run(None, feed)
    except Exception as exc:  # noqa: BLE001
        return ExampleOutcome(
            export=EXPORT_OK,
            runtime=RUNTIME_RUN_FAILED,
            error_type=type(exc).__name__,
            error_head=_clean(f"{exc}"),
            **outcome_meta,
        )

    if not config.check_numerics:
        return ExampleOutcome(
            export=EXPORT_OK, runtime=RUNTIME_OK, **outcome_meta
        )

    input_dtypes = tuple(t.dtype for t in inputs)
    try:
        if len(actual) != len(reference):
            raise ProptestComparatorError(
                f"output count mismatch: torch produced {len(reference)}, "
                f"onnxruntime produced {len(actual)}"
            )
        # strict: the count check above already guarantees equal lengths,
        # so a mismatch here would be a bug in this function.
        for idx, (ref, act) in enumerate(zip(reference, actual, strict=True)):
            compare_arrays(
                ref,
                np.asarray(act),
                name=f"output_{idx}",
                tolerance=config.tolerance,
                input_dtypes=input_dtypes,
            )
    except ProptestComparatorError as exc:
        return ExampleOutcome(
            export=EXPORT_OK,
            runtime=RUNTIME_OK,
            numerics=NUMERICS_DIVERGE,
            error_type=type(exc).__name__,
            error_head=_clean(f"{exc}"),
            **outcome_meta,
        )
    return ExampleOutcome(
        export=EXPORT_OK,
        runtime=RUNTIME_OK,
        numerics=NUMERICS_MATCH,
        **outcome_meta,
    )
