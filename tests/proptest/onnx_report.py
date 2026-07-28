"""Turn per-example ONNX outcomes into a graded, per-operator report.

The support page needs one verdict per `aten::` name; the sweep produces
many outcomes per spec, and several specs may exercise the same operator.
So results are merged at the operator level: every example drawn for every
spec declaring an op contributes to that op's counters, and the grade is
derived from the counters.

Merging (rather than taking the worst spec's grade) is what makes
`partial` meaningful. If `conv2d` exports but `conv2d-dilation-groups`
does not, the honest operator-level answer is "works for some
configurations", which is exactly what the merged counters say.

The artifact is committed to `docs/contributing/` so the page can be
regenerated offline, and so a grade change shows up in review as a diff
rather than as a silent difference between two runs.
"""

import datetime as dt
import json
import typing as T
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .onnx_backend import (
    EXPORT_CAPTURE_FAILED,
    EXPORT_OK,
    NOT_REACHED,
    NUMERICS_DIVERGE,
    NUMERICS_MATCH,
    RUNTIME_OK,
    ExampleOutcome,
    OnnxRunConfig,
)

SCHEMA_VERSION = 1

#: Axis grades. `blocked` is not a verdict about ONNX: it means
#: `torch.export` could not capture the module, so the exporter never got
#: a chance. Kept separate so it cannot be read as "ONNX lacks this op".
GRADE_FULL = "full"
GRADE_PARTIAL = "partial"
GRADE_NONE = "none"
GRADE_BLOCKED = "blocked"
#: Only ever produced by the page generator, for rows no spec covers.
GRADE_UNTESTED = "untested"

#: Numerics-axis grades.
NUMERICS_GRADE_MATCH = "match"
NUMERICS_GRADE_PARTIAL = "partial"
NUMERICS_GRADE_DIVERGE = "diverge"
NUMERICS_GRADE_NOT_REACHED = NOT_REACHED

#: How many failing examples to keep per spec in the artifact. Enough to
#: see the pattern behind a `partial`, few enough to keep the file
#: reviewable.
MAX_FAILURE_EXEMPLARS = 3

#: Fraction denominator for the anti-staleness rotation: a reused `full`
#: op is re-measured at least once every this many regenerations.
REVALIDATION_PERIOD = 10


@dataclass
class OpCounters:
    """Merged example counts for one aten op across all its specs."""

    examples: int = 0
    export_ok: int = 0
    capture_failed: int = 0
    onnx_gap: int = 0
    runtime_reached: int = 0
    runtime_ok: int = 0
    numerics_reached: int = 0
    numerics_match: int = 0
    specs: T.Set[str] = field(default_factory=set)

    def add(self, spec_name: str, outcome: ExampleOutcome) -> None:
        self.specs.add(spec_name)
        self.examples += 1
        if outcome.export == EXPORT_OK:
            self.export_ok += 1
        elif outcome.export == EXPORT_CAPTURE_FAILED:
            self.capture_failed += 1
        else:
            self.onnx_gap += 1
        if outcome.runtime != NOT_REACHED:
            self.runtime_reached += 1
            if outcome.runtime == RUNTIME_OK:
                self.runtime_ok += 1
        if outcome.numerics != NOT_REACHED:
            self.numerics_reached += 1
            if outcome.numerics == NUMERICS_MATCH:
                self.numerics_match += 1

    @property
    def export_grade(self) -> str:
        if self.examples == 0:
            return GRADE_UNTESTED
        if self.export_ok == self.examples:
            return GRADE_FULL
        if self.export_ok > 0:
            return GRADE_PARTIAL
        # Nothing exported. Blame ONNX only if at least one failure
        # actually reached the exporter.
        if self.onnx_gap == 0:
            return GRADE_BLOCKED
        return GRADE_NONE

    @property
    def runtime_grade(self) -> str:
        if self.runtime_reached == 0:
            return NOT_REACHED
        if self.runtime_ok == self.runtime_reached:
            return GRADE_FULL
        if self.runtime_ok > 0:
            return GRADE_PARTIAL
        return GRADE_NONE

    @property
    def numerics_grade(self) -> str:
        if self.numerics_reached == 0:
            return NUMERICS_GRADE_NOT_REACHED
        if self.numerics_match == self.numerics_reached:
            return NUMERICS_GRADE_MATCH
        if self.numerics_match > 0:
            return NUMERICS_GRADE_PARTIAL
        return NUMERICS_GRADE_DIVERGE

    #: Counter fields persisted in the artifact. Grades are stored too,
    #: for readability, but these are what make a record re-mergeable:
    #: the page generator has to combine several artifact keys when they
    #: normalize onto one table row (an alias, or an in-place variant).
    COUNT_FIELDS = (
        "examples",
        "export_ok",
        "capture_failed",
        "onnx_gap",
        "runtime_reached",
        "runtime_ok",
        "numerics_reached",
        "numerics_match",
    )

    def as_dict(self) -> T.Dict[str, T.Any]:
        record: T.Dict[str, T.Any] = {
            "export": self.export_grade,
            "runtime": self.runtime_grade,
            "numerics": self.numerics_grade,
        }
        for name in self.COUNT_FIELDS:
            record[name] = getattr(self, name)
        record["specs"] = sorted(self.specs)
        return record

    @classmethod
    def from_record(cls, record: T.Mapping[str, T.Any]) -> "OpCounters":
        """Rebuild counters from a serialized record."""
        counters = cls()
        for name in cls.COUNT_FIELDS:
            setattr(counters, name, int(record.get(name, 0)))
        counters.specs = set(record.get("specs", ()))
        return counters

    def __add__(self, other: "OpCounters") -> "OpCounters":
        merged = OpCounters()
        for name in self.COUNT_FIELDS:
            setattr(merged, name, getattr(self, name) + getattr(other, name))
        merged.specs = self.specs | other.specs
        return merged


def merge_op_records(
    records: T.Sequence[T.Mapping[str, T.Any]],
) -> T.Dict[str, T.Any]:
    """Combine several serialized op records into one graded record.

    Used by the page generator when more than one measured operator maps
    to the same table row. Grades are re-derived from the summed counters
    rather than picked between, so two `full` halves stay `full` and a
    `full` plus a `none` becomes `partial`, matching how several specs on
    one operator are merged during the sweep.
    """
    total = OpCounters()
    for record in records:
        total = total + OpCounters.from_record(record)
    merged = total.as_dict()
    merged["reused"] = any(r.get("reused") for r in records)
    return merged


def environment_fingerprint(config: OnnxRunConfig) -> T.Dict[str, T.Any]:
    """Everything that changes what an ONNX measurement means.

    A grade is evidence about one environment, not a fact about the op, so
    reuse is only valid when this dict matches exactly. The torch 2.8 ->
    2.9 exporter swap is the cautionary case: every TorchScript-supported
    op regressed at once, and only a version check catches that.
    """
    versions: T.Dict[str, T.Any] = {}
    for module_name, key in (
        ("torch", "torch"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnxscript", "onnxscript"),
    ):
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[key] = None
    versions["opset"] = config.opset
    versions["path"] = config.path
    versions["check_numerics"] = config.check_numerics
    versions.update(config.extras)
    return versions


def _fingerprint_key(fingerprint: T.Mapping[str, T.Any]) -> str:
    """Stable comparison key for a fingerprint dict."""
    return json.dumps(fingerprint, sort_keys=True)


def revalidation_slice_member(op_name: str, regen_index: int) -> bool:
    """Whether `op_name` is force-re-measured on this regeneration.

    Uses crc32, not `hash()`: Python salts string hashes per process, so
    `hash()` would rotate the slice on every run and make the schedule
    unreproducible.
    """
    bucket = zlib.crc32(op_name.encode("utf8")) % REVALIDATION_PERIOD
    return bucket == regen_index % REVALIDATION_PERIOD


class ReuseIndex:
    """Decides which specs can keep a previously measured `full` grade.

    Reuse is deliberately restricted to `full`: `none` and `partial` are
    the grades that improve when PyTorch adds support, so re-measuring
    them every run is where the value is.
    """

    def __init__(
        self,
        prior: T.Optional[T.Mapping[str, T.Any]],
        fingerprint: T.Mapping[str, T.Any],
        current_examples: int,
        enabled: bool = True,
    ):
        self.enabled = enabled and bool(prior)
        self._prior = prior or {}
        self._fingerprint_key = _fingerprint_key(fingerprint)
        self._current_examples = current_examples
        self._ops: T.Mapping[str, T.Any] = self._prior.get("ops", {})
        self._measurements: T.Mapping[str, T.Any] = self._prior.get(
            "measurements", {}
        )
        # Monotonic counter driving the revalidation rotation.
        self.regen_index = int(self._prior.get("regen_index", -1)) + 1
        self.reused_specs: T.Dict[str, str] = {}
        self.reused_ops: T.Set[str] = set()

    def _reusable_measurement(self, op_name: str) -> T.Optional[str]:
        record = self._ops.get(op_name, {}).get("onnx")
        if not record or record.get("export") != GRADE_FULL:
            return None
        measurement_id = record.get("measurement")
        measurement = self._measurements.get(measurement_id)
        if not measurement:
            return None
        stored = {
            k: v
            for k, v in measurement.items()
            if k not in ("examples", "profile", "generated_utc")
        }
        if _fingerprint_key(stored) != self._fingerprint_key:
            return None
        # A `full` from 25 draws is not evidence for a 200-draw run.
        if int(measurement.get("examples", 0)) < self._current_examples:
            return None
        if revalidation_slice_member(op_name, self.regen_index):
            return None
        return measurement_id

    def reuse_for(
        self,
        spec_name: str,
        aten_ops: T.Sequence[str],
        check_numerics: bool = True,
    ) -> bool:
        """Record and report whether `spec_name` can be skipped.

        `check_numerics` is the spec's own setting, which can differ from
        the sweep's: a nondeterministic op is measured with numerics off,
        and the environment fingerprint (computed once, per sweep) cannot
        express that. Without the check below, dropping
        `nondeterministic=True` from a spec would silently carry forward
        a record whose numerics axis was never measured, so the very run
        meant to start measuring it would skip it instead.
        """
        if not self.enabled or not aten_ops:
            return False
        measurement_ids = [self._reusable_measurement(op) for op in aten_ops]
        if any(mid is None for mid in measurement_ids):
            return False
        if check_numerics and any(
            not self._ops.get(op, {}).get("onnx", {}).get("numerics_reached")
            for op in aten_ops
        ):
            return False
        self.reused_specs[spec_name] = measurement_ids[0]  # type: ignore[assignment]
        self.reused_ops.update(aten_ops)
        return True

    def carried_op_record(self, op_name: str) -> T.Optional[T.Dict[str, T.Any]]:
        """The prior record for an op whose specs were all skipped."""
        record = self._ops.get(op_name, {}).get("onnx")
        if record is None:
            return None
        carried = dict(record)
        carried["reused"] = True
        return carried

    def carried_measurements(self) -> T.Dict[str, T.Any]:
        """Prior measurement entries still referenced by carried records."""
        return {
            mid: dict(entry)
            for mid, entry in self._measurements.items()
            if mid in set(self.reused_specs.values())
        }


class OnnxReport:
    """Collects outcomes during the sweep and serializes the artifact."""

    def __init__(
        self,
        config: OnnxRunConfig,
        max_examples: int,
        profile: str,
        reuse: T.Optional[ReuseIndex] = None,
        now_utc: T.Optional[str] = None,
    ):
        self.config = config
        self.max_examples = max_examples
        self.profile = profile
        self.reuse = reuse
        self.fingerprint = environment_fingerprint(config)
        self._now = now_utc or dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        )
        self._ops: T.Dict[str, OpCounters] = {}
        self._spec_failures: T.Dict[str, T.List[ExampleOutcome]] = {}
        self._spec_failure_signatures: T.Dict[str, T.Set[T.Tuple]] = {}
        self._spec_examples: T.Dict[str, int] = {}
        self._spec_ops: T.Dict[str, T.Tuple[str, ...]] = {}

    def record(
        self,
        spec_name: str,
        aten_ops: T.Sequence[str],
        outcome: ExampleOutcome,
    ) -> None:
        """Attribute one drawn example to each op the spec declares."""
        self._spec_ops[spec_name] = tuple(aten_ops)
        self._spec_examples[spec_name] = (
            self._spec_examples.get(spec_name, 0) + 1
        )
        if outcome.export != EXPORT_OK or outcome.numerics == NUMERICS_DIVERGE:
            failures = self._spec_failures.setdefault(spec_name, [])
            # Keep distinct signatures, not the first N examples: 25 draws
            # of an unsupported op yield 25 identical records, whereas a
            # `partial` grade is only informative if the exemplars show
            # the different ways it failed.
            signature = (
                outcome.export,
                outcome.runtime,
                outcome.numerics,
                outcome.blocking_op,
                outcome.error_head,
            )
            seen = self._spec_failure_signatures.setdefault(spec_name, set())
            if signature not in seen and len(failures) < MAX_FAILURE_EXEMPLARS:
                seen.add(signature)
                failures.append(outcome)
        for op_name in aten_ops:
            self._ops.setdefault(op_name, OpCounters()).add(spec_name, outcome)

    @property
    def measurement_id(self) -> str:
        """Id of the measurement this run contributes."""
        return f"m{self.regen_index}"

    @property
    def regen_index(self) -> int:
        return self.reuse.regen_index if self.reuse else 0

    def as_dict(self) -> T.Dict[str, T.Any]:
        measurements: T.Dict[str, T.Any] = {}
        if self.reuse:
            measurements.update(self.reuse.carried_measurements())
        measurements[self.measurement_id] = {
            **self.fingerprint,
            "examples": self.max_examples,
            "profile": self.profile,
            "generated_utc": self._now,
        }

        ops: T.Dict[str, T.Any] = {}
        for op_name, counters in self._ops.items():
            ops[op_name] = {
                "onnx": {
                    **counters.as_dict(),
                    "measurement": self.measurement_id,
                    "reused": False,
                }
            }
        # Ops whose every spec was skipped keep their prior verdict, tagged
        # so the page can say which grades were carried over.
        if self.reuse:
            for op_name in sorted(self.reuse.reused_ops):
                if op_name in ops:
                    continue
                carried = self.reuse.carried_op_record(op_name)
                if carried is not None:
                    ops[op_name] = {"onnx": carried}

        specs: T.Dict[str, T.Any] = {}
        for spec_name, count in sorted(self._spec_examples.items()):
            entry: T.Dict[str, T.Any] = {
                "aten_ops": list(self._spec_ops.get(spec_name, ())),
                "examples": count,
            }
            failures = self._spec_failures.get(spec_name)
            if failures:
                entry["failures"] = [f.as_dict() for f in failures]
            specs[spec_name] = entry
        if self.reuse:
            for spec_name, measurement_id in sorted(
                self.reuse.reused_specs.items()
            ):
                specs.setdefault(
                    spec_name,
                    {
                        "aten_ops": [],
                        "examples": 0,
                        "reused_from": measurement_id,
                    },
                )

        return {
            "schema": SCHEMA_VERSION,
            "regen_index": self.regen_index,
            "measurements": measurements,
            "ops": dict(sorted(ops.items())),
            "specs": specs,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as fh:
            json.dump(self.as_dict(), fh, indent=2, sort_keys=False)
            fh.write("\n")

    def summary_lines(self) -> T.List[str]:
        """Short human summary for the pytest terminal report."""
        buckets: T.Dict[str, int] = {}
        for counters in self._ops.values():
            grade = counters.export_grade
            buckets[grade] = buckets.get(grade, 0) + 1
        lines = [
            "ONNX support sweep (export axis, measured this run): "
            + ", ".join(
                f"{grade}={buckets.get(grade, 0)}"
                for grade in (
                    GRADE_FULL,
                    GRADE_PARTIAL,
                    GRADE_NONE,
                    GRADE_BLOCKED,
                )
            )
        ]
        if self.reuse and self.reuse.reused_specs:
            lines.append(
                f"reused {len(self.reuse.reused_specs)} spec(s) / "
                f"{len(self.reuse.reused_ops)} op(s) from a prior run "
                f"(regen_index={self.regen_index})"
            )
        return lines


def load_prior(path: T.Optional[Path]) -> T.Optional[T.Dict[str, T.Any]]:
    """Read a previous artifact, tolerating its absence."""
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf8") as fh:
        return json.load(fh)
