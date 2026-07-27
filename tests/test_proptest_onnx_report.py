"""Unit tests for the ONNX support report's grading and reuse rules.

These run in the normal suite: they exercise pure aggregation logic and
need neither `onnx` nor `onnxruntime`, only the vocabulary from
`onnx_backend`. The sweep itself lives behind the `proptest_onnx` marker.
"""

import json

import pytest

from .proptest.onnx_backend import (
    EXPORT_CAPTURE_FAILED,
    EXPORT_NO_ONNX_FUNCTION,
    EXPORT_OK,
    NUMERICS_DIVERGE,
    NUMERICS_MATCH,
    RUNTIME_LOAD_FAILED,
    RUNTIME_OK,
    ExampleOutcome,
    OnnxRunConfig,
)
from .proptest.onnx_report import (
    GRADE_BLOCKED,
    GRADE_FULL,
    GRADE_NONE,
    GRADE_PARTIAL,
    NUMERICS_GRADE_DIVERGE,
    NUMERICS_GRADE_PARTIAL,
    REVALIDATION_PERIOD,
    OnnxReport,
    ReuseIndex,
    environment_fingerprint,
    revalidation_slice_member,
)


def _ok(numerics: str = NUMERICS_MATCH) -> ExampleOutcome:
    return ExampleOutcome(
        export=EXPORT_OK, runtime=RUNTIME_OK, numerics=numerics
    )


def _gap() -> ExampleOutcome:
    return ExampleOutcome(export=EXPORT_NO_ONNX_FUNCTION)


def _capture_failure() -> ExampleOutcome:
    return ExampleOutcome(export=EXPORT_CAPTURE_FAILED)


def _report(**kwargs) -> OnnxReport:
    kwargs.setdefault("config", OnnxRunConfig())
    kwargs.setdefault("max_examples", 25)
    kwargs.setdefault("profile", "dev")
    kwargs.setdefault("now_utc", "2026-01-01T00:00:00+00:00")
    return OnnxReport(**kwargs)


def _grade(report: OnnxReport, op: str) -> str:
    return report.as_dict()["ops"][op]["onnx"]["export"]


class TestExportGrading:
    def test_all_examples_export_gives_full(self):
        report = _report()
        for _ in range(3):
            report.record("relu", ("relu",), _ok())
        assert _grade(report, "relu") == GRADE_FULL

    def test_mixed_examples_give_partial(self):
        report = _report()
        report.record("softmax", ("softmax",), _ok())
        report.record("softmax", ("softmax",), _gap())
        assert _grade(report, "softmax") == GRADE_PARTIAL

    def test_no_export_with_onnx_gap_gives_none(self):
        report = _report()
        report.record("digamma", ("digamma",), _gap())
        assert _grade(report, "digamma") == GRADE_NONE

    def test_capture_failure_alone_is_blocked_not_none(self):
        """A torch.export failure is not evidence about ONNX coverage."""
        report = _report()
        report.record("weird", ("weird",), _capture_failure())
        assert _grade(report, "weird") == GRADE_BLOCKED

    def test_capture_failure_mixed_with_onnx_gap_gives_none(self):
        report = _report()
        report.record("weird", ("weird",), _capture_failure())
        report.record("weird", ("weird",), _gap())
        assert _grade(report, "weird") == GRADE_NONE

    def test_two_specs_on_one_op_merge_into_partial(self):
        """The whole point of merging rather than taking the worst spec.

        `conv2d` exporting while `conv2d-dilation-groups` does not means
        the operator works for some configurations.
        """
        report = _report()
        report.record("conv2d", ("conv2d",), _ok())
        report.record("conv2d-dilation-groups", ("conv2d",), _gap())
        record = report.as_dict()["ops"]["conv2d"]["onnx"]
        assert record["export"] == GRADE_PARTIAL
        assert record["examples"] == 2
        assert record["specs"] == ["conv2d", "conv2d-dilation-groups"]

    def test_multi_op_spec_attributes_to_every_declared_op(self):
        report = _report()
        report.record("tanhshrink", ("sub", "tanh"), _gap())
        ops = report.as_dict()["ops"]
        assert ops["sub"]["onnx"]["export"] == GRADE_NONE
        assert ops["tanh"]["onnx"]["export"] == GRADE_NONE


class TestOtherAxes:
    def test_runtime_and_numerics_are_graded_independently(self):
        report = _report()
        report.record("foo", ("foo",), _ok(numerics=NUMERICS_DIVERGE))
        record = report.as_dict()["ops"]["foo"]["onnx"]
        # Export coverage is unaffected by a numeric mismatch: that is the
        # whole reason the axes are separate.
        assert record["export"] == GRADE_FULL
        assert record["runtime"] == GRADE_FULL
        assert record["numerics"] == NUMERICS_GRADE_DIVERGE

    def test_partial_numerics(self):
        report = _report()
        report.record("foo", ("foo",), _ok())
        report.record("foo", ("foo",), _ok(numerics=NUMERICS_DIVERGE))
        record = report.as_dict()["ops"]["foo"]["onnx"]
        assert record["numerics"] == NUMERICS_GRADE_PARTIAL

    def test_runtime_load_failure_keeps_export_full(self):
        report = _report()
        report.record(
            "foo",
            ("foo",),
            ExampleOutcome(export=EXPORT_OK, runtime=RUNTIME_LOAD_FAILED),
        )
        record = report.as_dict()["ops"]["foo"]["onnx"]
        assert record["export"] == GRADE_FULL
        assert record["runtime"] == GRADE_NONE


class TestFailureExemplars:
    def test_distinct_signatures_are_kept_not_first_n(self):
        report = _report()
        for _ in range(5):
            report.record(
                "foo",
                ("foo",),
                ExampleOutcome(
                    export=EXPORT_NO_ONNX_FUNCTION, blocking_op="prims.a"
                ),
            )
        report.record(
            "foo",
            ("foo",),
            ExampleOutcome(
                export=EXPORT_NO_ONNX_FUNCTION, blocking_op="prims.b"
            ),
        )
        failures = report.as_dict()["specs"]["foo"]["failures"]
        assert [f["blocking_op"] for f in failures] == ["prims.a", "prims.b"]


class TestReuse:
    def _prior(self, tmp_path, grade=GRADE_FULL, examples=25, **overrides):
        config = OnnxRunConfig()
        fingerprint = environment_fingerprint(config)
        fingerprint.update(overrides)
        payload = {
            "schema": 1,
            "regen_index": 0,
            "measurements": {
                "m0": {
                    **fingerprint,
                    "examples": examples,
                    "profile": "dev",
                    "generated_utc": "2026-01-01T00:00:00+00:00",
                }
            },
            "ops": {
                "relu": {
                    "onnx": {
                        "export": grade,
                        "runtime": GRADE_FULL,
                        "numerics": NUMERICS_MATCH,
                        "examples": examples,
                        "specs": ["relu"],
                        "measurement": "m0",
                    }
                }
            },
            "specs": {},
        }
        path = tmp_path / "prior.json"
        path.write_text(json.dumps(payload), encoding="utf8")
        return json.loads(path.read_text(encoding="utf8"))

    def _index(self, prior, current_examples=25, **kwargs):
        config = OnnxRunConfig()
        return ReuseIndex(
            prior=prior,
            fingerprint=environment_fingerprint(config),
            current_examples=current_examples,
            **kwargs,
        )

    def test_full_grade_is_reused_when_fingerprint_matches(self, tmp_path):
        index = self._index(self._prior(tmp_path))
        # regen_index 1 must not put "relu" in the revalidation slice for
        # this assertion to be about the fingerprint alone.
        if revalidation_slice_member("relu", index.regen_index):
            pytest.skip("relu falls in this regeneration's recheck slice")
        assert index.reuse_for("relu", ("relu",)) is True

    def test_partial_grade_is_never_reused(self, tmp_path):
        index = self._index(self._prior(tmp_path, grade=GRADE_PARTIAL))
        assert index.reuse_for("relu", ("relu",)) is False

    def test_none_grade_is_never_reused(self, tmp_path):
        """`none` -> `full` is the common upstream transition."""
        index = self._index(self._prior(tmp_path, grade=GRADE_NONE))
        assert index.reuse_for("relu", ("relu",)) is False

    def test_fingerprint_mismatch_invalidates(self, tmp_path):
        index = self._index(self._prior(tmp_path, torch="0.0.0-not-this-one"))
        assert index.reuse_for("relu", ("relu",)) is False

    def test_fewer_prior_examples_invalidates(self, tmp_path):
        """A `full` from 25 draws is not evidence for a 200-draw run."""
        index = self._index(
            self._prior(tmp_path, examples=25), current_examples=200
        )
        assert index.reuse_for("relu", ("relu",)) is False

    def test_more_prior_examples_is_acceptable(self, tmp_path):
        index = self._index(
            self._prior(tmp_path, examples=200), current_examples=25
        )
        if revalidation_slice_member("relu", index.regen_index):
            pytest.skip("relu falls in this regeneration's recheck slice")
        assert index.reuse_for("relu", ("relu",)) is True

    def test_disabled_reuse_never_reuses(self, tmp_path):
        index = self._index(self._prior(tmp_path), enabled=False)
        assert index.reuse_for("relu", ("relu",)) is False

    def test_spec_needs_all_declared_ops_full(self, tmp_path):
        """One unmeasured op in the spec forces the whole spec to re-run."""
        index = self._index(self._prior(tmp_path))
        assert index.reuse_for("relu", ("relu", "never_measured")) is False

    def test_reused_op_record_is_carried_and_tagged(self, tmp_path):
        prior = self._prior(tmp_path, examples=200)
        index = self._index(prior, current_examples=25)
        if revalidation_slice_member("relu", index.regen_index):
            pytest.skip("relu falls in this regeneration's recheck slice")
        assert index.reuse_for("relu", ("relu",)) is True
        report = _report(reuse=index)
        payload = report.as_dict()
        carried = payload["ops"]["relu"]["onnx"]
        assert carried["export"] == GRADE_FULL
        assert carried["reused"] is True
        # The measurement it came from must survive, or the grade would
        # reference a dangling id and never be reusable again.
        assert carried["measurement"] in payload["measurements"]

    def test_regen_index_advances_from_prior(self, tmp_path):
        index = self._index(self._prior(tmp_path))
        assert index.regen_index == 1

    def test_regen_index_starts_at_zero_without_prior(self):
        index = self._index(None)
        assert index.regen_index == 0
        assert index.enabled is False


class TestRevalidationSlice:
    def test_every_op_is_rechecked_within_one_period(self):
        """No op may be carried over forever."""
        for op in ("relu", "gelu", "conv2d", "digamma", "sum", "index_put"):
            rechecked = [
                revalidation_slice_member(op, i)
                for i in range(REVALIDATION_PERIOD)
            ]
            assert sum(rechecked) == 1, op

    @pytest.mark.parametrize(
        ("op_name", "bucket"),
        [("relu", 6), ("gelu", 5), ("conv2d", 7), ("digamma", 6)],
    )
    def test_membership_is_pinned_to_a_stable_hash(self, op_name, bucket):
        """crc32, not `hash()`.

        Python salts string hashes per process, so `hash()` would rotate
        the recheck schedule on every run: an op could be revalidated
        twice in a row and another never. Pinning the expected buckets
        makes that substitution fail here rather than silently degrade the
        anti-staleness guarantee.
        """
        assert revalidation_slice_member(op_name, bucket) is True
        others = [
            i
            for i in range(REVALIDATION_PERIOD)
            if i != bucket and revalidation_slice_member(op_name, i)
        ]
        assert others == []
