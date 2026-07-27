"""Tests for how the support page maps measured ONNX grades onto rows.

The artifact is keyed by the aten names proptest specs declare; the page
applies its own normalizations (aliases collapse onto a canonical name,
in-place variants merge into the base name). A mistake there attributes a
grade to the wrong operator, which is invisible in the rendered page, so
it gets tests.

Skipped when the docs toolchain (`bs4` / `requests` / `rich`) is absent:
those live in the `dev` dependency group, not `test`.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from .proptest.onnx_report import (
    GRADE_FULL,
    GRADE_NONE,
    GRADE_PARTIAL,
    GRADE_UNTESTED,
    SCHEMA_VERSION,
)

pytest.importorskip("bs4")
pytest.importorskip("requests")
pytest.importorskip("rich")

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen():
    """Load the generator script by path (it is not an importable module)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    path = _REPO_ROOT / "docs/contributing/generate_support_page.py"
    spec = importlib.util.spec_from_file_location("generate_support_page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(export=GRADE_FULL, examples=10, export_ok=10, **counts):
    record = {
        "export": export,
        "runtime": GRADE_FULL,
        "numerics": "match",
        "examples": examples,
        "export_ok": export_ok,
        "capture_failed": 0,
        "onnx_gap": examples - export_ok,
        "runtime_reached": export_ok,
        "runtime_ok": export_ok,
        "numerics_reached": export_ok,
        "numerics_match": export_ok,
        "specs": ["a_spec"],
        "measurement": "m0",
    }
    record.update(counts)
    return record


def _payload(ops, schema=SCHEMA_VERSION):
    return {
        "schema": schema,
        "regen_index": 0,
        "measurements": {
            "m0": {
                "torch": "2.13.0",
                "onnxruntime": "1.26.0",
                "opset": 18,
                "examples": 25,
                "profile": "nightly",
            }
        },
        "ops": {name: {"onnx": rec} for name, rec in ops.items()},
        "specs": {},
    }


def _view(gen, ops, page_rows, aliases=frozenset()):
    return gen.MeasuredOnnxSupport(
        _payload(ops), gen.AliasManager(set(aliases)), set(page_rows)
    )


class TestRowMapping:
    def test_direct_name_maps_to_its_row(self, gen):
        view = _view(gen, {"gelu": _record()}, {"gelu"})
        assert view.grade("gelu") == GRADE_FULL

    def test_alias_is_mapped_onto_canonical_row(self, gen):
        """`absolute` is measured, but the page lists it under `abs`."""
        view = _view(
            gen,
            {"absolute": _record()},
            {"abs"},
            aliases={("absolute", "abs")},
        )
        assert view.grade("abs") == GRADE_FULL

    def test_inplace_variant_merges_into_base_row(self, gen):
        view = _view(gen, {"relu_": _record()}, {"relu"})
        assert view.grade("relu") == GRADE_FULL

    def test_unknown_row_is_untested_not_unsupported(self, gen):
        view = _view(gen, {"gelu": _record()}, {"gelu", "histc"})
        assert view.grade("histc") == GRADE_UNTESTED

    def test_measured_op_absent_from_page_is_reported(self, gen):
        """`_`-prefixed names have no row; they must not vanish silently."""
        view = _view(gen, {"_upsample_nearest_exact2d": _record()}, {"gelu"})
        assert view.unmapped == {"_upsample_nearest_exact2d"}


class TestRowMerging:
    def test_two_names_on_one_row_merge_counters(self, gen):
        """A full and a failing name on the same row give `partial`.

        Grades are re-derived from summed counters rather than picked
        between, so this cannot silently report `full`.
        """
        view = _view(
            gen,
            {
                "relu": _record(export=GRADE_FULL, examples=10, export_ok=10),
                "relu_": _record(export=GRADE_NONE, examples=10, export_ok=0),
            },
            {"relu"},
        )
        assert view.grade("relu") == GRADE_PARTIAL
        assert view.record("relu")["examples"] == 20

    def test_two_full_names_on_one_row_stay_full(self, gen):
        view = _view(
            gen,
            {
                "relu": _record(examples=10, export_ok=10),
                "relu_": _record(examples=5, export_ok=5),
            },
            {"relu"},
        )
        assert view.grade("relu") == GRADE_FULL


class TestAxisReporting:
    def test_not_reached_axes_render_as_dash(self, gen):
        view = _view(
            gen,
            {
                "digamma": _record(
                    export=GRADE_NONE,
                    examples=5,
                    export_ok=0,
                    runtime_reached=0,
                    runtime_ok=0,
                    numerics_reached=0,
                    numerics_match=0,
                )
            },
            {"digamma"},
        )
        assert view.axis("digamma", "runtime") == "-"
        assert view.axis("digamma", "numerics") == "-"

    def test_untested_row_axes_render_as_dash(self, gen):
        view = _view(gen, {}, {"histc"})
        assert view.axis("histc", "runtime") == "-"


class TestSchemaGuard:
    def test_mismatched_schema_is_refused(self, gen):
        """Counters read as zero would regrade `none` into `blocked`."""
        payload = _payload({"gelu": _record()}, schema=SCHEMA_VERSION + 1)
        with pytest.raises(ValueError, match="schema"):
            gen.MeasuredOnnxSupport(payload, gen.AliasManager(set()), {"gelu"})


class TestGlyphs:
    def test_every_grade_has_a_glyph(self, gen):
        from .proptest.onnx_report import GRADE_BLOCKED

        for grade in (
            GRADE_FULL,
            GRADE_PARTIAL,
            GRADE_NONE,
            GRADE_BLOCKED,
            GRADE_UNTESTED,
            gen.DISPLAY_UNTESTED_DOCUMENTED,
        ):
            assert grade in gen.GRADE_GLYPH

    def test_untested_is_not_a_cross(self, gen):
        """A row with no spec must not read as "ONNX cannot do this"."""
        assert gen.GRADE_GLYPH[GRADE_UNTESTED] != gen.GRADE_GLYPH[GRADE_NONE]

    def test_claimed_unverified_is_distinct_from_measured_full(self, gen):
        """`✅*` must not be confusable with a measured `✅`.

        The star is the only thing separating "we exported this 25 times"
        from "a retired doc page said so and we never checked".
        """
        starred = gen.GRADE_GLYPH[gen.DISPLAY_UNTESTED_DOCUMENTED]
        assert starred != gen.GRADE_GLYPH[GRADE_FULL]
        assert starred != gen.GRADE_GLYPH[GRADE_UNTESTED]
        assert "*" in starred


class TestUntestedDocumentedDisplay:
    """The `✅*` state is display-only and must not enter the counts."""

    def test_display_state_is_not_a_report_grade(self, gen):
        from . import proptest

        # It deliberately lives in the generator, not the measurement
        # vocabulary: `onnx_report` only emits things it measured.
        assert not hasattr(proptest.onnx_report, "DISPLAY_UNTESTED_DOCUMENTED")

    def test_measured_counts_exclude_untested_rows(self, gen):
        """`counts()` covers measured rows only, starred or not."""
        view = _view(gen, {"gelu": _record()}, {"gelu", "addmm", "histc"})
        counts = view.counts()
        assert counts[GRADE_FULL] == 1
        assert sum(counts.values()) == 1  # addmm / histc contribute nothing

    def test_filter_modes_separate_the_two_untested_states(self, gen):
        values = [value for value, _label in gen.GRADED_FILTER_MODES]
        assert gen.DISPLAY_UNTESTED_DOCUMENTED in values
        assert GRADE_UNTESTED in values

    def test_filter_script_regex_matches_hyphenated_state(self, gen):
        """`grade-([a-z]+)` would truncate `untested-documented`."""
        import re

        pattern = re.search(
            r"match\(/grade-\(\[([a-z\-\\]+)\]\+\)/\)", gen.FILTER_SCRIPT
        )
        assert pattern is not None, "filter regex not found in FILTER_SCRIPT"
        char_class = pattern.group(1)
        assert "-" in char_class, (
            "filter regex must accept '-' or the two untested states "
            "become indistinguishable in the UI"
        )


class TestTorchVersionDefault:
    def test_defaults_to_installed_torch_major_minor(self, gen):
        import torch

        expected = ".".join(torch.__version__.split(".")[:2])
        assert gen.installed_torch_version() == expected

    def test_default_is_two_components(self, gen):
        assert len(gen.installed_torch_version().split(".")) == 2
