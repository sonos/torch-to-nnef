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

    def test_trailing_underscore_row_is_kept_when_no_base_row_exists(self, gen):
        view = _view(gen, {"set_": _record()}, {"set_"})
        assert view.grade("set_") == GRADE_FULL
        assert not view.unmapped

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
        ):
            assert grade in gen.GRADE_GLYPH

    def test_untested_is_not_a_cross(self, gen):
        """A row with no spec must not read as "ONNX cannot do this"."""
        assert gen.GRADE_GLYPH[GRADE_UNTESTED] != gen.GRADE_GLYPH[GRADE_NONE]

    def test_claimed_unverified_is_not_an_export_glyph(self, gen):
        """A historical claim must not look like a measured export."""
        assert gen.DISPLAY_UNTESTED_DOCUMENTED not in gen.GRADE_GLYPH


class TestUntestedDocumentedDisplay:
    """Historical claims are filter-only and must not enter counts."""

    def test_display_state_is_not_a_report_grade(self, gen):
        from . import proptest

        # It deliberately lives in the generator, not the measurement
        # vocabulary: `onnx_report` only emits things it measured.
        assert not hasattr(proptest.onnx_report, "DISPLAY_UNTESTED_DOCUMENTED")

    def test_measured_counts_exclude_untested_rows(self, gen):
        """`counts()` covers measured rows only, claimed or not."""
        view = _view(gen, {"gelu": _record()}, {"gelu", "addmm", "histc"})
        counts = view.counts()
        assert counts[GRADE_FULL] == 1
        assert sum(counts.values()) == 1  # addmm / histc contribute nothing

    def test_filter_modes_separate_unmeasured_causes(self, gen):
        values = [value for value, _label in gen.GRADED_FILTER_MODES]
        assert gen.DISPLAY_UNTESTED_DOCUMENTED in values
        assert gen.DISPLAY_NO_SPEC_NO_CLAIM in values
        assert GRADE_UNTESTED not in values

    def test_filter_script_handles_filter_only_claims(self, gen):
        """Claimed-only rows are not encoded as export grades."""
        assert "mode === 'untested-documented'" in gen.FILTER_SCRIPT
        assert "mode === 'no-spec-no-claim'" in gen.FILTER_SCRIPT


class TestHeadlineBars:
    """The two tabs sit side by side, so their bars must be one measure.

    A measured-only denominator (`108/108`) renders as a full bar next to
    the binary tab's `130/139` and reads as "ONNX covers more", which is
    the opposite of what the numbers say. And a numerator that drops the
    claimed-but-unverified rows scores our own missing spec coverage as an
    ONNX gap, which understates a competing exporter for our shortfall.
    """

    @staticmethod
    def _summary(gen, **kwargs):
        import io

        view = _view(gen, {"gelu": _record()}, {"gelu", "addmm", "histc"})
        fh = io.StringIO()
        gen._write_measured_summary(  # noqa: SLF001
            "intro",
            view,
            ["gelu", "addmm", "histc"],
            kwargs.get("full_qte", 1),
            kwargs.get("full_core", 1),
            kwargs.get("measured_core", 1),
            kwargs.get("qte_core", 3),
            kwargs.get("untested_documented", 0),
            kwargs.get("untested_documented_core", 0),
            kwargs.get("missing_supported_specs", 0),
            fh,
        )
        return fh.getvalue()

    def test_core_bar_is_over_the_whole_core_opset(self, gen):
        """Not over the measured subset: `1/3`, never `1/1`."""
        out = self._summary(gen, full_core=1, measured_core=1, qte_core=3)
        assert "[=1/3 " in out
        assert "[=1/1 " not in out

    def test_aten_bar_is_over_every_listed_row(self, gen):
        out = self._summary(gen, full_qte=1)
        assert "[=1/3 " in out  # 3 rows listed, 1 graded full

    def test_claimed_rows_are_credited_in_both_bars(self, gen):
        """Historical claims count in bars but are not export grades."""
        out = self._summary(
            gen,
            full_core=1,
            untested_documented_core=1,
            qte_core=5,
            full_qte=1,
            untested_documented=1,
        )
        assert "[=2/5 " in out  # core: 1 measured + 1 claimed
        assert "[=2/3 " in out  # aten: idem, over every listed row
        assert "Both bars also credit 1 historical ONNX claim" in out

    def test_claimed_rows_only_describe_the_bar_they_affect(self, gen):
        """The caption must not say both bars when core claims are zero."""
        out = self._summary(
            gen,
            full_core=1,
            untested_documented_core=0,
            qte_core=5,
            full_qte=1,
            untested_documented=1,
        )
        assert "[=1/5 " in out  # core: no historical claim credited
        assert "[=2/3 " in out  # aten: 1 measured + 1 claimed
        assert "overall bar also credits 1 historical ONNX claim" in out
        assert "core bar credits none" in out
        assert "both bars credit" not in out.lower()

    def test_rows_with_no_claim_stay_out_of_the_numerator(self, gen):
        """Crediting claims must not slide into crediting silence."""
        out = self._summary(gen, full_qte=1, untested_documented=0)
        assert "[=1/3 " in out
        assert "[=3/3 " not in out

    def test_core_bar_comes_before_the_aten_bar(self, gen):
        """Same order as the binary tab, or the bars pair up wrongly."""
        out = self._summary(gen, full_core=2, qte_core=3, full_qte=1)
        assert out.index("[=2/3 ") < out.index("[=1/3 ")

    def test_measured_only_ratio_is_still_reported(self, gen):
        """Crediting claims in the bar must not bury what was measured."""
        out = self._summary(
            gen, full_core=1, measured_core=1, untested_documented_core=1
        )
        assert "1/1 of the measured core operators" in out
        assert "1 core / 1 overall are measured fully exportable" in out

    def test_unmeasured_rows_are_split_by_cause(self, gen):
        out = self._summary(gen, missing_supported_specs=2)
        assert "2 are TractNNEF-supported rows" in out
        assert "excluded from proptest" not in out

    def test_both_sections_build_bars_from_one_helper(self, gen):
        """Structural guard: duplicated strings drift apart tab by tab."""
        import inspect

        source = inspect.getsource(gen.write_operator_support)
        assert source.count("headline_bars(") == 1
        assert "[=" not in source
        assert (
            "[=" not in inspect.getsource(gen._write_measured_summary)  # noqa: SLF001
        )


class TestCrossSectionGapFilter:
    """Supported by ONNX and missing here: the implementation shortlist.

    It spans both tables, so it is the one filter neither section can
    check on its own.
    """

    def test_credits_measured_full(self, gen):
        view = _view(gen, {"gelu": _record()}, {"gelu"})
        assert gen.onnx_credited_names(["gelu"], view, set()) == {"gelu"}

    def test_credits_claimed_but_unmeasured(self, gen):
        """Same generous rule as the headline bars."""
        view = _view(gen, {}, {"addmm"})
        assert gen.onnx_credited_names(["addmm"], view, {"addmm"}) == {"addmm"}

    def test_measurement_overrides_the_claim(self, gen):
        """A claim we disproved must not put the op on the shortlist."""
        view = _view(
            gen,
            {"histc": _record(export=GRADE_NONE, examples=5, export_ok=0)},
            {"histc"},
        )
        assert gen.onnx_credited_names(["histc"], view, {"histc"}) == set()

    def test_partial_is_not_credited(self, gen):
        view = _view(
            gen,
            {
                "relu": _record(examples=10, export_ok=10),
                "relu_": _record(export=GRADE_NONE, examples=10, export_ok=0),
            },
            {"relu"},
        )
        assert gen.onnx_credited_names(["relu"], view, {"relu"}) == set()

    def test_silence_is_not_credited(self, gen):
        view = _view(gen, {}, {"histc"})
        assert gen.onnx_credited_names(["histc"], view, set()) == set()

    def test_falls_back_to_the_listing_without_measurements(self, gen):
        assert gen.onnx_credited_names(["addmm"], None, {"addmm"}) == {"addmm"}

    def test_radio_value_matches_the_script_branch(self, gen):
        """Drift here silently filters to an empty table."""
        assert f"'{gen.CROSS_GAP_MODE}'" in gen.FILTER_SCRIPT
        assert f"'{gen.CROSS_OK_CLASS}'" in gen.FILTER_SCRIPT

    def test_marker_class_is_not_read_as_a_grade(self, gen):
        """`cross-ok` next to `grade-full` must not confuse the matcher."""
        import re

        match = re.search(r"grade-\(\[[a-z\-\\]+\]\+\)", gen.FILTER_SCRIPT)
        assert match is not None
        pattern = re.compile(match.group(0))
        assert pattern.search(f"op-row {gen.CROSS_OK_CLASS}") is None


class TestFilterModeLabels:
    """Radios follow the data: counted, and absent when they select none."""

    def test_empty_modes_are_dropped(self, gen):
        """`blocked` is 0 in the normal case; the radio must not show."""
        modes = gen.counted_modes(
            gen.GRADED_FILTER_MODES, {GRADE_FULL: 3}, 3, 0
        )
        values = [value for value, _label in modes]
        assert "blocked" not in values
        assert GRADE_FULL in values

    def test_labels_carry_their_count(self, gen):
        modes = gen.counted_modes(
            ((GRADE_FULL, "Exports fully"),), {GRADE_FULL: 7}, 7, 0
        )
        assert modes == ((GRADE_FULL, "Exports fully (7)"),)

    def test_all_counts_every_row(self, gen):
        modes = gen.counted_modes((("all", "All"),), {GRADE_FULL: 2}, 9, 0)
        assert modes == (("all", "All (9)"),)

    def test_binary_labels_map_onto_the_grades_they_select(self, gen):
        """`supported` selects `full` rows, `unsupported` selects `none`."""
        modes = gen.counted_modes(
            gen.BINARY_FILTER_MODES,
            {GRADE_FULL: 4, GRADE_NONE: 6},
            10,
            0,
        )
        assert modes == (
            ("all", "All (10)"),
            ("supported", "Supported only (4)"),
            ("unsupported", "Unsupported only (6)"),
        )

    def test_cross_gap_uses_its_own_tally(self, gen):
        """It counts an intersection, not rows of one display state."""
        modes = gen.counted_modes(
            ((gen.CROSS_GAP_MODE, "Gap vs ONNX"),), {}, 10, 3
        )
        assert modes == ((gen.CROSS_GAP_MODE, "Gap vs ONNX (3)"),)

    def test_extra_filters_use_their_own_tallies(self, gen):
        modes = gen.counted_modes(
            ((gen.DISPLAY_MISSING_SUPPORTED_SPEC, "Missing supported spec"),),
            {},
            10,
            0,
            {gen.DISPLAY_MISSING_SUPPORTED_SPEC: 4},
        )
        assert modes == (
            (gen.DISPLAY_MISSING_SUPPORTED_SPEC, "Missing supported spec (4)"),
        )

    def test_cross_gap_disappears_when_nothing_is_missing(self, gen):
        modes = gen.counted_modes(
            ((gen.CROSS_GAP_MODE, "Gap vs ONNX"),), {}, 10, 0
        )
        assert modes == ()

    def test_no_label_is_a_bare_none(self, gen):
        """A bare `None` next to `All` reads as "select nothing"."""
        labels = [label for _value, label in gen.GRADED_FILTER_MODES]
        assert "None" not in labels
        assert "Never exports" in labels
        assert "No spec, no claim" in labels

    def test_blocked_label_says_blocked_by_what(self, gen):
        """It is a `torch.export` failure, never an ONNX verdict."""
        labels = {value: label for value, label in gen.GRADED_FILTER_MODES}
        assert labels["blocked"] == "Blocked before ONNX"


class TestColumnLegend:
    def test_legend_is_its_own_admonition(self, gen):
        legend = gen._measured_onnx_legend("http://x", True)  # noqa: SLF001
        assert legend.startswith('!!! info "')

    def test_legend_body_is_indented_for_the_tabbed_block(self, gen):
        """`print_t` adds 4; the body needs 4 more or it leaves the box."""
        legend = gen._measured_onnx_legend("http://x", True)  # noqa: SLF001
        body = [ln for ln in legend.split("\n")[1:] if ln.strip()]
        assert body and all(ln.startswith("    ") for ln in body)

    def test_every_glyph_is_documented(self, gen):
        legend = gen._measured_onnx_legend("http://x", True)  # noqa: SLF001
        for glyph in gen.GRADE_GLYPH.values():
            assert glyph in legend

    def test_intro_no_longer_carries_the_legend(self, gen):
        view = _view(gen, {"gelu": _record()}, {"gelu"})
        intro = gen._measured_onnx_section_msg(view)  # noqa: SLF001
        assert "`export` column" not in intro
        assert "documented" not in intro

    def test_legend_documents_spec_coverage(self, gen):
        legend = gen._measured_onnx_legend("http://x", True)  # noqa: SLF001
        assert "`spec coverage`" in legend
        assert "missing a direct proptest spec" in legend


class TestTorchVersionDefault:
    def test_defaults_to_installed_torch_major_minor(self, gen):
        import torch

        expected = ".".join(torch.__version__.split(".")[:2])
        assert gen.installed_torch_version() == expected

    def test_default_is_two_components(self, gen):
        assert len(gen.installed_torch_version().split(".")) == 2
