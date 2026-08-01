"""Keep declared t2n translation gaps honest.

`OpSpec.nnef_gap` marks a spec whose operator we cannot export, so the
ONNX sweep can measure it anyway. The danger with any such marker is that
it outlives the gap: an emitter lands, and the spec goes on reporting the
operator as out of reach.

`tests/test_primitive_proptest.py` closes that by attempting the real
export, but it only runs in the `proptest` env. The checks here are
static, so they run in the default suite and fail the moment a declared
gap contradicts the registry.

Run with::

    pytest tests/test_proptest_nnef_gap.py -v
"""

import json
import re
import typing as T
from pathlib import Path

import pytest

from tests.proptest import nnef_gap as nnef_gap_mod
from tests.proptest.nnef_gap import NnefGapMismatch, assert_nnef_gap
from tests.proptest.op_specs import REGISTRY, NnefGap, NnefGapStage, OpSpec
from tests.proptest.op_specs.untranslated import EXCLUDED
from tests.proptest.trace_names import registry_lookup_names
from torch_to_nnef.op.aten import aten_ops_registry

GAP_SPECS: T.Tuple[OpSpec, ...] = tuple(
    spec for spec in REGISTRY if spec.nnef_gap is not None
)

ONNX_ARTIFACT = (
    Path(__file__).parent.parent
    / "docs"
    / "contributing"
    / "onnx_support_measured.json"
)

SUPPORT_PAGE = (
    Path(__file__).parent.parent
    / "docs"
    / "contributing"
    / "supported_operators.md"
)

#: One generated table row: the class list carries the verdict, and the
#: second cell carries the operator name.
_ROW_RE = re.compile(r'<tr class="([^"]*)">(.*)</tr>')
_CELL_RE = re.compile(r"<td>(.*?)</td>")
_TAG_RE = re.compile(r"<[^>]+>")


def _registered_aten_ops() -> T.Set[str]:
    """The emitter names, read the way the support page reads them."""
    return set(aten_ops_registry._registry.keys())


#: Sanity floor for the page parse. The listing has ~580 rows and only
#: grows with each torch release, so anything near zero means the parse
#: broke rather than the page changing.
_MIN_PARSED_ROWS = 100

#: The column header unique to the TractNNEF table (the ONNX one has
#: `export` / `runtime` / `numerics` instead).
_TRACT_HEADER = "export&amp;run"
_ONNX_COVERAGE_HEADER = "spec coverage"

#: The row classes that table is built from. Both must occur, or the
#: parse is reading a vocabulary the generator no longer emits.
_EXPECTED_ROW_CLASSES = ("supported", "unsupported")


def _tract_table_lines() -> T.List[str]:
    """Just the `TractNNEF` table's lines.

    The page renders two tables. Only the first is about our own support;
    the second is the ONNX measurement, and it reuses the same
    `unsupported` row class whenever the generator runs *without*
    `--onnx-report`. Scanning the whole file would then union the two,
    and every coverage check below would fail for reasons that have
    nothing to do with the contributor's change.
    """
    lines = SUPPORT_PAGE.read_text().splitlines()
    table_starts = [i for i, line in enumerate(lines) if "<table" in line]
    assert table_starts, (
        f"no <table> found in {SUPPORT_PAGE.name}: the generator's output "
        "shape changed, and every page-based check here is now blind"
    )
    # Identified by its header, not by being first. Both tables carry the
    # same ~580 rows, so if the generator ever emits the ONNX tab first,
    # a positional pick would hand every check below a table about a
    # different exporter and no row count would notice.
    for index, start in enumerate(table_starts):
        end = (
            table_starts[index + 1]
            if index + 1 < len(table_starts)
            else len(lines)
        )
        block = lines[start:end]
        if any(_TRACT_HEADER in line for line in block):
            return block
    raise AssertionError(
        f"no table in {SUPPORT_PAGE.name} has a {_TRACT_HEADER!r} column, "
        "so the TractNNEF table cannot be told apart from the ONNX one. "
        "The generator's header changed; update `_TRACT_HEADER`."
    )


def _onnx_table_lines() -> T.List[str]:
    """Just the measured `ONNX` table's lines."""
    lines = SUPPORT_PAGE.read_text().splitlines()
    table_starts = [i for i, line in enumerate(lines) if "<table" in line]
    assert table_starts, (
        f"no <table> found in {SUPPORT_PAGE.name}: the generator's output "
        "shape changed, and every page-based check here is now blind"
    )
    for index, start in enumerate(table_starts):
        end = (
            table_starts[index + 1]
            if index + 1 < len(table_starts)
            else len(lines)
        )
        block = lines[start:end]
        if any(_ONNX_COVERAGE_HEADER in line for line in block):
            return block
    raise AssertionError(
        f"no table in {SUPPORT_PAGE.name} has a "
        f"{_ONNX_COVERAGE_HEADER!r} column, so ONNX unmeasured rows "
        "cannot be classified. Regenerate the support page."
    )


def _row_classes_by_name(
    table_lines: T.Iterable[str],
) -> T.Dict[str, T.Set[str]]:
    rows: T.Dict[str, T.Set[str]] = {}
    parsed = 0
    for line in table_lines:
        match = _ROW_RE.search(line)
        if match is None:
            continue
        parsed += 1
        cells = _CELL_RE.findall(match.group(2))
        if len(cells) > 1:
            rows[_TAG_RE.sub("", cells[1])] = set(match.group(1).split())
    assert parsed >= _MIN_PARSED_ROWS, (
        f"parsed only {parsed} rows from {SUPPORT_PAGE.name} (expected at "
        f"least {_MIN_PARSED_ROWS}). `_ROW_RE` no longer matches what "
        "`generate_support_page.py` emits, so the coverage checks in this "
        "module are blind rather than passing."
    )
    return rows


def _page_unsupported_ops() -> T.Set[str]:
    """Row names the committed support page marks unsupported.

    Read from the generated page rather than recomputed, so this checks
    what a reader actually sees.
    """
    names, parsed, seen_classes = set(), 0, set()
    for line in _tract_table_lines():
        match = _ROW_RE.search(line)
        if match is None:
            continue
        parsed += 1
        classes = match.group(1).split()
        seen_classes.update(classes)
        if "unsupported" not in classes:
            continue
        cells = _CELL_RE.findall(match.group(2))
        if len(cells) > 1:
            names.add(_TAG_RE.sub("", cells[1]))
    # Two ways the parse can go blind, and the row count only catches
    # the first: the markup stops matching `_ROW_RE` at all, or the row
    # *classes* get renamed so every row parses and none is recognised.
    # Either way the coverage checks below would pass by describing an
    # empty page, which is the outcome they exist to prevent.
    assert parsed >= _MIN_PARSED_ROWS, (
        f"parsed only {parsed} rows from {SUPPORT_PAGE.name} (expected at "
        f"least {_MIN_PARSED_ROWS}). `_ROW_RE` no longer matches what "
        "`generate_support_page.py` emits, so the coverage checks in this "
        "module are blind rather than passing."
    )
    missing = [c for c in _EXPECTED_ROW_CLASSES if c not in seen_classes]
    assert not missing, (
        f"parsed {parsed} rows from {SUPPORT_PAGE.name} but none carries "
        f"the {missing} class. The generator renamed its row classes, so "
        "this module reads every row as supported and its coverage "
        "checks are blind rather than passing."
    )
    return names


def _page_supported_ops() -> T.Set[str]:
    """Row names the committed support page marks supported."""
    rows = _row_classes_by_name(_tract_table_lines())
    supported = {
        name
        for name, classes in rows.items()
        if "supported" in classes and "unsupported" not in classes
    }
    assert supported, (
        f"no supported rows parsed from {SUPPORT_PAGE.name}; the support "
        "page checks are blind rather than passing"
    )
    return supported


def test_gap_specs_exist():
    """Guard the guard: an empty tuple would make every check vacuous."""
    assert GAP_SPECS, "no spec declares `nnef_gap`; did the module move?"


def test_every_unsupported_row_is_specced_or_excluded():
    """No unsupported operator may sit in neither list.

    This is what makes the gap package a *map* rather than a sample. An
    operator we cannot translate is either measured (so the page can say
    what ONNX does with it) or explicitly excluded with a reason. Adding
    a row to the page without doing one of those fails here, which is
    the only moment anyone is looking.
    """
    if not SUPPORT_PAGE.exists():  # pragma: no cover - page always shipped
        pytest.skip("generated support page is not in this checkout")
    specced = {op for spec in GAP_SPECS for op in spec.aten_ops}
    unclassified = _page_unsupported_ops() - specced - set(EXCLUDED)
    assert not unclassified, (
        "these operators are unsupported on the support page but neither "
        f"measured nor excluded: {sorted(unclassified)}. Add a spec in "
        "the themed module it belongs in, or an entry in its `EXCLUDED` "
        "map saying why measuring it is not worth it."
    )


def test_excluded_entries_have_no_spec():
    """A row cannot be both measured and written off."""
    specced = {op for spec in GAP_SPECS for op in spec.aten_ops}
    both = specced & set(EXCLUDED)
    assert not both, (
        f"{sorted(both)} appear in both the gap specs and `EXCLUDED`; "
        "drop the exclusion, since a measured operator is not excluded"
    )


def test_excluded_entries_are_not_in_the_support_tables():
    """An exclusion must be outside the comparison denominator.

    `EXCLUDED` is reserved for names that cannot be attributed to a
    proptest graph target. They belong in the appendix, not as `-` rows
    inside either support table.
    """
    if not SUPPORT_PAGE.exists():  # pragma: no cover - page always shipped
        pytest.skip("generated support page is not in this checkout")
    tract_rows = set(_row_classes_by_name(_tract_table_lines()))
    onnx_rows = set(_row_classes_by_name(_onnx_table_lines()))
    stale = sorted(set(EXCLUDED) & (tract_rows | onnx_rows))
    assert not stale, (
        f"{stale} are listed in `EXCLUDED` but still appear in the support "
        "tables. Either drop the exclusion and add a spec, or filter the "
        "row into the appendix."
    )


def test_excluded_entries_are_documented_in_the_appendix():
    """Filtered rows should remain visible with the other exclusions."""
    if not SUPPORT_PAGE.exists():  # pragma: no cover - page always shipped
        pytest.skip("generated support page is not in this checkout")
    appendix = SUPPORT_PAGE.read_text()
    missing = sorted(name for name in EXCLUDED if f"`{name}`" not in appendix)
    assert not missing, (
        f"{missing} are listed in `EXCLUDED` but do not appear in the "
        "support-page appendix. Readers need to know why the row left "
        "the comparison tables."
    )


def test_page_credits_every_registered_emitter():
    """The committed page must agree with the live emitter registry.

    The step everyone forgets: you add an emitter, the tests go green,
    and the page still says the operator is unsupported until somebody
    regenerates it. Nothing else notices, because the page is a build
    artifact that only changes when a human runs the generator.

    Compared against the registry rather than by regenerating, so this
    needs no network and runs in the fast suite. Only registry keys that
    are page rows are checked: the registry also holds names the page
    drops (`_`-prefixed spellings, in-place variants it merges), and
    those legitimately have no row of their own.
    """
    if not SUPPORT_PAGE.exists():  # pragma: no cover - page always shipped
        pytest.skip("generated support page is not in this checkout")
    rows_marked_unsupported = _page_unsupported_ops()
    registered = _registered_aten_ops()
    stale = sorted(registered & rows_marked_unsupported)
    assert not stale, (
        f"{stale} have an emitter but the committed support page still "
        "lists them as unsupported. Regenerate it:\n"
        "  python docs/contributing/generate_support_page.py \\\n"
        "      --onnx-report docs/contributing/onnx_support_measured.json"
    )


def test_onnx_unmeasured_rows_say_why_they_have_no_spec():
    """A `-` row must not leave the missing-spec cause implicit."""
    if not SUPPORT_PAGE.exists():  # pragma: no cover - page always shipped
        pytest.skip("generated support page is not in this checkout")
    supported_rows = _page_supported_ops()
    onnx_rows = _row_classes_by_name(_onnx_table_lines())

    unclassified = []
    stale_supported_markers = []
    for name, classes in onnx_rows.items():
        unmeasured = "grade-untested" in classes
        if not unmeasured:
            continue
        has_missing_marker = "missing-supported-spec" in classes
        has_cause = bool(
            {
                "missing-supported-spec",
                "untested-documented",
                "no-spec-no-claim",
            }
            & classes
        )
        if not has_cause:
            unclassified.append(name)
        if name in supported_rows and not has_missing_marker:
            stale_supported_markers.append(name)

    assert not unclassified, (
        "these ONNX rows have no measurement but the page does not say "
        f"why: {sorted(unclassified)}"
    )
    assert not stale_supported_markers, (
        "these TractNNEF-supported rows have no ONNX measurement but are "
        "not marked as missing supported specs: "
        f"{sorted(stale_supported_markers)}"
    )


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_gap_spec_declares_aten_ops(spec: OpSpec):
    """A gap nobody can attribute to an operator is not worth recording."""
    assert spec.aten_ops, (
        f"gap spec {spec.name!r} declares no `aten_ops`, so neither the "
        "support page nor the ONNX sweep can attribute it"
    )


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_gap_registry_state_matches_the_declared_stage(spec: OpSpec):
    """The registry must agree with what the gap says about it.

    The cheap half of the anti-rot check: it catches the exact moment
    someone registers an operator, without exporting anything.

    Which way it points depends on `emitter_registered`, because the
    stages disagree about whether an emitter exists. `no-emitter` means
    nothing is registered. `tract-error` is the opposite: NNEF has to be
    written before tract can refuse it, so an emitter *must* exist, and a
    blanket "must be absent" rule would make that stage impossible to
    declare. `export-error` is either, depending on whether the pipeline
    died before the lookup or an emitter rejected the configuration.

    Checking every stage, not just `no-emitter`, is what keeps a spec
    from passing on someone else's failure: the `linalg` solvers
    factorize first, so `lu_solve` would keep failing on `lu_factor`
    long after `lu_solve` itself was implemented.

    Lookup goes through `registry_lookup_names`, because `aten_ops`
    carries the support-page row name while an emitter registers under
    the name torch dispatches (`gamma` arrives as `_standard_gamma`).
    """
    gap = spec.nnef_gap
    assert gap is not None
    registered = _registered_aten_ops()
    found = {
        op: [k for k in registry_lookup_names(op) if k in registered]
        for op in spec.aten_ops
    }

    if gap.emitter_registered:
        missing = sorted(op for op, keys in found.items() if not keys)
        assert not missing, (
            f"spec {spec.name!r} declares a `{gap.stage.value}` gap with "
            f"`emitter_registered=True`, but {missing} has no emitter. "
            "Either the emitter was removed (drop the flag, and the "
            "stage is now `no-emitter`), or the flag was set by mistake."
        )
        return

    hits = {op: keys for op, keys in found.items() if keys}
    assert not hits, (
        f"spec {spec.name!r} declares a `{gap.stage.value}` gap but an "
        f"emitter is registered: {hits}. Either the gap is closed, in "
        "which case delete `nnef_gap=...` from the spec (it already sits "
        "in the themed module it belongs in), give it a `tolerance` if "
        "the comparison needs one, and regenerate the support page; or "
        "the failure is downstream of that emitter, in which case set "
        "`emitter_registered=True` on the gap."
    )


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_tract_error_gaps_declare_their_emitter(spec: OpSpec):
    """`tract-error` is only reachable once an emitter exists.

    The stage means NNEF was written and tract then refused it, which
    cannot happen without a translation. Catching the contradiction here
    is friendlier than letting the registry check above fail with a
    message about a gap that has closed.
    """
    gap = spec.nnef_gap
    assert gap is not None
    if gap.stage is not NnefGapStage.TRACT_ERROR:
        pytest.skip(f"stage is {gap.stage.value}")
    assert gap.emitter_registered, (
        f"spec {spec.name!r} declares `tract-error` without "
        "`emitter_registered=True`, but tract can only refuse a graph we "
        "managed to write, which needs an emitter"
    )


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_gap_and_xfail_are_exclusive(spec: OpSpec):
    """`xfail_reason` means a shipped translation that disagrees.

    A spec cannot both ship a translation and lack one, and the tract
    driver checks `xfail_reason` first, so allowing both would silently
    disable the gap assertion.
    """
    assert spec.xfail_reason is None, (
        f"spec {spec.name!r} sets both `nnef_gap` and `xfail_reason`; "
        "the second would shadow the first in the tract driver"
    )


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_gap_reason_is_informative(spec: OpSpec):
    """The reason is what a reader gets when the assertion fires."""
    assert spec.nnef_gap is not None
    assert len(spec.nnef_gap.reason) > 20, (
        f"spec {spec.name!r} has a `nnef_gap.reason` too short to explain "
        "anything to whoever hits the failure"
    )


class _FakeGapObserver:
    """Stand in for a real export attempt in the assertion-logic tests."""

    def __init__(self, stage: T.Optional[NnefGapStage]):
        self.stage = stage

    def __call__(self, model, inputs, inference_target):
        return self.stage


def _assert_with_observed(
    observed: T.Optional[NnefGapStage],
    declared: NnefGapStage,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        nnef_gap_mod, "observe_nnef_gap", _FakeGapObserver(observed)
    )
    assert_nnef_gap(
        gap=NnefGap(stage=declared, reason="a reason long enough to pass"),
        spec_name="fake-spec",
        model=None,
        inputs=(),
        inference_target=None,
    )


def test_assert_nnef_gap_accepts_the_declared_stage(monkeypatch):
    _assert_with_observed(
        NnefGapStage.NO_EMITTER, NnefGapStage.NO_EMITTER, monkeypatch
    )


def test_assert_nnef_gap_rejects_a_closed_gap(monkeypatch):
    """Nothing failed, so the operator now exports and runs."""
    with pytest.raises(NnefGapMismatch, match="gap is closed"):
        _assert_with_observed(None, NnefGapStage.NO_EMITTER, monkeypatch)


def test_assert_nnef_gap_rejects_a_moved_failure(monkeypatch):
    """Still broken, but not where the spec says it is."""
    with pytest.raises(NnefGapMismatch, match="failed at `tract-error`"):
        _assert_with_observed(
            NnefGapStage.TRACT_ERROR, NnefGapStage.NO_EMITTER, monkeypatch
        )


def test_artifact_spec_ids_match_the_catalog():
    """The measured artifact must name specs that still exist.

    The artifact is the documented way back from a grade to the spec that
    produced it, and nothing else checks that link: the other guards
    compare operator names, which survive a spec rename untouched. Renaming
    a spec without re-running the sweep leaves every grade pointing at an
    id that collects nothing.
    """
    if not ONNX_ARTIFACT.exists():  # pragma: no cover - artifact is shipped
        pytest.skip("measured ONNX artifact is not in this checkout")
    recorded = set(json.loads(ONNX_ARTIFACT.read_text()).get("specs", {}))
    assert recorded, "artifact has no `specs` section to check"
    unknown = sorted(recorded - {spec.name for spec in REGISTRY})
    assert not unknown, (
        f"{len(unknown)} spec ids in {ONNX_ARTIFACT.name} no longer exist "
        f"in the catalog (e.g. {unknown[:5]}). Re-run the sweep so the "
        "grades point back at real specs:\n"
        "  tox -e proptest_onnx"
    )
