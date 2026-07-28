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

import re
import typing as T
from pathlib import Path

import pytest

from tests.proptest import nnef_gap as nnef_gap_mod
from tests.proptest.nnef_gap import NnefGapMismatch, assert_nnef_gap
from tests.proptest.op_specs import REGISTRY, NnefGap, NnefGapStage, OpSpec
from tests.proptest.op_specs.gaps import EXCLUDED
from torch_to_nnef.op.aten import aten_ops_registry

GAP_SPECS: T.Tuple[OpSpec, ...] = tuple(
    spec for spec in REGISTRY if spec.nnef_gap is not None
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


def _page_unsupported_ops() -> T.Set[str]:
    """Row names the committed support page marks unsupported.

    Read from the generated page rather than recomputed, so this checks
    what a reader actually sees.
    """
    names = set()
    for line in SUPPORT_PAGE.read_text().splitlines():
        match = _ROW_RE.search(line)
        if match is None or "unsupported" not in match.group(1).split():
            continue
        cells = _CELL_RE.findall(match.group(2))
        if len(cells) > 1:
            names.add(_TAG_RE.sub("", cells[1]))
    return names


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
        "`tests/proptest/op_specs/gaps/`, or an entry in its `EXCLUDED` "
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


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_gap_spec_declares_aten_ops(spec: OpSpec):
    """A gap nobody can attribute to an operator is not worth recording."""
    assert spec.aten_ops, (
        f"gap spec {spec.name!r} declares no `aten_ops`, so neither the "
        "support page nor the ONNX sweep can attribute it"
    )


@pytest.mark.parametrize("spec", GAP_SPECS, ids=lambda s: s.name)
def test_no_emitter_gaps_are_absent_from_the_registry(spec: OpSpec):
    """A `no-emitter` gap must not have an emitter.

    This is the cheap half of the anti-rot check: it catches the exact
    moment someone registers the operator, without exporting anything.

    Only `NO_EMITTER` is checked. The other stages describe failures
    downstream of a registered emitter (or, for `EXPORT_ERROR`, of the
    constant-folding pass that runs before the lookup), so the registry
    says nothing about them.
    """
    assert spec.nnef_gap is not None
    if spec.nnef_gap.stage is not NnefGapStage.NO_EMITTER:
        pytest.skip(f"stage is {spec.nnef_gap.stage.value}")
    registered = _registered_aten_ops() & set(spec.aten_ops)
    assert not registered, (
        f"spec {spec.name!r} declares a `no-emitter` gap but "
        f"{sorted(registered)} now has one. The gap is closed: drop "
        "`nnef_gap`, let the spec assert real agreement with tract, and "
        "regenerate the support page."
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
