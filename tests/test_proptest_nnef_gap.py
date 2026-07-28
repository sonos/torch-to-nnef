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

import typing as T

import pytest

from tests.proptest import nnef_gap as nnef_gap_mod
from tests.proptest.nnef_gap import NnefGapMismatch, assert_nnef_gap
from tests.proptest.op_specs import REGISTRY, NnefGap, NnefGapStage, OpSpec
from torch_to_nnef.op.aten import aten_ops_registry

GAP_SPECS: T.Tuple[OpSpec, ...] = tuple(
    spec for spec in REGISTRY if spec.nnef_gap is not None
)


def _registered_aten_ops() -> T.Set[str]:
    """The emitter names, read the way the support page reads them."""
    return set(aten_ops_registry._registry.keys())


def test_gap_specs_exist():
    """Guard the guard: an empty tuple would make every check vacuous."""
    assert GAP_SPECS, "no spec declares `nnef_gap`; did the module move?"


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
