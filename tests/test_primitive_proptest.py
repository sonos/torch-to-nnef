r"""Hypothesis-driven property tests for torch-to-nnef primitives.

This file is intentionally tiny: the heavy lifting lives in
`tests.proptest` (strategies, joint composites, comparator, op registry).
Each entry in :data:`tests.proptest.op_specs.REGISTRY` becomes one
parametrized pytest case here, and hypothesis sweeps shapes/dtypes/values
within the spec's strategy.

Run with::

    pytest tests/test_primitive_proptest.py -m proptest -v \
        --hypothesis-show-statistics
"""

import pytest
from hypothesis import given

from torch_to_nnef.inference_target import TractNNEF

from .proptest.comparator import assert_outputs_close_nan_aware
from .proptest.op_specs import REGISTRY, OpSample, OpSpec

pytestmark = pytest.mark.proptest


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda s: s.name)
def test_op_property(spec: OpSpec) -> None:
    """Run hypothesis examples against tract for one op spec."""
    if spec.xfail_reason is not None:
        # Imperative xfail: the spec is known-bad against the current
        # tract / t2n. We skip execution rather than running and matching
        # the failure: subprocess noise from a guaranteed-fail run is
        # not worth the signal. When the upstream fix lands, removing
        # `xfail_reason` flips the spec back to a normal pass.
        pytest.xfail(spec.xfail_reason)
    target = TractNNEF.latest()

    @given(sample=spec.sample_st)
    def _inner(sample: OpSample) -> None:
        assert_outputs_close_nan_aware(
            model=sample.module,
            inputs=sample.inputs,
            inference_target=target,
            tolerance=spec.tolerance,
        )

    _inner()


def _dynamic_axes_for_inputs(inputs):
    """Mark axis 0 of every rank>=1 input as a runtime dim.

    Inputs that share the same axis-0 size in the drawn sample share
    the same symbolic dim name (`d_axis0_<size>`); this matches the
    tracing constraint that "these two dims are equal" so ops like
    `reshape_as` / `dot` whose semantics require axis-0 equality keep
    that fact visible to tract's shape inference. Inputs of different
    sizes get different names, mirroring the "these dims are unrelated"
    case (e.g. `mv`'s (M, K) and (K,) where M != K).
    """
    config = {}
    for i, t in enumerate(inputs):
        if t.ndim >= 1:
            config[f"input_{i}"] = {0: f"d_axis0_size{int(t.shape[0])}"}
    return config


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda s: s.name)
def test_op_property_dynamic(spec: OpSpec) -> None:
    """Same as `test_op_property` but with axis 0 marked dynamic.

    Specs opt in via `dynamic_axes_compatible=True`. The default is
    False so this driver only exercises ops that have been deliberately
    audited for dynamic-axes correctness.
    """
    if spec.xfail_reason is not None:
        pytest.xfail(spec.xfail_reason)
    if not spec.dynamic_axes_compatible:
        pytest.skip(
            spec.dynamic_axes_skip_reason
            or "spec opted out of the dynamic-axes proptest variant"
        )
    base_target = TractNNEF.latest()

    @given(sample=spec.sample_st)
    def _inner(sample: OpSample) -> None:
        target = base_target.with_dynamic_axes(
            _dynamic_axes_for_inputs(sample.inputs)
        )
        assert_outputs_close_nan_aware(
            model=sample.module,
            inputs=sample.inputs,
            inference_target=target,
            tolerance=spec.tolerance,
        )

    _inner()
