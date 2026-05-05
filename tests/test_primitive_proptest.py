r"""Hypothesis-driven property tests for torch-to-nnef primitives.

This file is intentionally tiny: the heavy lifting lives in
``tests.proptest`` (strategies, joint composites, comparator, op registry).
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
        # the failure -- subprocess noise from a guaranteed-fail run is
        # not worth the signal. When the upstream fix lands, removing
        # ``xfail_reason`` flips the spec back to a normal pass.
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
