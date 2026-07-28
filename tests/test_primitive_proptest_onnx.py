r"""Measure PyTorch's ONNX export support across the proptest catalog.

This is a *measurement* run, not a pass/fail suite. An op that ONNX cannot
export is a fact to record, not a torch-to-nnef regression, so these tests
stay green on ONNX-side failures: they only fail when this harness itself
breaks (a missing dependency, a bug in the collector).

That is also why the outcome is accumulated instead of asserted.
"Partial support", meaning an op exports at f32 rank-2 but raises at f16
or rank 0, is only visible if every drawn example is measured, and
`@given` would abort the whole spec on its first failure.

Results land in a graded per-operator artifact consumed by
`docs/contributing/generate_support_page.py`. Run with::

    pytest tests/test_primitive_proptest_onnx.py -m proptest_onnx \
        --onnx-report=docs/contributing/onnx_support_measured.json

`--onnx-reuse` / `--onnx-no-reuse` control whether previously measured
`full` grades are carried over; see `tests/proptest/onnx_report.py`.
"""

import dataclasses
import tempfile
from pathlib import Path

import pytest
from hypothesis import given

from .proptest.onnx_backend import measure_example
from .proptest.op_specs import REGISTRY, OpSample, OpSpec

pytestmark = pytest.mark.proptest_onnx


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda s: s.name)
def test_onnx_support(spec: OpSpec, onnx_sweep) -> None:
    """Record what ONNX export does with every example this spec draws."""
    assert spec.aten_ops, (
        f"spec {spec.name!r} declares no `aten_ops`, so its result cannot "
        "be attributed to an operator; see OpSpec.aten_ops"
    )
    if onnx_sweep.reuse and onnx_sweep.reuse.reuse_for(
        spec.name, spec.aten_ops
    ):
        pytest.skip(
            "carried over from a prior run: every declared op graded "
            "`full` under this exact environment fingerprint"
        )

    # `xfail_reason` marks a tract/t2n divergence and `nnef_gap` marks a
    # translation we do not ship. Neither says anything about ONNX, and
    # skipping either would leave a hole in the support data, so these
    # specs are measured like any other. `nnef_gap` specs exist for this
    # sweep specifically: without them the measured population would be
    # exactly the set of operators t2n already handles.
    config = onnx_sweep.config
    if spec.nondeterministic:
        # Two runs of an RNG op disagree by definition, so a numerics
        # verdict here would report the definition rather than the
        # exporter. Export and runtime stay measured.
        config = dataclasses.replace(config, check_numerics=False)

    @given(sample=spec.sample_st)
    def _inner(sample: OpSample) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outcome = measure_example(
                model=sample.module,
                inputs=sample.inputs,
                workdir=Path(tmpdir),
                config=config,
            )
        onnx_sweep.record(spec.name, spec.aten_ops, outcome)

    _inner()
