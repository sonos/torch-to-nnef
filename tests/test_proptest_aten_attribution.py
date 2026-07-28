r"""Check that every proptest spec declares the aten op it really tests.

`OpSpec.aten_ops` turns a per-spec result into a per-operator verdict for
the support page. Nothing else validates it: a spec whose strategy is
rewritten to exercise a different op would keep its stale declaration and
quietly attribute its results to the wrong row.

So: trace one drawn example per spec and assert each declared name shows
up in the trace, modulo the renames below.

Run with::

    pytest tests/test_proptest_aten_attribution.py -m proptest_onnx -v
"""

import typing as T

import pytest
import torch
from hypothesis import given, settings

from .proptest.op_specs import REGISTRY, OpSample, OpSpec
from .proptest.trace_names import KNOWN_TRACE_RENAMES

pytestmark = pytest.mark.proptest_onnx


def _traced_aten_ops(sample: OpSample) -> T.Set[str]:
    """Return the `aten::*` op names a drawn sample traces to."""
    module = sample.module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, sample.inputs, check_trace=False)
    return {
        node.kind().split("::", 1)[1]
        for node in traced.inlined_graph.nodes()
        if node.kind().startswith("aten::")
    }


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda s: s.name)
def test_spec_declares_the_op_it_traces(spec: OpSpec) -> None:
    """Each name in `spec.aten_ops` is present in the spec's trace."""
    assert spec.aten_ops, (
        f"spec {spec.name!r} declares no `aten_ops`; the support page "
        "cannot attribute its result to an operator"
    )

    # One example is enough: we are checking the declaration against the
    # op the module dispatches to, which does not vary with the drawn
    # shapes or values.
    @settings(max_examples=1, deadline=None, derandomize=True)
    @given(sample=spec.sample_st)
    def _inner(sample: OpSample) -> None:
        traced = _traced_aten_ops(sample)
        for declared in spec.aten_ops:
            # The page merges an in-place variant into its base row, so
            # a row named `uniform` is the only home a traced
            # `aten::uniform_` has. Accept it generically rather than
            # listing every mutating op in KNOWN_TRACE_RENAMES.
            accepted = (
                declared,
                f"{declared}_",
                *KNOWN_TRACE_RENAMES.get(declared, ()),
            )
            assert any(name in traced for name in accepted), (
                f"spec {spec.name!r} declares aten op {declared!r} but "
                f"traces {sorted(traced)}. Either fix `aten_ops` or, if "
                "torch renames/decomposes this op, add it to "
                "KNOWN_TRACE_RENAMES with the reason."
            )

    _inner()
