"""Shared types for the proptest op_specs package.

`OpSpec` and `OpSample` are the public types every spec module produces;
`_unary_sample_st` is the only sample-strategy helper actually shared
across multiple op groups (elementwise unary, activations, conv/pool input
generation). Group-specific helpers (binary broadcast, pow variants, the
unary-domain constants) live in their consumer module.
"""

import typing as T
from dataclasses import dataclass

import torch
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import UnaryPrimitive
from ..inputs import Interval, tensor_st
from ..shapes import shape_st


@dataclass(frozen=True)
class OpSample:
    """One concrete forward-call payload drawn by an op strategy."""

    inputs: T.Tuple[torch.Tensor, ...]
    module: torch.nn.Module


@dataclass(frozen=True)
class OpSpec:
    name: str
    sample_st: st.SearchStrategy[OpSample]
    tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE
    # When set, the test driver marks this spec's pytest case as xfail with
    # the given reason. Use for known divergences that have a tracked fix
    # (in t2n or tract) so the bug stays visible in CI without blocking PRs.
    # When the underlying fix lands, removing this field flips the spec
    # back to a normal pass and surfaces any regression.
    xfail_reason: T.Optional[str] = None
    # When True, the spec is also exercised under the dynamic-axes
    # variant of the proptest driver: every rank>=1 input has axis 0
    # marked as a runtime dim. Default False so that adding this knob
    # does not retroactively gate the existing 200+ specs through a
    # path they were not designed for; specs that are confident in the
    # dynamic codegen path opt in explicitly.
    dynamic_axes_compatible: bool = False
    # Optional human-readable note about why a spec opted out, shown
    # by pytest's skip reason. Only meaningful when
    # `dynamic_axes_compatible=False`.
    dynamic_axes_skip_reason: T.Optional[str] = None
    # The `aten::` operator name(s) this spec is testing, without the
    # `aten::` prefix. Used to turn a per-spec result into a per-operator
    # verdict for the support page (`docs/contributing/`), so these must
    # be the names that page lists: it builds its rows from a source grep
    # that drops `_`-prefixed identifiers and merges in-place variants,
    # which means the name torch puts in the trace is not always the row
    # name (`conv2d` traces `aten::_convolution`; `sdpa` traces
    # `aten::scaled_dot_product_attention`).
    #
    # Name only the operator(s) actually under test, not every op the
    # module happens to trace: a spec's failure is attributed to each
    # name listed here, so adding incidental scaffolding ops would
    # smear one op's gap across unrelated rows.
    #
    # `test_proptest_aten_attribution.py` checks each declared name
    # against what the spec really traces, so this stays honest as specs
    # are edited.
    aten_ops: T.Tuple[str, ...] = ()


def _unary_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
    domain: T.Optional[Interval],
    finite: bool = True,
) -> st.SearchStrategy[OpSample]:
    """Build a unary-op sample strategy (rank 0..4, f32, optional domain)."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(tensor_st(shape, torch.float32, finite=finite, domain=domain))
        return OpSample(inputs=(x,), module=UnaryPrimitive(op))

    return _draw()
