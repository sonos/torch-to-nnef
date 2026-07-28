"""Shared types for the proptest op_specs package.

`OpSpec` and `OpSample` are the public types every spec module produces;
`_unary_sample_st` is the only sample-strategy helper genuinely shared
across multiple op groups (elementwise unary, activations, conv/pool input
generation). Group-specific helpers (binary broadcast, pow variants, the
unary-domain constants) live in their consumer module.
"""

import enum
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


class NnefGapStage(enum.Enum):
    """Where a spec is expected to fail on the way to a running NNEF.

    The stage is asserted, not just declared, so a spec cannot keep
    claiming a gap that has since been closed.
    """

    #: No emitter is registered for the operator at all
    #: (`T2NErrorMissingOpEmitter`). This is what the support page counts
    #: as unsupported.
    NO_EMITTER = "no-emitter"
    #: An emitter exists but refuses what this spec draws, or the export
    #: pipeline raises some other `T2NError` on the way out.
    EXPORT_ERROR = "export-error"
    #: NNEF is written, and tract then declines to load or run it.
    TRACT_ERROR = "tract-error"


@dataclass(frozen=True)
class NnefGap:
    """A translation we do not ship, recorded as an expected failure.

    Marks a spec that exists to *measure* rather than to guard: the op is
    out of reach for t2n today, so the tract driver asserts the failure
    instead of asserting agreement, while the ONNX sweep measures it like
    any other spec. Without such specs the ONNX column can only ever be
    measured where t2n already succeeds, which biases the comparison in
    our favour precisely where we are weakest.
    """

    stage: NnefGapStage
    #: Why it cannot be translated, in one line. Shown in the assertion
    #: failure when the gap closes, so make it the sentence a reader
    #: needs to decide whether closing it is now the right move.
    reason: str
    #: Optional tracking issue / PR, when closing the gap is planned.
    tracked_by: T.Optional[str] = None


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
    # When set, t2n is *expected* not to produce a running NNEF for this
    # op, and the spec exists to measure the other exporters (and to keep
    # the gap on the books) rather than to guard a translation we ship.
    #
    # Distinct from `xfail_reason`, which marks a translation we *do*
    # ship whose output currently disagrees with torch. Here there is no
    # translation to disagree with.
    #
    # The tract driver asserts the declared stage really happens, so the
    # day the gap closes the spec fails and has to be converted into a
    # normal one. See `tests/proptest/nnef_gap.py`.
    nnef_gap: T.Optional[NnefGap] = None
    # When True, two runs of this op on the same input legitimately
    # differ (anything drawing from an RNG, plus `empty`, whose buffer is
    # uninitialized). Value comparison is then meaningless, so the ONNX
    # sweep still measures export and runtime but skips the numerics
    # axis rather than recording a guaranteed divergence.
    nondeterministic: bool = False


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
