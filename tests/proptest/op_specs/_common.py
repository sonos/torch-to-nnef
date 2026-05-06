"""Shared types for the proptest op_specs package.

``OpSpec`` and ``OpSample`` are the public types every spec module produces;
``_unary_sample_st`` is the only sample-strategy helper genuinely shared
across multiple op groups (elementwise unary, activations, conv/pool input
generation). Group-specific helpers (binary broadcast, pow variants, the
unary-domain constants) live in their consumer module.
"""

import typing as T
from dataclasses import dataclass, field

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
    kwargs: T.Dict[str, T.Any]
    module: torch.nn.Module


@dataclass(frozen=True)
class OpSpec:
    name: str
    sample_st: st.SearchStrategy[OpSample]
    tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE
    dtypes_hint: T.Tuple[torch.dtype, ...] = field(default_factory=tuple)
    # When set, the test driver marks this spec's pytest case as xfail with
    # the given reason. Use for known divergences that have a tracked fix
    # (in t2n or tract) so the bug stays visible in CI without blocking PRs.
    # When the underlying fix lands, removing this field flips the spec
    # back to a normal pass and surfaces any regression.
    xfail_reason: T.Optional[str] = None


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
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(op))

    return _draw()
