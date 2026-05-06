"""Shared types and strategies for the proptest op_specs package.

``OpSpec`` / ``OpSample`` are the public types; the ``_*_sample_st`` helpers
are reused across multiple op groups (unary domain, binary broadcast,
multi-dtype, pow variants).
"""

import typing as T
from dataclasses import dataclass, field

import torch
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    BinaryPrimitive,
    UnaryPrimitive,
)
from ..inputs import Interval, dtype_st, tensor_st
from ..shapes import binary_broadcast_shapes_st, shape_st


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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


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


def _binary_broadcast_sample_st(
    op: T.Callable[..., torch.Tensor],
    dtype: torch.dtype = torch.float32,
    domain: T.Optional[Interval] = None,
    finite: bool = True,
) -> st.SearchStrategy[OpSample]:
    """Build a binary-op sample strategy with mutually broadcastable shapes."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(tensor_st(sa, dtype, finite=finite, domain=domain))
        b = draw(tensor_st(sb, dtype, finite=finite, domain=domain))
        return OpSample(inputs=(a, b), kwargs={}, module=BinaryPrimitive(op))

    return _draw()


def _binary_multi_dtype_sample_st(
    op: T.Callable[..., torch.Tensor],
    dtypes: T.Sequence[torch.dtype] = (torch.float32, torch.float16),
    domain_f32: T.Optional[Interval] = None,
    domain_f16: T.Optional[Interval] = None,
) -> st.SearchStrategy[OpSample]:
    """Build a binary-op sample that sweeps a list of float dtypes.

    Inputs are drawn with a per-dtype domain (f16 has a tighter range to
    keep results in its representable interval). Both inputs share the
    drawn dtype -- broadcasting is independent.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        dtype = draw(dtype_st(list(dtypes)))
        domain = (
            domain_f16
            if dtype == torch.float16 and domain_f16 is not None
            else domain_f32
        )
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(tensor_st(sa, dtype, finite=True, domain=domain))
        b = draw(tensor_st(sb, dtype, finite=True, domain=domain))
        return OpSample(inputs=(a, b), kwargs={}, module=BinaryPrimitive(op))

    return _draw()


def _binary_pow_int_exp_sample_st() -> st.SearchStrategy[OpSample]:
    """Pow with integer-valued exponent tensors.

    Integer exponents go through a different code path in tract (a
    repeated-multiply or sqr/rsqr fragment for small constants -- see
    ``torch_to_nnef/op/aten/math.py:_pow``). Cover small absolute values
    to keep results bounded.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        base = draw(
            tensor_st(
                sa,
                torch.float32,
                finite=True,
                domain=Interval(0.5, 10.0),
            )
        )
        # Integer-valued floats from a small range (PyTorch's pow with int
        # exponent goes through ``aten::pow.Tensor_Tensor_int`` if the
        # exponent is an int tensor -- here we test the float-exponent
        # path with integer values).
        exp_int = draw(st.integers(min_value=-3, max_value=3))
        exponent = torch.full(sb, float(exp_int), dtype=torch.float32)
        return OpSample(
            inputs=(base, exponent),
            kwargs={},
            module=BinaryPrimitive(torch.pow),
        )

    return _draw()


def _binary_pow_sample_st() -> st.SearchStrategy[OpSample]:
    """Pow needs separate domains for base (>=0) and exponent (small)."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        base = draw(
            tensor_st(
                sa,
                torch.float32,
                finite=True,
                domain=Interval(0.1, 100.0),
            )
        )
        exponent = draw(
            tensor_st(
                sb,
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        return OpSample(
            inputs=(base, exponent),
            kwargs={},
            module=BinaryPrimitive(torch.pow),
        )

    return _draw()


# -----------------------------------------------------------------------------
# Unary float specs (15)
# -----------------------------------------------------------------------------

# Domain bounds chosen to keep outputs in a numerically meaningful range and
# avoid trivial saturation while still exercising edge cases.
_UNARY_TRIG_DOMAIN = Interval(-6.283, 6.283)  # ~ +/- 2*pi
_UNARY_TAN_DOMAIN = Interval(-1.4, 1.4)  # avoid tan(pi/2) explosion
_UNARY_EXP_DOMAIN = Interval(-30.0, 30.0)
_UNARY_LOG_DOMAIN = Interval(1e-3, 1e4)
_UNARY_FINITE_DOMAIN = Interval(-1e4, 1e4)
_UNARY_SQRT_DOMAIN = Interval(0.0, 1e4)
_UNARY_RSQRT_DOMAIN = Interval(1e-3, 1e4)
# v1 keeps reciprocal positive-only; coverage of negatives lands in v2 once
# the strategy supports disjoint intervals.
_UNARY_RECIP_DOMAIN = Interval(1e-2, 1e3)
_UNARY_TANH_DOMAIN = Interval(-30.0, 30.0)


# Domain for inverse-trig ops (asin, acos): input must be in [-1, 1].
_UNARY_INVTRIG_DOMAIN = Interval(-1.0, 1.0)
# Domain for acosh: input must be in [1, inf).
_UNARY_ACOSH_DOMAIN = Interval(1.0, 1e3)
# Domain for atanh: input must be in (-1, 1) strict; epsilon margin avoids
# the singularities at the boundary.
_UNARY_ATANH_DOMAIN = Interval(-0.999, 0.999)
# Domain for hyperbolic sinh/cosh: bounded to avoid overflow at f32.
_UNARY_HYP_DOMAIN = Interval(-30.0, 30.0)
