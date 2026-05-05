"""Reusable joint-constraint composites for hypothesis op specs.

Each composite encodes a cross-input or input-vs-kwarg constraint that cannot
be satisfied by drawing inputs and kwargs independently (matmul inner-dim,
clamp ordered pair, permutation of [0..rank-1], etc.). v1 only ships the
composites needed by the v1 op coverage list (see the design plan); heavier
composites (matmul, conv, cat, gather) land in v2.
"""

import typing as T

from hypothesis import strategies as st


@st.composite
def clamp_kwargs_st(
    draw, lo: float = -10.0, hi: float = 10.0
) -> T.Dict[str, float]:
    """Draw a sorted (min, max) pair for `torch.clamp`.

    Always yields ``min <= max`` by construction.
    """
    floats = st.floats(
        min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False
    )
    a = draw(floats)
    b = draw(floats)
    lo_v, hi_v = (a, b) if a <= b else (b, a)
    return {"min": lo_v, "max": hi_v}


@st.composite
def permutation_st(draw, rank: int) -> T.Tuple[int, ...]:
    """Draw a permutation of ``range(rank)``."""
    return tuple(draw(st.permutations(list(range(rank)))))


@st.composite
def reduction_dim_st(draw, rank: int) -> int:
    """Draw a reduction dim index in ``[0, rank)``.

    Only valid when rank >= 1; the caller must guarantee that.
    """
    return draw(st.integers(min_value=0, max_value=rank - 1))


@st.composite
def transpose_dims_st(draw, rank: int) -> T.Tuple[int, int]:
    """Draw a pair (dim0, dim1) with both in ``[0, rank)`` and dim0 != dim1.

    Only valid when rank >= 2.
    """
    a = draw(st.integers(min_value=0, max_value=rank - 1))
    b = draw(
        st.integers(min_value=0, max_value=rank - 1).filter(lambda x: x != a)
    )
    return (a, b)


@st.composite
def reshape_target_st(
    draw, source_shape: T.Tuple[int, ...], max_rank: int = 4
) -> T.Tuple[int, ...]:
    """Draw a target shape with the same total number of elements as source.

    Strategy: factor ``prod(source_shape)`` into a random number of factors
    (between 1 and max_rank), respecting that the per-axis size stays small.
    For zero-size source, returns the source shape unchanged.
    """
    total = 1
    for s in source_shape:
        total *= s
    if total == 0:
        return source_shape
    # Build a divisor list of `total`.
    divisors = [d for d in range(1, total + 1) if total % d == 0]
    target_rank = draw(st.integers(min_value=1, max_value=max_rank))
    factors: T.List[int] = []
    remaining = total
    for _ in range(target_rank - 1):
        valid = [d for d in divisors if remaining % d == 0]
        f = draw(st.sampled_from(valid))
        factors.append(f)
        remaining //= f
    factors.append(remaining)
    return tuple(factors)
