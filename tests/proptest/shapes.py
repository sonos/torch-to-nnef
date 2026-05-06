"""Shape strategies for hypothesis-driven primitive tests.

`shape_st` draws a shape with bounded rank and bounded per-dim size; zero-sized
dims are off by default since most ops misbehave on them in tract pre-0.22.
`binary_broadcast_shapes_st` and `ternary_broadcast_shapes_st` wrap the
hypothesis numpy extra to draw mutually broadcastable shape tuples for
binary/ternary ops.
"""

import typing as T

from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

# Default shape budget:
# - rank up to 4 (5+ only for ops with explicit higher-rank semantics)
# - per-dim size up to 8 (keeps total elements <= 4096 at rank 4)
# - zero-sized dims off by default
DEFAULT_MAX_RANK = 4
DEFAULT_MAX_DIM = 8


def shape_st(
    min_rank: int = 0,
    max_rank: int = DEFAULT_MAX_RANK,
    min_dim: int = 1,
    max_dim: int = DEFAULT_MAX_DIM,
    allow_zero: bool = False,
) -> st.SearchStrategy[T.Tuple[int, ...]]:
    """Strategy returning a shape tuple of length in [min_rank, max_rank].

    Args:
        min_rank: minimum tensor rank (0 = scalar).
        max_rank: maximum tensor rank.
        min_dim: minimum per-axis size when allow_zero is False.
        max_dim: maximum per-axis size.
        allow_zero: when True, lowers the per-axis lower bound to 0 (for ops
            that legitimately handle zero-sized dims, e.g. cat/stack).
    """
    lo = 0 if allow_zero else min_dim
    return st.lists(
        st.integers(min_value=lo, max_value=max_dim),
        min_size=min_rank,
        max_size=max_rank,
    ).map(tuple)


def binary_broadcast_shapes_st(
    max_rank: int = DEFAULT_MAX_RANK,
    max_dim: int = DEFAULT_MAX_DIM,
) -> st.SearchStrategy[T.Tuple[T.Tuple[int, ...], T.Tuple[int, ...]]]:
    """Strategy returning two mutually broadcastable shape tuples.

    Useful for binary arithmetic, comparison, and logical ops.
    """
    return npst.mutually_broadcastable_shapes(
        num_shapes=2, max_dims=max_rank, max_side=max_dim
    ).map(lambda r: tuple(tuple(s) for s in r.input_shapes))


def ternary_broadcast_shapes_st(
    max_rank: int = DEFAULT_MAX_RANK,
    max_dim: int = DEFAULT_MAX_DIM,
) -> st.SearchStrategy[
    T.Tuple[T.Tuple[int, ...], T.Tuple[int, ...], T.Tuple[int, ...]]
]:
    """Strategy returning three mutually broadcastable shape tuples.

    Useful for `torch.where(cond, a, b)`-style ternary ops.
    """
    return npst.mutually_broadcastable_shapes(
        num_shapes=3, max_dims=max_rank, max_side=max_dim
    ).map(lambda r: tuple(tuple(s) for s in r.input_shapes))
