"""Gap specs for order statistics, histograms and elementwise specials.

Three loose groups that share a property: unlike the linalg family, most
of these are *expressible* in NNEF. `median` is a sort plus a gather;
`gcd` is a loop we could unroll at a fixed bound. They are unsupported
because nobody needed them, not because the format cannot say them, so
the reasons here point at the shape of the missing lowering rather than
at a missing primitive.

The exceptions are the histogram family and `mode`, whose output either
depends on the values (bin edges chosen from the data) or requires a
count-and-compare pass NNEF has no idiom for.
"""

import typing as T

import torch
from hypothesis import strategies as st

from ...inputs import Interval, tensor_st
from ...shapes import shape_st
from .._common import NnefGapStage, OpSample, OpSpec
from ._helpers import (
    GapModule,
    binary_st,
    bounded,
    gap_spec,
    int_binary_st,
    small_int_st,
    unary_st,
)

_SORT = "expressible as a sort plus a gather, but no emitter composes it"
_DATA_BINS = (
    "bin edges are chosen from the values, so neither the output extent "
    "nor the edges are known before the data is"
)
_SERIES = (
    "no NNEF primitive and no algebraic decomposition: a series "
    "evaluation, so support means a kernel"
)


@st.composite
def _vector_st(draw, fn, name: str, dtype=torch.float32, domain=None):
    """A rank-1 tensor: enough for the reductions that take one."""
    size = draw(st.integers(min_value=2, max_value=12))
    if dtype.is_floating_point:
        domain = bounded(domain)
    x = draw(tensor_st((size,), dtype, domain=domain))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _isin_st(draw):
    shape = draw(shape_st(min_rank=1, max_rank=2, max_dim=6))
    dom = Interval(0, 8)
    x = draw(tensor_st(shape, torch.int64, domain=dom))
    test = draw(tensor_st((4,), torch.int64, domain=dom))
    return OpSample(inputs=(x, test), module=GapModule(torch.isin, "isin"))


@st.composite
def _matrix_rows_st(draw, fn, name: str, min_cols=3, max_cols=8):
    """A `(rows, cols)` float matrix, for the correlation-style ops."""
    rows = draw(st.integers(min_value=2, max_value=4))
    cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    x = draw(tensor_st((rows, cols), torch.float32, domain=Interval(-5.0, 5.0)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


SPECS: T.Tuple[OpSpec, ...] = (
    # -- order statistics --
    gap_spec("median", unary_st(torch.median, "median"), _SORT),
    gap_spec("nanmedian", unary_st(torch.nanmedian, "nanmedian"), _SORT),
    gap_spec("msort", unary_st(torch.msort, "msort", min_rank=2), _SORT),
    gap_spec(
        "kthvalue",
        unary_st(
            lambda x: torch.kthvalue(x, 1, dim=-1)[0], "kthvalue", min_rank=2
        ),
        _SORT,
    ),
    gap_spec(
        "quantile",
        unary_st(lambda x: torch.quantile(x, 0.5), "quantile"),
        _SORT,
    ),
    gap_spec(
        "nanquantile",
        unary_st(lambda x: torch.nanquantile(x, 0.5), "nanquantile"),
        _SORT,
    ),
    gap_spec(
        "mode",
        small_int_st(lambda x: torch.mode(x, dim=-1)[0], "mode", min_rank=2),
        "needs a count-and-compare over equal values, which has no NNEF idiom",
    ),
    # -- histograms and counting --
    gap_spec(
        "bincount",
        _vector_st(
            torch.bincount, "bincount", dtype=torch.int64, domain=Interval(0, 6)
        ),
        "output length is `max(input) + 1`, so it is data-dependent",
    ),
    gap_spec(
        "histc",
        _vector_st(lambda x: torch.histc(x, bins=4), "histc"),
        _DATA_BINS,
    ),
    gap_spec(
        "histogram",
        _vector_st(lambda x: torch.histogram(x, bins=4)[0], "histogram"),
        _DATA_BINS,
    ),
    gap_spec(
        "histogramdd",
        # `bins` must have one entry per innermost axis, so the column
        # count is fixed rather than drawn.
        _matrix_rows_st(
            lambda x: torch.histogramdd(x, bins=[3, 3])[0],
            "histogramdd",
            min_cols=2,
            max_cols=2,
        ),
        f"{_DATA_BINS}; also fails before the emitter lookup",
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    gap_spec("isin", _isin_st(), _SORT),
    # -- statistics --
    gap_spec(
        "corrcoef",
        _matrix_rows_st(torch.corrcoef, "corrcoef"),
        "expressible as a centred matmul plus a normalisation, but no "
        "emitter composes it",
    ),
    gap_spec(
        "cumulative_trapezoid",
        unary_st(
            lambda x: torch.cumulative_trapezoid(x, dim=-1),
            "cumulative_trapezoid",
            min_rank=2,
        ),
        "expressible as a cumulative sum of averaged neighbours, but no "
        "emitter composes it",
    ),
    # -- elementwise: integer --
    gap_spec(
        "gcd",
        int_binary_st(torch.gcd, "gcd"),
        "iterative (Euclid), so a lowering means unrolling to a fixed bound",
    ),
    gap_spec(
        "lcm",
        int_binary_st(torch.lcm, "lcm"),
        "same iteration as `gcd`, which it is defined in terms of",
    ),
    # -- elementwise: float --
    gap_spec(
        "erfinv",
        unary_st(torch.erfinv, "erfinv", domain=Interval(-0.99, 0.99)),
        _SERIES,
    ),
    gap_spec(
        "igamma",
        binary_st(torch.igamma, "igamma", domain=Interval(0.1, 10.0)),
        _SERIES,
    ),
    gap_spec(
        "igammac",
        binary_st(torch.igammac, "igammac", domain=Interval(0.1, 10.0)),
        _SERIES,
    ),
    gap_spec(
        "polygamma",
        unary_st(
            lambda x: torch.polygamma(1, x),
            "polygamma",
            domain=Interval(0.1, 10.0),
        ),
        _SERIES,
    ),
    gap_spec(
        "nextafter",
        binary_st(torch.nextafter, "nextafter"),
        "steps one representable float, which is a bit-level operation "
        "NNEF cannot express",
    ),
    gap_spec(
        "isreal",
        unary_st(torch.isreal, "isreal"),
        "trivially true for the real dtypes tract runs, but no emitter "
        "folds it to a constant",
    ),
)
