"""Gap specs for shape-, index- and layout-driven operators.

The unifying theme is that the *shape* is the problem, not the
arithmetic. Either the output extent depends on the values (`nonzero`,
`unique`, `masked_select`), or the operator describes a memory layout
NNEF has no way to name (`as_strided`, `empty_permuted`).

That distinction is worth keeping: an op in the first group cannot be
supported by writing an emitter, because a static graph has nowhere to
put an unknown extent. An op in the second group only needs someone to
decide what the layout means once the tensor is a value rather than a
buffer.
"""

import typing as T

import torch
from hypothesis import strategies as st

from ...inputs import Interval, tensor_st
from ...shapes import shape_st
from .._common import NnefGapStage, OpSample, OpSpec
from ._helpers import (
    DEFAULT_DOMAIN,
    GapModule,
    gap_spec,
    small_int_st,
    unary_st,
)

_DATA_DEPENDENT = (
    "output extent depends on the input values, which a static NNEF "
    "graph cannot declare"
)
_LAYOUT = (
    "describes a memory layout rather than a value, and NNEF has no "
    "notion of strides"
)


@st.composite
def _mask_st(draw, fn, name: str, with_source: bool = False):
    """A float tensor plus a bool mask of the same shape."""
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    x = draw(tensor_st(shape, torch.float32, domain=DEFAULT_DOMAIN))
    mask = draw(tensor_st(shape, torch.bool))
    if not with_source:
        return OpSample(inputs=(x, mask), module=GapModule(fn, name))
    numel = 1
    for dim in shape:
        numel *= dim
    # Sized to the full numel so any mask is legal without inspecting it.
    source = draw(tensor_st((numel,), torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(inputs=(x, mask, source), module=GapModule(fn, name))


@st.composite
def _nonzero_st(draw, as_tuple: bool):
    """Small integers, so zeros actually occur and the extent varies."""
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    x = draw(tensor_st(shape, torch.int64, domain=Interval(0, 2)))
    fn = (
        (lambda t: torch.nonzero(t, as_tuple=True)[0])
        if as_tuple
        else torch.nonzero
    )
    name = "nonzero_numpy" if as_tuple else "nonzero"
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _as_strided_st(draw):
    rows = draw(st.integers(min_value=1, max_value=4))
    cols = draw(st.integers(min_value=1, max_value=4))
    x = draw(tensor_st((rows * cols,), torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(
        inputs=(x,),
        module=GapModule(
            lambda t: torch.as_strided(t, (rows, cols), (cols, 1)),
            "as_strided",
        ),
    )


@st.composite
def _rows_st(draw, fn, name: str, cols: int = 2):
    """A `(rows, cols)` int matrix, for the row-wise unique ops."""
    rows = draw(st.integers(min_value=2, max_value=6))
    x = draw(tensor_st((rows, cols), torch.int64, domain=Interval(0, 2)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _index_pair_st(draw, fn, name: str):
    """A destination matrix plus a small source, for scatter-likes."""
    cols = draw(st.integers(min_value=1, max_value=4))
    x = draw(tensor_st((3, cols), torch.float32, domain=DEFAULT_DOMAIN))
    source = draw(tensor_st((2, cols), torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(inputs=(x, source), module=GapModule(fn, name))


@st.composite
def _shape_only_st(draw, fn, name: str):
    """A tensor used only for its shape / as a graph anchor."""
    shape = draw(shape_st(min_rank=1, max_rank=3))
    x = draw(tensor_st(shape, torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _segment_st(draw):
    """Rows plus segment lengths that sum to exactly the row count."""
    segments = draw(st.integers(min_value=1, max_value=3))
    lengths = [
        draw(st.integers(min_value=1, max_value=3)) for _ in range(segments)
    ]
    cols = draw(st.integers(min_value=1, max_value=4))
    x = draw(
        tensor_st((sum(lengths), cols), torch.float32, domain=DEFAULT_DOMAIN)
    )
    lengths_t = torch.tensor(lengths)
    return OpSample(
        inputs=(x,),
        module=GapModule(
            lambda t: torch.segment_reduce(t, "sum", lengths=lengths_t),
            "segment_reduce",
        ),
    )


SPECS: T.Tuple[OpSpec, ...] = (
    # -- data-dependent extent --
    gap_spec("nonzero", _nonzero_st(as_tuple=False), _DATA_DEPENDENT),
    gap_spec(
        "nonzero_numpy",
        _nonzero_st(as_tuple=True),
        f"`nonzero(as_tuple=True)`: {_DATA_DEPENDENT}, once per axis",
    ),
    gap_spec(
        "argwhere",
        small_int_st(torch.argwhere, "argwhere", lo=0, hi=2),
        f"the same as `nonzero`: {_DATA_DEPENDENT}",
    ),
    gap_spec(
        "masked_select",
        _mask_st(torch.masked_select, "masked_select"),
        _DATA_DEPENDENT,
    ),
    gap_spec(
        "unique_dim",
        _rows_st(lambda t: torch.unique(t, dim=0), "unique_dim"),
        _DATA_DEPENDENT,
    ),
    gap_spec(
        "unique_consecutive",
        small_int_st(torch.unique_consecutive, "unique_consecutive", hi=2),
        _DATA_DEPENDENT,
    ),
    gap_spec(
        "unique_dim_consecutive",
        _rows_st(
            lambda t: torch.unique_consecutive(t, dim=0),
            "unique_dim_consecutive",
        ),
        _DATA_DEPENDENT,
    ),
    gap_spec(
        "combinations",
        small_int_st(
            lambda t: torch.combinations(t, r=2), "combinations", max_rank=1
        ),
        "output length is a binomial coefficient of the input length: "
        "static, but no emitter builds the index pairs",
    ),
    gap_spec(
        "nonzero_static",
        small_int_st(
            lambda t: torch.nonzero_static(t, size=3), "nonzero_static", hi=2
        ),
        "the padded form of `nonzero`, so the extent *is* static here; "
        "it fails before the emitter lookup instead",
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    # -- layout --
    gap_spec("as_strided", _as_strided_st(), _LAYOUT),
    gap_spec(
        "empty_strided",
        _shape_only_st(
            lambda t: torch.empty_strided((2, 2), (2, 1)) + t.sum() * 0,
            "empty_strided",
        ),
        f"{_LAYOUT}; uninitialized as well",
        nondeterministic=True,
    ),
    gap_spec(
        "new_empty_strided",
        _shape_only_st(
            lambda t: t.new_empty_strided((2, 2), (2, 1)), "new_empty_strided"
        ),
        f"{_LAYOUT}; uninitialized as well",
        nondeterministic=True,
    ),
    gap_spec(
        "empty_permuted",
        _shape_only_st(
            lambda t: torch.empty_permuted((2, 2), (1, 0)) + t.sum() * 0,
            "empty_permuted",
        ),
        f"{_LAYOUT}; uninitialized as well",
        nondeterministic=True,
    ),
    gap_spec(
        "empty",
        _shape_only_st(lambda t: torch.empty(t.shape, dtype=t.dtype), "empty"),
        "allocates without initializing, so there is nothing for a "
        "declarative graph to describe",
        nondeterministic=True,
    ),
    # -- scatter / gather variants --
    gap_spec(
        "masked_scatter",
        _mask_st(
            lambda t, m, s: t.masked_scatter(m, s),
            "masked_scatter",
            with_source=True,
        ),
        "consumes the source in mask order, so the read index is itself "
        "a cumulative sum of the mask",
    ),
    gap_spec(
        "put",
        _index_pair_st(
            # Flattened so the source has two elements whatever the
            # drawn column count is.
            lambda t, s: t.clone().put_(
                torch.tensor([0, 2]), s.reshape(-1)[:2]
            ),
            "put",
        ),
        "indexes the flattened tensor, so a lowering has to reshape "
        "around it; no emitter does",
    ),
    gap_spec(
        "index_reduce",
        _index_pair_st(
            lambda t, s: t.clone().index_reduce_(
                0, torch.tensor([0, 1]), s, "amax"
            ),
            "index_reduce",
        ),
        "no emitter, and the export crashes with a bare TypeError "
        "instead of naming the operator",
        stage=NnefGapStage.RAW_ERROR,
    ),
    gap_spec(
        "segment_reduce",
        _segment_st(),
        "segment lengths are a second, ragged shape the graph would "
        "have to carry",
    ),
    gap_spec(
        "pad_sequence",
        _index_pair_st(
            lambda a, b: torch.nn.utils.rnn.pad_sequence([a, b]),
            "pad_sequence",
        ),
        "pads to the longest input, so the output extent depends on a "
        "list of shapes rather than on one",
    ),
    # -- dtype / range --
    gap_spec(
        "chalf",
        unary_st(lambda t: t.chalf().float(), "chalf"),
        "complex32 has no NNEF dtype, and the export crashes with a "
        "bare KeyError instead of naming it",
        stage=NnefGapStage.RAW_ERROR,
    ),
    gap_spec(
        "range",
        _shape_only_st(lambda t: torch.range(0, 4) + t.sum() * 0, "range"),
        "the inclusive-end variant of `arange`, which we do translate; "
        "no emitter maps the deprecated spelling onto it",
    ),
)
