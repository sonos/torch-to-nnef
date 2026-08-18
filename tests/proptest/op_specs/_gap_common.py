"""Shared pieces for specs of operators we do not translate yet.

These specs live in the same themed modules as everything else, marked
with `nnef_gap` (see `_common.NnefGap`), so that implementing an
operator means deleting one field rather than moving a spec between
files. They are thinner than ordinary specs: they exist to be
*exported*, not compared, so most need nothing but a module and one
legal input. What varies is the operator, the reason and the stage, so
`gap_spec` takes those three and derives the rest.
"""

import typing as T

import torch
import torch.nn.functional as F
from hypothesis import strategies as st

from ..inputs import Interval, tensor_st
from ..shapes import shape_st
from ._common import NnefGap, NnefGapStage, OpSample, OpSpec


class GapModule(torch.nn.Module):
    """Call a plain function on the drawn inputs.

    Named rather than anonymous so a hypothesis falsifying example
    prints which operator it came from.
    """

    def __init__(self, fn: T.Callable[..., T.Any], name: str):
        super().__init__()
        self.fn = fn
        self.name = name

    def extra_repr(self) -> str:
        return f"op={self.name}"

    def forward(self, *args):
        return self.fn(*args)


def gap_spec(
    op: str,
    sample_st,
    reason: str,
    stage: NnefGapStage = NnefGapStage.NO_EMITTER,
    nondeterministic: bool = False,
    tracked_by: T.Optional[str] = None,
    emitter_registered: bool = False,
) -> OpSpec:
    """One spec for one operator we do not translate.

    `op` is the support page's row name, used as both the declared
    `aten_ops` entry and the spec id, so a grade in the artifact is
    traceable back here by name alone. The id carries no marker of the
    gap on purpose: closing one must not rename the spec, or it would
    churn pytest ids, hypothesis's example database and the artifact.
    """
    return OpSpec(
        name=op,
        sample_st=sample_st,
        aten_ops=(op,),
        nnef_gap=NnefGap(
            stage=stage,
            reason=reason,
            tracked_by=tracked_by,
            emitter_registered=emitter_registered,
        ),
        nondeterministic=nondeterministic,
    )


# -- input strategies --------------------------------------------------
#
# Deliberately narrow. A gap spec is measuring "can the other exporter
# lower this operator at all", and a wide shape/dtype sweep buys little
# on an operator nobody has lowered yet, while costing 25 exports per
# example on every regeneration.

#: Default value range for float draws in this package.
#:
#: Ordinary specs sweep the full float range on purpose, because a
#: tolerance bug hides at the extremes. Nothing here compares values, so
#: magnitude only buys flakiness: `torch.histc` aborts the whole process
#: (SIGABRT, not an exception) on some FLT_MAX-scale draws, which would
#: take the rest of the suite down with it.
DEFAULT_DOMAIN = Interval(-1e4, 1e4)


def bounded(domain: T.Optional[Interval]) -> Interval:
    """The caller's domain, or the package default."""
    return DEFAULT_DOMAIN if domain is None else domain


@st.composite
def unary_st(draw, fn, name: str, domain=None, min_rank=1, max_rank=3):
    """`fn(x)` on one float tensor."""
    shape = draw(shape_st(min_rank=min_rank, max_rank=max_rank, max_dim=6))
    x = draw(tensor_st(shape, torch.float32, domain=bounded(domain)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def binary_st(draw, fn, name: str, domain=None, max_rank=3):
    """`fn(x, y)` on two same-shaped float tensors."""
    shape = draw(shape_st(min_rank=1, max_rank=max_rank, max_dim=6))
    dom = bounded(domain)
    x = draw(tensor_st(shape, torch.float32, domain=dom))
    y = draw(tensor_st(shape, torch.float32, domain=dom))
    return OpSample(inputs=(x, y), module=GapModule(fn, name))


@st.composite
def int_binary_st(draw, fn, name: str, lo=1, hi=20):
    """`fn(x, y)` on two same-shaped int tensors."""
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    dom = Interval(lo, hi)
    x = draw(tensor_st(shape, torch.int64, domain=dom))
    y = draw(tensor_st(shape, torch.int64, domain=dom))
    return OpSample(inputs=(x, y), module=GapModule(fn, name))


@st.composite
def poly_st(draw, fn, name: str):
    """`fn(x, n)`: a special polynomial evaluated at integer degree."""
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    x = draw(tensor_st(shape, torch.float32, domain=Interval(-1.0, 1.0)))
    n = draw(tensor_st(shape, torch.int64, domain=Interval(0, 4)))
    return OpSample(inputs=(x, n), module=GapModule(fn, name))


@st.composite
def small_int_st(draw, fn, name: str, lo=0, hi=4, min_rank=1, max_rank=2):
    """`fn(x)` on a small-range int tensor (duplicates are the point)."""
    shape = draw(shape_st(min_rank=min_rank, max_rank=max_rank, max_dim=6))
    x = draw(tensor_st(shape, torch.int64, domain=Interval(lo, hi)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


# -- matrix strategies -------------------------------------------------


def _square(draw, size, batched: bool):
    shape = ((2,) if batched else ()) + (size, size)
    return draw(tensor_st(shape, torch.float32, domain=Interval(-3.0, 3.0)))


@st.composite
def matrix_st(
    draw,
    fn,
    name: str,
    kind: str = "any",
    rhs: bool = False,
    batched: T.Optional[bool] = None,
):
    """A square matrix in one of the conditionings linalg ops require.

    `kind` picks how the drawn matrix is fixed up:

    - `any`: as drawn. Fine for determinant-like ops.
    - `spd`: `A @ A.T + n*I`, symmetric positive definite. Required by
      cholesky and the `eigh` family, and the safe choice for inverses
      since a random matrix is singular often enough to make the failure
      about torch rather than about the exporter.
    - `lower`: the lower triangle of an SPD matrix, for the solvers that
      take a factor.
    - `tall`: a non-square `(n+1, n)`, for least squares and pinv.

    `rhs=True` adds a second `(n, 2)` right-hand side input. `batched`
    forces a leading batch axis on or off; left unset, square kinds draw
    it, which is how a batch-only gap would show up.
    """
    size = draw(st.integers(min_value=2, max_value=4))
    if batched is None:
        batched = draw(st.booleans()) if kind in {"any", "spd"} else False
    if kind == "tall":
        mat = draw(
            tensor_st(
                (size + 1, size), torch.float32, domain=Interval(-3.0, 3.0)
            )
        )
    else:
        mat = _square(draw, size, batched)
        if kind in {"spd", "lower"}:
            mat = mat @ mat.transpose(-1, -2) + torch.eye(size) * size
            if kind == "lower":
                mat = torch.tril(mat)
    if not rhs:
        return OpSample(inputs=(mat,), module=GapModule(fn, name))
    rhs_shape = ((2,) if batched else ()) + (size, 2)
    vec = draw(tensor_st(rhs_shape, torch.float32, domain=Interval(-3.0, 3.0)))
    return OpSample(inputs=(mat, vec), module=GapModule(fn, name))


# -- strategies for specific untranslated families ---------------------


@st.composite
def vector_st(draw, fn, name: str, dtype=torch.float32, domain=None):
    """A rank-1 tensor: enough for the reductions that take one."""
    size = draw(st.integers(min_value=2, max_value=12))
    if dtype.is_floating_point:
        domain = bounded(domain)
    x = draw(tensor_st((size,), dtype, domain=domain))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def isin_st(draw):
    shape = draw(shape_st(min_rank=1, max_rank=2, max_dim=6))
    dom = Interval(0, 8)
    x = draw(tensor_st(shape, torch.int64, domain=dom))
    test = draw(tensor_st((4,), torch.int64, domain=dom))
    return OpSample(inputs=(x, test), module=GapModule(torch.isin, "isin"))


@st.composite
def matrix_rows_st(draw, fn, name: str, min_cols=3, max_cols=8):
    """A `(rows, cols)` float matrix, for the correlation-style ops."""
    rows = draw(st.integers(min_value=2, max_value=4))
    cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    x = draw(tensor_st((rows, cols), torch.float32, domain=Interval(-5.0, 5.0)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def mask_st(draw, fn, name: str, with_source: bool = False):
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
def nonzero_st(draw, as_tuple: bool):
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
def as_strided_st(draw):
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
def rows_st(draw, fn, name: str, cols: int = 2):
    """A `(rows, cols)` int matrix, for the row-wise unique ops."""
    rows = draw(st.integers(min_value=2, max_value=6))
    x = draw(tensor_st((rows, cols), torch.int64, domain=Interval(0, 2)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def index_pair_st(draw, fn, name: str):
    """A destination matrix plus a small source, for scatter-likes."""
    cols = draw(st.integers(min_value=1, max_value=4))
    x = draw(tensor_st((3, cols), torch.float32, domain=DEFAULT_DOMAIN))
    source = draw(tensor_st((2, cols), torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(inputs=(x, source), module=GapModule(fn, name))


@st.composite
def shape_only_st(draw, fn, name: str):
    """A tensor used only for its shape / as a graph anchor."""
    shape = draw(shape_st(min_rank=1, max_rank=3))
    x = draw(tensor_st(shape, torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def segment_st(draw):
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


@st.composite
def image_st(draw, fn, name: str, rank: int, side: int = 4):
    """A `(1, 1, side, ...)` feature map with `rank` spatial axes."""
    shape = (1, 1) + (side,) * rank
    x = draw(tensor_st(shape, torch.float32, domain=Interval(-5.0, 5.0)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def spectral_st(draw, fn, name: str):
    """A small square signal, since these are the n-D transforms."""
    side = draw(st.sampled_from([2, 4]))
    x = draw(tensor_st((side, side), torch.float32, domain=Interval(-5.0, 5.0)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def class_loss_st(draw, fn, name: str, target_rank: int):
    """Logits `(n, c)` with either per-sample or per-class targets."""
    samples = draw(st.integers(min_value=1, max_value=4))
    classes = draw(st.integers(min_value=2, max_value=5))
    logits = draw(
        tensor_st((samples, classes), torch.float32, domain=Interval(-5.0, 5.0))
    )
    shape = (samples,) if target_rank == 1 else (samples, classes)
    target = draw(
        tensor_st(shape, torch.int64, domain=Interval(0, classes - 1))
    )
    return OpSample(inputs=(logits, target), module=GapModule(fn, name))


@st.composite
def poisson_nll_st(draw):
    samples = draw(st.integers(min_value=1, max_value=4))
    feats = draw(st.integers(min_value=1, max_value=4))
    pred = draw(
        tensor_st((samples, feats), torch.float32, domain=Interval(-3.0, 3.0))
    )
    target = draw(
        tensor_st((samples, feats), torch.float32, domain=Interval(0.0, 5.0))
    )
    return OpSample(
        inputs=(pred, target),
        module=GapModule(F.poisson_nll_loss, "poisson_nll_loss"),
    )


@st.composite
def ctc_st(draw):
    """Log-probs `(T, N, C)` with fixed input/target lengths."""
    time = draw(st.integers(min_value=4, max_value=8))
    classes = draw(st.integers(min_value=3, max_value=5))
    target_len = draw(st.integers(min_value=1, max_value=3))
    logits = draw(
        tensor_st((time, 2, classes), torch.float32, domain=Interval(-3.0, 3.0))
    )
    targets = draw(
        tensor_st((2, target_len), torch.int64, domain=Interval(1, classes - 1))
    )

    def _fn(lp, tg):
        return F.ctc_loss(
            lp.log_softmax(-1),
            tg,
            torch.full((2,), time, dtype=torch.long),
            torch.full((2,), target_len, dtype=torch.long),
        )

    return OpSample(inputs=(logits, targets), module=GapModule(_fn, "ctc_loss"))


def unpool(rank: int):
    """`max_unpool` fed by the matching `max_pool`'s indices.

    `output_size` is passed explicitly: without it torch's shape check
    compares a traced tensor in a Python `if` and refuses to trace.
    """
    pool = F.max_pool2d if rank == 2 else F.max_pool3d
    unpool = F.max_unpool2d if rank == 2 else F.max_unpool3d
    size = [4] * rank

    def _fn(x):
        return unpool(*pool(x, 2, return_indices=True), 2, output_size=size)

    return _fn


# -- reasons shared by several untranslated families -------------------
REASON_SORT = "expressible as a sort plus a gather, but no emitter composes it"
REASON_SERIES = (
    "no NNEF primitive and no algebraic decomposition: a series "
    "evaluation, so support means a kernel"
)
REASON_DATA_BINS = (
    "bin edges are chosen from the values, so neither the output extent "
    "nor the edges are known before the data is"
)
REASON_DATA_DEPENDENT = (
    "output extent depends on the input values, which a static NNEF "
    "graph cannot declare"
)
REASON_LAYOUT = (
    "describes a memory layout rather than a value, and NNEF has no "
    "notion of strides"
)
REASON_FFT_AXES = (
    "the 1-D transforms are translated; these take an axis list, and no "
    "emitter maps that onto repeated applications"
)
