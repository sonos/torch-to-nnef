"""Shared pieces for the gap-spec modules.

Gap specs are thinner than ordinary ones: they exist to be *exported*,
not compared, so most need nothing but a module and one legal input.
What varies between them is the operator, the reason and the stage, so
`gap_spec` takes those three and derives the rest.
"""

import typing as T

import torch
from hypothesis import strategies as st

from ...inputs import Interval, tensor_st
from ...shapes import shape_st
from .._common import NnefGap, NnefGapStage, OpSample, OpSpec


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
) -> OpSpec:
    """One spec for one operator we do not translate.

    `op` is the support page's row name, used both as the declared
    `aten_ops` entry and (prefixed) as the spec id, so a grade in the
    artifact is traceable back here by name alone.
    """
    return OpSpec(
        name=f"gap-{op}",
        sample_st=sample_st,
        aten_ops=(op,),
        nnef_gap=NnefGap(stage=stage, reason=reason, tracked_by=tracked_by),
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
