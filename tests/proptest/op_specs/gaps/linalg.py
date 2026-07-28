"""Gap specs for the dense linear-algebra family.

Almost all of these reduce to one missing capability: NNEF has no
factorization. Give tract an LU, a QR and a symmetric eigendecomposition
and most of this module collapses at once, which is why the reasons below
name the factorization rather than the operator.

Two shapes of spec appear here:

- direct: the operator takes a matrix (and maybe a right-hand side).
- factor-consuming: the operator takes the *output* of a factorization
  (`lu_solve` wants `lu_factor`'s pivots, `ormqr` wants `geqrf`'s
  reflectors). Those specs compute the factor inside the module, so the
  traced graph holds two unsupported operators and only the one under
  test is declared. A grade therefore describes the pair, which is the
  honest limit of measuring an operator whose input is another
  operator's output: nobody can run `lu_solve` without first running
  something that factorizes.
"""

import typing as T

import torch
from hypothesis import strategies as st

from ...inputs import Interval, tensor_st
from .._common import NnefGapStage, OpSample, OpSpec
from ._helpers import GapModule, gap_spec, matrix_st

_LU = "no NNEF factorization primitive: needs LU"
_QR = "no NNEF factorization primitive: needs QR / Householder"
_EIG = "no NNEF factorization primitive: needs an eigensolver"
_SVD = "no NNEF factorization primitive: needs SVD"
_CHOL = "no NNEF factorization primitive: needs Cholesky"


def _direct(
    row: str,
    fn: T.Callable[..., T.Any],
    reason: str,
    kind: str = "any",
    rhs: bool = False,
    stage: NnefGapStage = NnefGapStage.NO_EMITTER,
    batched: T.Optional[bool] = None,
) -> OpSpec:
    return gap_spec(
        row,
        matrix_st(fn, row, kind=kind, rhs=rhs, batched=batched),
        reason,
        stage,
    )


def _lu_factor(a):
    return torch.linalg.lu_factor(a)


def _geqrf(a):
    return torch.geqrf(a)


@st.composite
def _tensor_system_st(draw):
    """`tensorsolve(A, B)` with `A: (r, c, r*c)` and `B: (r, c)`."""
    rows = draw(st.integers(min_value=2, max_value=3))
    cols = draw(st.integers(min_value=2, max_value=3))
    dom = Interval(-3.0, 3.0)
    a = draw(tensor_st((rows, cols, rows * cols), torch.float32, domain=dom))
    # A random system is singular often enough that the failure would be
    # torch's rather than the exporter's, so bias the diagonal of the
    # square matrix it reshapes to.
    size = rows * cols
    a = (a.reshape(size, size) + torch.eye(size) * size).reshape(
        rows, cols, size
    )
    b = draw(tensor_st((rows, cols), torch.float32, domain=dom))
    return OpSample(
        inputs=(a, b),
        module=GapModule(torch.linalg.tensorsolve, "linalg_tensorsolve"),
    )


@st.composite
def _tensorinv_st(draw):
    """`tensorinv(A, ind=2)` with `A: (n, n, n, n)`."""
    size = draw(st.integers(min_value=2, max_value=3))
    a = draw(
        tensor_st(
            (size, size, size, size), torch.float32, domain=Interval(-2.0, 2.0)
        )
    )
    # A random 4-D tensor reshapes to a square matrix that is singular
    # often enough to make the failure about torch rather than about the
    # exporter, so bias the diagonal.
    flat = a.reshape(size * size, size * size)
    flat = flat + torch.eye(size * size) * (size * size)
    a = flat.reshape(size, size, size, size)
    return OpSample(
        inputs=(a,),
        module=GapModule(
            lambda t: torch.linalg.tensorinv(t, ind=2), "linalg_tensorinv"
        ),
    )


SPECS: T.Tuple[OpSpec, ...] = (
    # -- determinants (the two the `Gap vs ONNX` batch started with) --
    _direct("linalg_det", torch.linalg.det, _LU),
    # `logdet` of a matrix with a negative determinant is NaN, which
    # says nothing about export support, so keep it positive definite.
    _direct("logdet", torch.logdet, _LU, kind="spd"),
    _direct(
        "linalg_slogdet", lambda a: torch.linalg.slogdet(a)[1], _LU, kind="spd"
    ),
    # -- cholesky --
    _direct("cholesky", torch.cholesky, _CHOL, kind="spd"),
    _direct("cholesky_inverse", torch.cholesky_inverse, _CHOL, kind="lower"),
    _direct(
        "cholesky_solve",
        lambda a, b: torch.cholesky_solve(b, a),
        _CHOL,
        kind="lower",
        rhs=True,
    ),
    # -- LU --
    _direct("linalg_lu", lambda a: torch.linalg.lu(a)[0], _LU),
    _direct("linalg_inv", torch.linalg.inv, _LU, kind="spd"),
    _direct("linalg_solve", torch.linalg.solve, _LU, kind="spd", rhs=True),
    _direct(
        "linalg_solve_triangular",
        lambda a, b: torch.linalg.solve_triangular(a, b, upper=False),
        _LU,
        kind="lower",
        rhs=True,
    ),
    _direct(
        "triangular_solve",
        lambda a, b: torch.triangular_solve(b, a, upper=False)[0],
        _LU,
        kind="lower",
        rhs=True,
    ),
    _direct(
        "linalg_lu_solve",
        lambda a, b: torch.linalg.lu_solve(*_lu_factor(a), b),
        _LU,
        kind="spd",
        rhs=True,
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    _direct(
        "lu_solve",
        lambda a, b: torch.lu_solve(b, *_lu_factor(a)),
        _LU,
        kind="spd",
        rhs=True,
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    _direct(
        "lu_unpack",
        lambda a: torch.lu_unpack(*_lu_factor(a))[0],
        _LU,
        kind="spd",
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    _direct(
        "linalg_ldl_solve",
        lambda a, b: torch.linalg.ldl_solve(*torch.linalg.ldl_factor(a), b),
        _LU,
        kind="spd",
        rhs=True,
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    # Both of these want a tensor whose trailing axes flatten to a
    # square system, so they get their own strategy rather than the
    # matrix one.
    gap_spec(
        "linalg_tensorsolve",
        _tensor_system_st(),
        _LU,
    ),
    gap_spec(
        "linalg_tensorinv",
        _tensorinv_st(),
        _LU,
    ),
    # -- QR / Householder --
    _direct("qr", lambda a: torch.qr(a)[0], _QR),
    _direct("linalg_qr", lambda a: torch.linalg.qr(a)[0], _QR),
    _direct("geqrf", lambda a: torch.geqrf(a)[0], _QR),
    _direct(
        "linalg_householder_product",
        lambda a: torch.linalg.householder_product(*_geqrf(a)),
        _QR,
    ),
    _direct(
        "ormqr",
        lambda a, b: torch.ormqr(*_geqrf(a), b),
        _QR,
        rhs=True,
        batched=False,
    ),
    _direct(
        "linalg_lstsq",
        lambda a: torch.linalg.lstsq(a, a[..., :1])[0],
        f"{_QR}; the multi-output form raises before the emitter lookup",
        kind="tall",
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    # -- eigen --
    _direct("linalg_eig", lambda a: torch.linalg.eig(a)[0].real, _EIG),
    _direct("linalg_eigvals", lambda a: torch.linalg.eigvals(a).real, _EIG),
    _direct("linalg_eigh", lambda a: torch.linalg.eigh(a)[0], _EIG, kind="spd"),
    _direct("linalg_eigvalsh", torch.linalg.eigvalsh, _EIG, kind="spd"),
    _direct("linalg_matrix_exp", torch.linalg.matrix_exp, _EIG),
    # -- SVD --
    _direct("linalg_svd", lambda a: torch.linalg.svd(a)[1], _SVD, kind="tall"),
    _direct("linalg_pinv", torch.linalg.pinv, _SVD, kind="tall"),
    _direct("linalg_cond", torch.linalg.cond, _SVD, kind="spd"),
    _direct(
        "linalg_matrix_rank",
        torch.linalg.matrix_rank,
        _SVD,
        kind="spd",
        stage=NnefGapStage.EXPORT_ERROR,
    ),
    # 2-D only. `torch.nuclear_norm` is the deprecated spelling, kept
    # here because it is the one that traces to the row's own name:
    # `torch.norm(p="nuc")` goes through `linalg_matrix_norm`, which
    # would attribute this row's grade to a different operator.
    _direct(
        "nuclear_norm",
        torch.nuclear_norm,
        _SVD,
        batched=False,
    ),
    # -- expressible, just unwritten --
    _direct(
        "linalg_matrix_power",
        lambda a: torch.linalg.matrix_power(a, 3),
        "expressible as repeated matmul, but no emitter unrolls it",
    ),
    # `torch.trace` is 2-D only.
    _direct(
        "trace",
        torch.trace,
        "expressible as a diagonal gather plus a sum, but no emitter "
        "composes it",
        batched=False,
    ),
)
