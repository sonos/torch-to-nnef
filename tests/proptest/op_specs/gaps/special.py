"""Gap specs for `torch.special`: Bessel, orthogonal polynomials, ndtr.

The largest single family of operators we do not translate, and the most
uniform: every one is elementwise, so a spec is just "which function"
plus "over what domain". They are grouped in one table rather than
written out, because twenty-nine near-identical spec literals would hide
the two things that actually differ.

None of them has an NNEF primitive, and none can be composed from what
NNEF does have (they are polynomial/series evaluations, not algebraic
identities). So the reason is shared, and closing any of them means
either a tract-side kernel or a fragment library.
"""

import typing as T

import torch

from ...inputs import Interval
from .._common import OpSpec
from ._helpers import binary_st, gap_spec, poly_st, unary_st

_REASON = (
    "no NNEF primitive and no decomposition into one: these are series "
    "evaluations, so support means a kernel rather than a lowering"
)

#: `(row name, torch.special attribute, input domain)`.
#:
#: The domain matters for the two that are not defined on all of R:
#: `ndtri` inverts a CDF so it needs (0, 1), and `zeta` diverges at 1.
_UNARY: T.Tuple[T.Tuple[str, str, T.Optional[Interval]], ...] = (
    ("special_airy_ai", "airy_ai", None),
    ("special_bessel_j0", "bessel_j0", None),
    ("special_bessel_j1", "bessel_j1", None),
    ("special_bessel_y0", "bessel_y0", Interval(0.1, 20.0)),
    ("special_bessel_y1", "bessel_y1", Interval(0.1, 20.0)),
    ("special_erfcx", "erfcx", None),
    ("special_log_ndtr", "log_ndtr", None),
    ("special_modified_bessel_i0", "modified_bessel_i0", None),
    ("special_modified_bessel_i1", "modified_bessel_i1", None),
    ("special_modified_bessel_k0", "modified_bessel_k0", Interval(0.1, 20.0)),
    ("special_modified_bessel_k1", "modified_bessel_k1", Interval(0.1, 20.0)),
    ("special_ndtr", "ndtr", None),
    ("special_ndtri", "ndtri", Interval(0.01, 0.99)),
    (
        "special_scaled_modified_bessel_k0",
        "scaled_modified_bessel_k0",
        Interval(0.1, 20.0),
    ),
    (
        "special_scaled_modified_bessel_k1",
        "scaled_modified_bessel_k1",
        Interval(0.1, 20.0),
    ),
    ("special_spherical_bessel_j0", "spherical_bessel_j0", None),
)

#: Orthogonal polynomial families, all `(x, n)` with an integer degree.
_POLY: T.Tuple[T.Tuple[str, str], ...] = (
    ("special_chebyshev_polynomial_t", "chebyshev_polynomial_t"),
    ("special_chebyshev_polynomial_u", "chebyshev_polynomial_u"),
    ("special_chebyshev_polynomial_v", "chebyshev_polynomial_v"),
    ("special_chebyshev_polynomial_w", "chebyshev_polynomial_w"),
    (
        "special_shifted_chebyshev_polynomial_t",
        "shifted_chebyshev_polynomial_t",
    ),
    (
        "special_shifted_chebyshev_polynomial_u",
        "shifted_chebyshev_polynomial_u",
    ),
    (
        "special_shifted_chebyshev_polynomial_v",
        "shifted_chebyshev_polynomial_v",
    ),
    (
        "special_shifted_chebyshev_polynomial_w",
        "shifted_chebyshev_polynomial_w",
    ),
    ("special_hermite_polynomial_h", "hermite_polynomial_h"),
    ("special_hermite_polynomial_he", "hermite_polynomial_he"),
    ("special_laguerre_polynomial_l", "laguerre_polynomial_l"),
    ("special_legendre_polynomial_p", "legendre_polynomial_p"),
)


def _unary_specs() -> T.Tuple[OpSpec, ...]:
    return tuple(
        gap_spec(
            row,
            unary_st(getattr(torch.special, attr), row, domain=domain),
            _REASON,
        )
        for row, attr, domain in _UNARY
    )


def _poly_specs() -> T.Tuple[OpSpec, ...]:
    return tuple(
        gap_spec(row, poly_st(getattr(torch.special, attr), row), _REASON)
        for row, attr in _POLY
    )


SPECS: T.Tuple[OpSpec, ...] = (
    *_unary_specs(),
    *_poly_specs(),
    gap_spec(
        "special_zeta",
        # Hurwitz zeta diverges as x -> 1 from above, so both arguments
        # start past it.
        binary_st(
            torch.special.zeta, "special_zeta", domain=Interval(1.5, 8.0)
        ),
        _REASON,
    ),
)
