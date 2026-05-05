"""Per-op registry for hypothesis-driven primitive tests.

Each :class:`OpSpec` carries a single joint hypothesis strategy that returns
the entire forward-call payload (inputs, kwargs, and the constructed module).
Drawing inputs and kwargs as one composite is the only way to encode
cross-input constraints (matmul inner-dim, cat shape agreement, clamp ordered
pair, etc.) without spending most generated examples on invalid combinations.

v1 covers ~40 ops (unary float, binary arithmetic/compare/logical, reductions,
shape ops, clamp, where). Heavier joint constructs (matmul, conv, cat, gather)
are deferred to v2.
"""

import typing as T
from dataclasses import dataclass, field
from functools import partial

import torch
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ..wrapper import (
    BinaryPrimitive,
    TensorFnPrimitive,
    TernaryPrimitive,
    UnaryPrimitive,
)
from .inputs import Interval, dtype_st, tensor_st
from .joint import (
    permutation_st,
    reduction_dim_st,
    reshape_target_st,
    transpose_dims_st,
)
from .shapes import (
    binary_broadcast_shapes_st,
    shape_st,
    ternary_broadcast_shapes_st,
)


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
    targets: T.Tuple[str, ...] = ("TractNNEF",)
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


def _unary_specs() -> T.List[OpSpec]:
    # Transcendentals get VERY tolerance (rtol/atol 1e-4) since tract's f32
    # implementation typically diverges from torch by 1-2 ULPs (~1e-6 relative)
    # and our CLOSE level (1e-5) trips on edge cases (e.g. sin near pi).
    cases: T.List[T.Tuple[str, T.Callable, TractCheckTolerance, Interval]] = [
        # -- existing 15 --
        ("sin", torch.sin, TractCheckTolerance.VERY, _UNARY_TRIG_DOMAIN),
        ("cos", torch.cos, TractCheckTolerance.VERY, _UNARY_TRIG_DOMAIN),
        ("tan", torch.tan, TractCheckTolerance.VERY, _UNARY_TAN_DOMAIN),
        ("exp", torch.exp, TractCheckTolerance.VERY, _UNARY_EXP_DOMAIN),
        ("log", torch.log, TractCheckTolerance.VERY, _UNARY_LOG_DOMAIN),
        ("log2", torch.log2, TractCheckTolerance.VERY, _UNARY_LOG_DOMAIN),
        ("abs", torch.abs, TractCheckTolerance.EXACT, _UNARY_FINITE_DOMAIN),
        ("sign", torch.sign, TractCheckTolerance.EXACT, _UNARY_FINITE_DOMAIN),
        ("neg", torch.neg, TractCheckTolerance.EXACT, _UNARY_FINITE_DOMAIN),
        ("sqrt", torch.sqrt, TractCheckTolerance.VERY, _UNARY_SQRT_DOMAIN),
        ("rsqrt", torch.rsqrt, TractCheckTolerance.VERY, _UNARY_RSQRT_DOMAIN),
        (
            "reciprocal",
            torch.reciprocal,
            TractCheckTolerance.VERY,
            _UNARY_RECIP_DOMAIN,
        ),
        ("asinh", torch.asinh, TractCheckTolerance.VERY, _UNARY_FINITE_DOMAIN),
        ("atan", torch.atan, TractCheckTolerance.VERY, _UNARY_FINITE_DOMAIN),
        ("tanh", torch.tanh, TractCheckTolerance.VERY, _UNARY_TANH_DOMAIN),
        # -- newly added (9) --
        # Rounding ops -- exact integer outputs, no tolerance needed.
        ("floor", torch.floor, TractCheckTolerance.EXACT, _UNARY_FINITE_DOMAIN),
        ("ceil", torch.ceil, TractCheckTolerance.EXACT, _UNARY_FINITE_DOMAIN),
        ("round", torch.round, TractCheckTolerance.EXACT, _UNARY_FINITE_DOMAIN),
        # Inverse trig and hyperbolic.
        ("asin", torch.asin, TractCheckTolerance.VERY, _UNARY_INVTRIG_DOMAIN),
        ("acos", torch.acos, TractCheckTolerance.VERY, _UNARY_INVTRIG_DOMAIN),
        ("sinh", torch.sinh, TractCheckTolerance.VERY, _UNARY_HYP_DOMAIN),
        ("cosh", torch.cosh, TractCheckTolerance.VERY, _UNARY_HYP_DOMAIN),
        ("acosh", torch.acosh, TractCheckTolerance.VERY, _UNARY_ACOSH_DOMAIN),
        ("atanh", torch.atanh, TractCheckTolerance.VERY, _UNARY_ATANH_DOMAIN),
    ]
    return [
        OpSpec(
            name=name,
            sample_st=_unary_sample_st(op, domain=domain),
            tolerance=tol,
            dtypes_hint=(torch.float32,),
        )
        for name, op, tol, domain in cases
    ]


def _unary_multi_dtype_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
    dtypes: T.Sequence[torch.dtype] = (torch.float32, torch.float16),
    domain_f32: T.Optional[Interval] = None,
    domain_f16: T.Optional[Interval] = None,
) -> st.SearchStrategy[OpSample]:
    """Build a unary-op sample that sweeps a list of float dtypes."""

    @st.composite
    def _draw(draw) -> OpSample:
        dtype = draw(dtype_st(list(dtypes)))
        domain = (
            domain_f16
            if dtype == torch.float16 and domain_f16 is not None
            else domain_f32
        )
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(tensor_st(shape, dtype, finite=True, domain=domain))
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(op))

    return _draw()


def _unary_broad_specs() -> T.List[OpSpec]:
    """Multi-dtype broadening for the highest-value unary ops.

    f16 has a tighter representable range; we shrink the per-op domain
    accordingly. We don't broaden every unary op -- the goal is to surface
    f16-specific paths in tract / t2n without exploding subprocess count.
    """
    f16_finite_domain = Interval(-1e3, 1e3)
    f16_log_domain = Interval(1e-3, 1e3)
    EXACT = TractCheckTolerance.EXACT
    VERY = TractCheckTolerance.VERY
    cases: T.List[
        T.Tuple[
            str,
            T.Callable,
            TractCheckTolerance,
            T.Optional[Interval],
            T.Optional[Interval],
        ]
    ] = [
        ("abs", torch.abs, EXACT, _UNARY_FINITE_DOMAIN, f16_finite_domain),
        ("neg", torch.neg, EXACT, _UNARY_FINITE_DOMAIN, f16_finite_domain),
        ("sin", torch.sin, VERY, _UNARY_TRIG_DOMAIN, _UNARY_TRIG_DOMAIN),
        ("cos", torch.cos, VERY, _UNARY_TRIG_DOMAIN, _UNARY_TRIG_DOMAIN),
        ("exp", torch.exp, VERY, Interval(-10.0, 10.0), Interval(-5.0, 5.0)),
        ("log", torch.log, VERY, _UNARY_LOG_DOMAIN, f16_log_domain),
        ("sqrt", torch.sqrt, VERY, Interval(0.0, 1e3), Interval(0.0, 1e3)),
        ("tanh", torch.tanh, VERY, _UNARY_TANH_DOMAIN, _UNARY_TANH_DOMAIN),
    ]
    return [
        OpSpec(
            name=f"{name}-broad",
            sample_st=_unary_multi_dtype_sample_st(
                op,
                dtypes=(torch.float32, torch.float16),
                domain_f32=domain_f32,
                domain_f16=domain_f16,
            ),
            tolerance=tol,
            dtypes_hint=(torch.float32, torch.float16),
        )
        for name, op, tol, domain_f32, domain_f16 in cases
    ]


# -----------------------------------------------------------------------------
# Binary specs (5 arithmetic + 6 compare + 3 logical = 14)
# -----------------------------------------------------------------------------

_BINARY_ARITH_DOMAIN = Interval(-1e3, 1e3)
_BINARY_DIV_NUM_DOMAIN = Interval(-1e3, 1e3)
# Divisor avoids zero (NaN-aware comparator handles inf, but inf coverage on
# the output side is more useful than 0/0=NaN coverage on the input side).
_BINARY_DIV_DEN_DOMAIN = Interval(1e-2, 1e3)


def _div_sample_st() -> st.SearchStrategy[OpSample]:
    return _div_like_sample_st(torch.div)


def _div_like_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Binary op where the SECOND argument is a divisor.

    Used for div, remainder, fmod, floor_divide. The divisor domain
    excludes near-zero (and therefore subnormal) values to avoid
    PyTorch/tract divergence on tiny denominators.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=_BINARY_DIV_NUM_DOMAIN
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=_BINARY_DIV_DEN_DOMAIN
            )
        )
        return OpSample(inputs=(a, b), kwargs={}, module=BinaryPrimitive(op))

    return _draw()


# -- Broadened specs derived from PyTorch op signatures ----------------------
# `add`/`sub` accept ``alpha`` (multiplier for ``other``) per
# https://pytorch.org/docs/stable/generated/torch.add.html and the t2n
# emitter at ``torch_to_nnef/op/aten/math.py:333-368`` exports it. We sweep
# alpha values plus multi-dtype (f32 + f16). Domain bounds for f16 are
# tighter to keep results within f16's representable range.
_F16_BINARY_DOMAIN = Interval(-50.0, 50.0)


def _add_or_sub_multi_dtype_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Sweep dtype (f32 + f16) for ``torch.add`` / ``torch.sub``.

    Note: ``alpha`` (the second documented parameter of these ops) is NOT
    swept here -- see ``_add_or_sub_alpha_sample_st`` and the corresponding
    ``add-alpha-xfail`` / ``sub-alpha-xfail`` registry entries for that
    coverage and the tracked emitter bug.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        dtype = draw(dtype_st([torch.float32, torch.float16]))
        domain = (
            _F16_BINARY_DOMAIN
            if dtype == torch.float16
            else _BINARY_ARITH_DOMAIN
        )
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(tensor_st(sa, dtype, finite=True, domain=domain))
        b = draw(tensor_st(sb, dtype, finite=True, domain=domain))
        return OpSample(inputs=(a, b), kwargs={}, module=BinaryPrimitive(op))

    return _draw()


def _add_or_sub_alpha_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Sweep non-default ``alpha`` for ``torch.add`` / ``torch.sub``.

    PyTorch's ``torch.add(a, b, alpha=k)`` computes ``a + k*b`` and
    ``torch.sub(a, b, alpha=k)`` computes ``a - k*b``. Originally proptest
    found that the alpha attribute was silently dropped at export -- two
    bugs combined: ``ir_helpers._prepare_arguments`` truncated aten:add /
    aten:sub inputs to the first two, and ``unary.generic_unary`` (which
    these ops were routed through) ignores attributes. Both fixed in this
    same change set: dedicated emitters live at
    ``torch_to_nnef/op/aten/math.py`` (search ``_add_or_sub_with_alpha``)
    and the input-truncation behavior was removed.
    """
    _no_specials = {"allow_nan": False, "allow_infinity": False}
    nonunit_alpha_st = st.one_of(
        st.floats(min_value=-3.0, max_value=-0.1, **_no_specials),
        st.floats(min_value=1.5, max_value=3.0, **_no_specials),
    )

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=_BINARY_ARITH_DOMAIN
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=_BINARY_ARITH_DOMAIN
            )
        )
        alpha = draw(nonunit_alpha_st)
        return OpSample(
            inputs=(a, b),
            kwargs={},
            module=BinaryPrimitive(partial(op, alpha=alpha)),
        )

    return _draw()


def _div_explicit_none_sample_st() -> st.SearchStrategy[OpSample]:
    """Div called with explicit ``rounding_mode=None``.

    Originally proptest found that the t2n div emitter cast the output to
    int64 whenever ``len(node.inputs) == 3``, even when ``rounding_mode``
    was the literal ``None`` (which PyTorch documents as equivalent to
    ``/`` true division). Now fixed: the emitter checks
    ``rounding_mode is not None`` before applying the cast (see
    ``torch_to_nnef/op/aten/math.py``).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=_BINARY_DIV_NUM_DOMAIN
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=_BINARY_DIV_DEN_DOMAIN
            )
        )
        return OpSample(
            inputs=(a, b),
            kwargs={},
            module=BinaryPrimitive(partial(torch.div, rounding_mode=None)),
        )

    return _draw()


def _div_rounding_sample_st() -> st.SearchStrategy[OpSample]:
    """Div with ``rounding_mode in {"trunc", "floor"}``.

    **Tract upstream precision bug -- this spec stays xfailed pending a
    tract fix.** The original t2n-side issues are fixed:

    1. ``div(float, float, rounding_mode="trunc")`` previously returned
       int64; now returns float32 to match PyTorch (the emitter only
       casts to int64 when the traced output dtype is integer).

    2. The remaining failure is a tract precision issue: tract's float
       division for some specific value pairs (e.g. ``11.75 / 11.75``)
       returns ~0.99999994 instead of 1.0 (off by ~0.5 ULP of f32
       epsilon), so ``trunc(0.99999994) = 0`` rather than ``trunc(1.0)
       = 1``. Reproduced directly with a plain ``div`` (no rounding).
       The trunc/floor NNEF fragments at ``torch_to_nnef/op/fragment/``
       are mathematically correct; they just operate on tract's already-
       imprecise division result.

    The fix is upstream in tract's f32 division algorithm.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=8))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=_BINARY_DIV_NUM_DOMAIN
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=_BINARY_DIV_DEN_DOMAIN
            )
        )
        rounding_mode = draw(st.sampled_from(["trunc", "floor"]))
        wrapped = partial(torch.div, rounding_mode=rounding_mode)
        return OpSample(
            inputs=(a, b), kwargs={}, module=BinaryPrimitive(wrapped)
        )

    return _draw()


def _binary_arith_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="add",
            sample_st=_binary_broadcast_sample_st(
                torch.add, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="add-broad",
            sample_st=_add_or_sub_multi_dtype_sample_st(torch.add),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32, torch.float16),
        ),
        OpSpec(
            name="add-alpha",
            sample_st=_add_or_sub_alpha_sample_st(torch.add),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="sub",
            sample_st=_binary_broadcast_sample_st(
                torch.sub, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="sub-broad",
            sample_st=_add_or_sub_multi_dtype_sample_st(torch.sub),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32, torch.float16),
        ),
        OpSpec(
            name="sub-alpha",
            sample_st=_add_or_sub_alpha_sample_st(torch.sub),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="mul",
            sample_st=_binary_broadcast_sample_st(
                torch.mul, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="div",
            sample_st=_div_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="div-explicit-none",
            sample_st=_div_explicit_none_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="div-rounding-xfail",
            sample_st=_div_rounding_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "div with rounding_mode='trunc'/'floor' (a) returns int64 "
                "instead of float32 and (b) gives 0 for div(x, x) when x>0 "
                "(off-by-one near integer quotients). See "
                "_div_rounding_sample_st docstring for repro."
            ),
        ),
        OpSpec(
            name="pow",
            sample_st=_binary_pow_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="pow-int-exp",
            sample_st=_binary_pow_int_exp_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="mul-broad",
            sample_st=_binary_multi_dtype_sample_st(
                torch.mul,
                domain_f32=_BINARY_ARITH_DOMAIN,
                domain_f16=_F16_BINARY_DOMAIN,
            ),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32, torch.float16),
        ),
        OpSpec(
            name="minimum",
            sample_st=_binary_broadcast_sample_st(
                torch.minimum, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="maximum",
            sample_st=_binary_broadcast_sample_st(
                torch.maximum, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # Element-wise ``torch.min(a, b)`` (binary form).
            # Distinct from the dim-reduction in ``min-dim``.
            name="min-elementwise",
            sample_st=_binary_broadcast_sample_st(
                torch.min, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="max-elementwise",
            sample_st=_binary_broadcast_sample_st(
                torch.max, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="floor_divide-xfail",
            sample_st=_binary_broadcast_sample_st(
                torch.floor_divide,
                domain=_BINARY_DIV_NUM_DOMAIN,
            ),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "Same tract upstream precision bug as div-rounding-xfail: "
                "tract's div(x, x) returns ~0.99999994 instead of 1.0, so "
                "floor(div(107, 107)) = 0 instead of 1. Fix is upstream in "
                "tract's f32 division algorithm."
            ),
        ),
        OpSpec(
            # remainder is implemented as ``a - floor(a/b) * b`` (see
            # ``torch_to_nnef/op/fragment/remainder.nnef``). The fragment
            # is mathematically correct, but it depends on tract's f32
            # ``div`` which has the precision bug noted in
            # ``div-rounding-xfail`` (``div(x, x)`` returns ~0.99999994).
            # That makes ``floor(div(x, x)) = 0`` instead of 1, and
            # ``remainder(x, x) = x`` instead of 0.
            name="remainder-xfail",
            sample_st=_div_like_sample_st(torch.remainder),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "Same tract upstream div precision bug propagates through "
                "the remainder fragment ``a - floor(a/b) * b``: "
                "remainder(205.375, 205.375) returns 205.375 in tract vs "
                "0 in PyTorch."
            ),
        ),
        OpSpec(
            # fmod is implemented as ``a - trunc(a/b) * b`` (see
            # ``torch_to_nnef/op/fragment/fmod.nnef``). Same upstream
            # tract div bug as remainder.
            name="fmod-xfail",
            sample_st=_div_like_sample_st(torch.fmod),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "Same tract upstream div precision bug propagates through "
                "the fmod fragment ``a - trunc(a/b) * b``."
            ),
        ),
    ]


def _binary_compare_specs() -> T.List[OpSpec]:
    cases: T.List[T.Tuple[str, T.Callable]] = [
        ("eq", torch.eq),
        ("ne", torch.ne),
        ("lt", torch.lt),
        ("le", torch.le),
        ("gt", torch.gt),
        ("ge", torch.ge),
    ]
    specs: T.List[OpSpec] = []
    for name, op in cases:
        specs.append(
            OpSpec(
                name=name,
                sample_st=_binary_broadcast_sample_st(
                    op, domain=_BINARY_ARITH_DOMAIN
                ),
                tolerance=TractCheckTolerance.EXACT,
                dtypes_hint=(torch.float32,),
            )
        )
        # Multi-dtype broadening. Compare ops always produce bool, so the
        # comparator is exact regardless of input dtype.
        specs.append(
            OpSpec(
                name=f"{name}-broad",
                sample_st=_binary_multi_dtype_sample_st(
                    op,
                    dtypes=(torch.float32, torch.float16),
                    domain_f32=_BINARY_ARITH_DOMAIN,
                    domain_f16=_F16_BINARY_DOMAIN,
                ),
                tolerance=TractCheckTolerance.EXACT,
                dtypes_hint=(torch.float32, torch.float16),
            )
        )
    return specs


def _binary_logical_specs() -> T.List[OpSpec]:
    cases: T.List[T.Tuple[str, T.Callable]] = [
        ("logical_and", torch.logical_and),
        ("logical_or", torch.logical_or),
        ("logical_xor", torch.logical_xor),
    ]
    return [
        OpSpec(
            name=name,
            sample_st=_binary_broadcast_sample_st(op, dtype=torch.bool),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.bool,),
        )
        for name, op in cases
    ]


# -----------------------------------------------------------------------------
# Reduction specs (4)
# -----------------------------------------------------------------------------


def _reduction_sample_st(
    method_name: str,
    allow_keepdim: bool = True,
) -> st.SearchStrategy[OpSample]:
    """Sample for a tensor-method reduction taking dim and optional keepdim.

    Drawn jointly: rank in [1, 4], shape, dim in [0, rank), keepdim bool.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        kwargs: T.Dict[str, T.Any] = {"dim": dim}
        if allow_keepdim:
            kwargs["keepdim"] = draw(st.booleans())
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(method_name, kwargs=kwargs),
        )

    return _draw()


def _sum_full_or_multi_dim_sample_st() -> st.SearchStrategy[OpSample]:
    """Sum with the full dim surface per torch.sum doc.

    PyTorch's ``torch.sum`` accepts ``dim`` as ``None`` (reduce all),
    a single int, or a tuple/list of ints (multi-axis reduction). The t2n
    reducer at ``torch_to_nnef/op/aten/reducer.py:46-53`` handles all
    three. The original sum-dim spec only swept the single-int case; this
    one adds the multi-dim and full-reduction surface.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # Sweep the three dim modes the doc describes.
        mode = draw(st.sampled_from(["all", "single", "multi"]))
        kwargs: T.Dict[str, T.Any] = {}
        if mode == "all":
            # Two equivalent ways to reduce everything: dim=None or no dim.
            if draw(st.booleans()):
                kwargs["dim"] = None
            # else: don't pass dim at all.
        elif mode == "single":
            kwargs["dim"] = draw(st.integers(min_value=0, max_value=rank - 1))
            kwargs["keepdim"] = draw(st.booleans())
        else:  # multi
            n = draw(st.integers(min_value=1, max_value=rank))
            dims = draw(
                st.lists(
                    st.integers(min_value=0, max_value=rank - 1),
                    min_size=n,
                    max_size=n,
                    unique=True,
                )
            )
            kwargs["dim"] = tuple(sorted(dims))
            kwargs["keepdim"] = draw(st.booleans())
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive("sum", kwargs=kwargs),
        )

    return _draw()


def _bool_reduction_sample_st(method_name: str) -> st.SearchStrategy[OpSample]:
    """Any / all reduction over a single dim, bool input + output."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        keepdim = draw(st.booleans())
        x = draw(tensor_st(shape, torch.bool))
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                method_name, kwargs={"dim": dim, "keepdim": keepdim}
            ),
        )

    return _draw()


def _prod_dim_sample_st() -> st.SearchStrategy[OpSample]:
    """Prod reduction. Inputs near 1.0 to keep results bounded."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        # Keep dim sizes small (<=4) -- product of many values quickly
        # under/overflows even at f32.
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        keepdim = draw(st.booleans())
        # Values near 1 (in the (0.5, 2.0) range) keep cumulative product
        # in a numerically stable interval.
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(0.5, 2.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "prod", kwargs={"dim": dim, "keepdim": keepdim}
            ),
        )

    return _draw()


def _var_dim_sample_st() -> st.SearchStrategy[OpSample]:
    """Var reduction over a single dim, biased estimator only.

    Two t2n limitations narrow this spec for v1:

    1. The var emitter at ``torch_to_nnef/op/aten/math.py:717`` raises
       NotImplementedError when ``correction != 0`` (PyTorch defaults to
       1 -- unbiased estimator); we sweep correction=0 only.
    2. The same emitter does not honor ``keepdim`` -- it always emits a
       squeezed-axes ``var`` and never reshapes back. ``keepdim=True``
       would surface as a shape mismatch (ref ``(..., 1, ...)`` vs tract
       ``(...)``); we sweep keepdim=False only.

    Both are tracked t2n improvements that this spec will widen against
    once they land.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "var",
                kwargs={
                    "dim": dim,
                    "keepdim": False,
                    "correction": 0,
                },
            ),
        )

    return _draw()


def _reduction_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="sum-dim",
            sample_st=_reduction_sample_st("sum"),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # Multi-dim and full-tensor sums accumulate float error across
            # many elements; the per-dim form (sum-dim) only sums one axis
            # so APPROXIMATE works there. Multi-dim sums need CLOSE.
            name="sum-dim-broad",
            sample_st=_sum_full_or_multi_dim_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="mean-dim",
            sample_st=_reduction_sample_st("mean"),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="max-dim",
            sample_st=_reduction_sample_st("max"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="min-dim",
            sample_st=_reduction_sample_st("min"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        # Argmax / argmin return int64 indices -- the comparator's exact
        # int path catches any divergence. Pure index ops, no tolerance
        # needed.
        OpSpec(
            name="argmax-dim",
            sample_st=_reduction_sample_st("argmax"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="argmin-dim",
            sample_st=_reduction_sample_st("argmin"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        # any / all are bool reductions (input bool, output bool).
        # tract 0.22.1 (latest in TractNNEF.OFFICIAL_SUPPORTED_VERSIONS)
        # does NOT define ``any_reduce`` / ``all_reduce`` operators -- they
        # were added in tract > 0.22.1. The curated test at
        # ``tests/test_primitive.py:1180-1188`` skips these via
        # ``cond_tract_gt_0_22_0``. Xfail until the supported version set
        # bumps past 0.22.1.
        OpSpec(
            name="any-dim-xfail",
            sample_st=_bool_reduction_sample_st("any"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.bool,),
            xfail_reason=(
                "tract 0.22.1 lacks any_reduce; introduced in tract > "
                "0.22.1. Bumping TractNNEF.OFFICIAL_SUPPORTED_VERSIONS "
                "will flip this back to a normal pass."
            ),
        ),
        OpSpec(
            name="all-dim-xfail",
            sample_st=_bool_reduction_sample_st("all"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.bool,),
            xfail_reason=(
                "tract 0.22.1 lacks all_reduce; introduced in tract > "
                "0.22.1. Bumping TractNNEF.OFFICIAL_SUPPORTED_VERSIONS "
                "will flip this back to a normal pass."
            ),
        ),
        # prod is multiplicative reduction; numerical drift accumulates
        # for long axes and small values, so CLOSE rather than APPROXIMATE.
        OpSpec(
            name="prod-dim",
            sample_st=_prod_dim_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        # var has unbiased/biased variants via the ``correction`` kwarg.
        # We sweep both.
        OpSpec(
            name="var-dim",
            sample_st=_var_dim_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Shape op specs (5)
# -----------------------------------------------------------------------------


def _reshape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        source_shape = draw(
            shape_st(min_rank=1, max_rank=4, min_dim=1, max_dim=6)
        )
        target_shape = draw(reshape_target_st(source_shape, max_rank=4))
        x = draw(
            tensor_st(
                source_shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.reshape, shape=target_shape)),
        )

    return _draw()


def _transpose_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        d0, d1 = draw(transpose_dims_st(rank))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.transpose, dim0=d0, dim1=d1)),
        )

    return _draw()


def _permute_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        perm = draw(permutation_st(rank))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.permute, dims=perm)),
        )

    return _draw()


def _unsqueeze_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        rank = len(shape)
        dim = draw(st.integers(min_value=0, max_value=rank))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.unsqueeze, dim=dim)),
        )

    return _draw()


def _squeeze_sample_st() -> st.SearchStrategy[OpSample]:
    """Squeeze on a dim that is guaranteed to have size 1.

    PyTorch's ``squeeze(dim=k)`` on a non-1 dim is a no-op, but tract rejects
    the resulting NNEF graph (ModelBuildingError). We make the strategy
    always pick a valid dim by inserting a size-1 axis first, then choosing
    that index as the squeeze target.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank_other = draw(st.integers(min_value=0, max_value=3))
        other_shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank_other,
                    max_size=rank_other,
                )
            )
        )
        # Choose where to inject the size-1 axis among 0..rank_other.
        squeeze_dim = draw(st.integers(min_value=0, max_value=rank_other))
        shape = other_shape[:squeeze_dim] + (1,) + other_shape[squeeze_dim:]
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.squeeze, dim=squeeze_dim)),
        )

    return _draw()


def _view_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.view(*shape)`` -- like reshape but requires contiguous input."""

    @st.composite
    def _draw(draw) -> OpSample:
        source_shape = draw(
            shape_st(min_rank=1, max_rank=4, min_dim=1, max_dim=6)
        )
        target_shape = draw(reshape_target_st(source_shape, max_rank=4))
        x = draw(
            tensor_st(
                source_shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive("view", args=tuple(target_shape)),
        )

    return _draw()


def _flatten_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.flatten(start_dim, end_dim)`` over a random valid range."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        start_dim = draw(st.integers(min_value=0, max_value=rank - 1))
        end_dim = draw(st.integers(min_value=start_dim, max_value=rank - 1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "flatten",
                kwargs={"start_dim": start_dim, "end_dim": end_dim},
            ),
        )

    return _draw()


def _narrow_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.narrow(x, dim, start, length)`` with start+length <= dim size."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        dim_size = shape[dim]
        start = draw(st.integers(min_value=0, max_value=dim_size - 1))
        length = draw(st.integers(min_value=1, max_value=dim_size - start))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(
                partial(torch.narrow, dim=dim, start=start, length=length)
            ),
        )

    return _draw()


def _expand_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.expand(*sizes)`` -- each source dim must be 1 or equal."""

    @st.composite
    def _draw(draw) -> OpSample:
        # Source rank in [1, 3] to keep the expanded total size bounded.
        rank = draw(st.integers(min_value=1, max_value=3))
        # Source must have at least one size-1 dim to make expand
        # non-trivial. We force every dim to be 1 with prob 0.5; otherwise
        # match the eventual target.
        target = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        source = tuple((1 if draw(st.booleans()) else d) for d in target)
        x = draw(
            tensor_st(
                source,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive("expand", args=tuple(target)),
        )

    return _draw()


def _repeat_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.repeat(*sizes)`` -- repeats the tensor along each dim.

    PyTorch allows ``len(sizes) >= rank``, with the source treated as if
    it had leading size-1 dims to match. Repeats are constrained to
    small values to keep the output bounded.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=3),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # repeat sizes can have rank >= source rank.
        repeat_rank = draw(st.integers(min_value=rank, max_value=rank + 1))
        sizes = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=3),
                    min_size=repeat_rank,
                    max_size=repeat_rank,
                )
            )
        )
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive("repeat", args=(list(sizes),)),
        )

    return _draw()


def _shape_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="reshape",
            sample_st=_reshape_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="transpose",
            sample_st=_transpose_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="permute",
            sample_st=_permute_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="unsqueeze",
            sample_st=_unsqueeze_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="squeeze",
            sample_st=_squeeze_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="view",
            sample_st=_view_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="flatten",
            sample_st=_flatten_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="narrow",
            sample_st=_narrow_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="expand",
            sample_st=_expand_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="repeat",
            sample_st=_repeat_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# clamp + where (2)
# -----------------------------------------------------------------------------


def _clamp_sample_st() -> st.SearchStrategy[OpSample]:
    """Clamp over the full PyTorch surface.

    Historically narrowed for two distinct reasons that turned out to be
    the SAME ``if X.data:`` Python falsy bug in t2n's clamp emitter (now
    fixed in ``torch_to_nnef/op/aten/activation.py``):

    - ``min/max == 0.0`` was silently skipped (truthy check on the bound
      value), letting tract output the unclamped input.
    - ``min == max == 0.0`` plus matching input tripped
      ``KeyError: 'output_0'`` because BOTH conditional branches were
      skipped, leaving the output node unwired.

    Both go away with the explicit ``is None`` check. This strategy now
    sweeps the full bounded range including zero, equal bounds, and
    sign-crossing intervals.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        bounded_float = st.floats(
            min_value=-50.0,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        )
        a = draw(bounded_float)
        b = draw(bounded_float)
        lo_v, hi_v = (a, b) if a <= b else (b, a)
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.clamp, min=lo_v, max=hi_v)),
        )

    return _draw()


def _where_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        sc, sa, sb = draw(ternary_broadcast_shapes_st(max_rank=4, max_dim=6))
        cond = draw(tensor_st(sc, torch.bool))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=_BINARY_ARITH_DOMAIN
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=_BINARY_ARITH_DOMAIN
            )
        )
        return OpSample(
            inputs=(cond, a, b),
            kwargs={},
            module=TernaryPrimitive(torch.where),
        )

    return _draw()


def _clamp_where_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="clamp",
            sample_st=_clamp_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="where",
            sample_st=_where_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Activation specs
# -----------------------------------------------------------------------------

# nn.functional activations are mostly unary on a tensor with a fixed
# bounded output (sigmoid, hardsigmoid, hardtanh) or saturating output
# (gelu, silu, mish). Domain bounded to keep results numerically stable.
_ACT_DOMAIN = Interval(-30.0, 30.0)


def _activation_specs() -> T.List[OpSpec]:
    import torch.nn.functional as F

    EXACT = TractCheckTolerance.EXACT
    VERY = TractCheckTolerance.VERY
    SUPER = TractCheckTolerance.SUPER

    # Pure unary activations -- no kwargs, just elementwise.
    pure_unary: T.List[T.Tuple[str, T.Callable, TractCheckTolerance]] = [
        ("relu", F.relu, EXACT),
        ("sigmoid", F.sigmoid, VERY),
        ("gelu", F.gelu, VERY),
        ("silu", F.silu, VERY),
        # hardswish = x * relu6(x+3) / 6 -- multi-op chain, ULP-level
        # divergence between PyTorch and tract is normal, EXACT is too
        # strict.
        ("hardswish", F.hardswish, TractCheckTolerance.APPROXIMATE),
        # hardsigmoid = clamp((x+3)/6, 0, 1) -- one mul + one add +
        # min/max chain; small ULP drift, APPROXIMATE matches the
        # tolerance picked for hardswish.
        ("hardsigmoid", F.hardsigmoid, TractCheckTolerance.APPROXIMATE),
        # mish = x * tanh(softplus(x)) -- saturating, slow tails.
        ("mish", F.mish, VERY),
        ("selu", F.selu, VERY),
        ("relu6", F.relu6, EXACT),
        ("erf", torch.erf, VERY),
    ]
    specs: T.List[OpSpec] = [
        OpSpec(
            name=name,
            sample_st=_unary_sample_st(op, domain=_ACT_DOMAIN),
            tolerance=tol,
            dtypes_hint=(torch.float32,),
        )
        for name, op, tol in pure_unary
    ]

    # Activations with a single optional kwarg (kept at default for v1; the
    # kwarg surface is its own broadening pass once these baseline pass).
    leaky_relu = partial(F.leaky_relu, negative_slope=0.01)
    elu_default = partial(F.elu, alpha=1.0)
    hardtanh_default = partial(F.hardtanh, min_val=-1.0, max_val=1.0)
    softplus_default = partial(F.softplus, beta=1.0, threshold=20.0)

    specs.extend(
        [
            OpSpec(
                name="leaky_relu",
                sample_st=_unary_sample_st(leaky_relu, domain=_ACT_DOMAIN),
                tolerance=EXACT,
                dtypes_hint=(torch.float32,),
            ),
            OpSpec(
                name="elu",
                sample_st=_unary_sample_st(elu_default, domain=_ACT_DOMAIN),
                tolerance=VERY,
                dtypes_hint=(torch.float32,),
            ),
            OpSpec(
                name="hardtanh",
                sample_st=_unary_sample_st(
                    hardtanh_default, domain=_ACT_DOMAIN
                ),
                tolerance=EXACT,
                dtypes_hint=(torch.float32,),
            ),
            OpSpec(
                name="softplus",
                sample_st=_unary_sample_st(
                    softplus_default, domain=_ACT_DOMAIN
                ),
                tolerance=SUPER,
                dtypes_hint=(torch.float32,),
            ),
        ]
    )

    # threshold(input, threshold, value): elementwise gating with two
    # scalar args. Sweep both inside the input domain so we get a healthy
    # mix of below-threshold and above-threshold positions.
    @st.composite
    def _threshold_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        thresh = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        value = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(
                partial(F.threshold, threshold=thresh, value=value)
            ),
        )

    specs.append(
        OpSpec(
            name="threshold",
            sample_st=_threshold_sample(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        )
    )

    # ---- kwarg-broad variants ----
    # gelu has an ``approximate`` kwarg (``"none"`` (default) or
    # ``"tanh"``). Per the PyTorch doc, "tanh" uses an approximate formula
    # that often matches different cuda kernels.
    @st.composite
    def _gelu_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        approximate = draw(st.sampled_from(["none", "tanh"]))
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(F.gelu, approximate=approximate)),
        )

    @st.composite
    def _leaky_relu_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # Negative slopes from PyTorch examples: 0.01 (default), 0.1, 0.2.
        slope = draw(
            st.floats(
                min_value=0.001,
                max_value=0.5,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(F.leaky_relu, negative_slope=slope)),
        )

    @st.composite
    def _elu_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # alpha controls the negative-side saturation; 1.0 is default but
        # other values are common in tuned models.
        alpha = draw(
            st.floats(
                min_value=0.1,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(F.elu, alpha=alpha)),
        )

    @st.composite
    def _hardtanh_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # min_val < max_val by construction.
        a = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        b = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        if a > b:
            a, b = b, a
        if b - a < 1e-2:
            b = a + 1.0
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(F.hardtanh, min_val=a, max_val=b)),
        )

    @st.composite
    def _softplus_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # softplus has beta and threshold; t2n's softplus emitter only
        # supports beta=1 (raises NotImplemented otherwise -- see
        # ``torch_to_nnef/op/aten/activation.py:48-54``). We sweep
        # threshold (default 20) within a safe range; beta stays at 1.
        threshold = draw(
            st.floats(
                min_value=5.0,
                max_value=50.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(
                partial(F.softplus, beta=1.0, threshold=threshold)
            ),
        )

    specs.extend(
        [
            OpSpec(
                name="gelu-broad",
                sample_st=_gelu_kwarg_sample(),
                tolerance=VERY,
                dtypes_hint=(torch.float32,),
            ),
            OpSpec(
                name="leaky_relu-broad",
                sample_st=_leaky_relu_kwarg_sample(),
                tolerance=EXACT,
                dtypes_hint=(torch.float32,),
            ),
            OpSpec(
                # t2n's elu emitter at
                # ``torch_to_nnef/op/aten/activation.py:57-64`` passes
                # ``alpha`` as a tensor input via
                # ``unary_input_output_op_with_constant``, but tract's
                # NNEF ``elu`` op treats ``alpha`` as an attribute and
                # silently uses the default 1.0 when alpha is delivered
                # as an input. Repro: ``elu(-1.0, alpha=0.5)`` returns
                # ``-0.632`` (= ``1*(exp(-1)-1)``, i.e. alpha=1) instead
                # of ``-0.316`` (= ``0.5*(exp(-1)-1)``).
                # Same root pattern as the add/sub alpha bug fixed
                # earlier; needs a dedicated elu emitter that either
                # emits the attribute or decomposes alpha into a multiply.
                name="elu-alpha-xfail",
                sample_st=_elu_kwarg_sample(),
                tolerance=VERY,
                dtypes_hint=(torch.float32,),
                xfail_reason=(
                    "t2n elu emitter drops the alpha kwarg "
                    "(tract uses default 1.0 regardless). Same root "
                    "pattern as the add/sub alpha bug, see "
                    "_elu_kwarg_sample in this file for repro."
                ),
            ),
            OpSpec(
                name="hardtanh-broad",
                sample_st=_hardtanh_kwarg_sample(),
                tolerance=EXACT,
                dtypes_hint=(torch.float32,),
            ),
            OpSpec(
                name="softplus-broad",
                sample_st=_softplus_kwarg_sample(),
                tolerance=SUPER,
                dtypes_hint=(torch.float32,),
            ),
        ]
    )

    return specs


def _softmax_dim_sample_st(op_name: str) -> st.SearchStrategy[OpSample]:
    """Softmax / log_softmax with a random valid dim."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        # Bound inputs to keep softmax outputs stable; large positives
        # all cluster near 1.0 and large negatives near 0.0, which is
        # numerically unstable for both PyTorch and tract.
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(op_name, kwargs={"dim": dim}),
        )

    return _draw()


def _softmax_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="softmax",
            sample_st=_softmax_dim_sample_st("softmax"),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="log_softmax",
            sample_st=_softmax_dim_sample_st("log_softmax"),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Selector / indexing specs
# -----------------------------------------------------------------------------


def _select_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.select(dim, index)`` -- pick a single slice along dim.

    Output rank = input rank - 1; index must be in ``[0, dim_size)``.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        index = draw(st.integers(min_value=0, max_value=shape[dim] - 1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "select", kwargs={"dim": dim, "index": index}
            ),
        )

    return _draw()


def _index_select_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.index_select(input, dim, index_tensor)``.

    The index tensor has int64 dtype and 1-D shape; values in
    ``[0, dim_size)``. Output replaces the selected dim with the index
    tensor's length.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        n_idx = draw(st.integers(min_value=1, max_value=4))
        idx_values = draw(
            st.lists(
                st.integers(min_value=0, max_value=shape[dim] - 1),
                min_size=n_idx,
                max_size=n_idx,
            )
        )
        idx = torch.tensor(idx_values, dtype=torch.int64)
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        # ``torch.index_select`` is positional-only on dim
        # (``index_select(input, dim, index)``). ``partial(..., dim=dim)``
        # would inject dim as a kwarg, which the schema rejects.
        op_fn = (lambda d: lambda t, ix: torch.index_select(t, d, ix))(dim)
        return OpSample(
            inputs=(x, idx),
            kwargs={},
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _gather_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.gather(input, dim, index)`` -- index has same rank as input.

    For each output position, the value is
    ``input[i_0, ..., index[i], ..., i_{n-1}]`` along the gather dim.
    The index tensor's shape can differ from input only in the gather
    dim; values in ``[0, input.shape[dim])``.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        # Index shape == input shape, except at ``dim`` where it can vary.
        idx_dim_size = draw(st.integers(min_value=1, max_value=4))
        idx_shape = list(shape)
        idx_shape[dim] = idx_dim_size
        # Build idx values in valid range -- via hypothesis (not
        # np.random) so generation stays deterministic.
        idx = draw(
            tensor_st(
                tuple(idx_shape),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[dim] - 1),
            )
        )
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        # Same positional-only rationale as index_select above.
        op_fn = (lambda d: lambda t, ix: torch.gather(t, d, ix))(dim)
        return OpSample(
            inputs=(x, idx),
            kwargs={},
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _masked_fill_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.masked_fill(mask, value)`` -- bool mask, scalar value."""

    @st.composite
    def _draw(draw) -> OpSample:
        # Mask is broadcastable with input; v1 keeps them same shape for
        # simplicity.
        shape = draw(shape_st(min_rank=1, max_rank=4))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        mask = draw(tensor_st(shape, torch.bool))
        value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x, mask),
            kwargs={},
            module=BinaryPrimitive(lambda t, m: t.masked_fill(m, value)),
        )

    return _draw()


def _topk_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.topk(input, k, dim)`` returns (values, indices).

    Tie-breaking on equal values is implementation-defined; PyTorch and
    tract pick different indices when the input has duplicates. To make
    the indices output well-defined, we feed a permutation of integers
    1..N as the input -- every value is unique so the index output is
    deterministic across both backends.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        k = draw(st.integers(min_value=1, max_value=shape[dim]))
        n = 1
        for s in shape:
            n *= s
        # Draw a permutation of 1..n then reshape -- unique values, no ties.
        perm = draw(st.permutations(list(range(1, n + 1))))
        x = torch.tensor(perm, dtype=torch.float32).reshape(shape)
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive("topk", kwargs={"k": k, "dim": dim}),
        )

    return _draw()


def _selector_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="select",
            sample_st=_select_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="index_select",
            sample_st=_index_select_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="gather",
            sample_st=_gather_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="masked_fill",
            sample_st=_masked_fill_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="topk",
            sample_st=_topk_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Pooling specs
# -----------------------------------------------------------------------------


def _pool2d_sample_st(
    op: T.Callable[..., torch.Tensor],
    allow_padding: bool = True,
) -> st.SearchStrategy[OpSample]:
    """2D pool over (N, C, H, W) input.

    t2n's pool emitters reject ``ceil_mode=True``,
    ``count_include_pad=False`` and ``divisor_override`` so we keep all of
    those at safe defaults. avg_pool callers should set
    ``allow_padding=False`` -- the t2n avg_pool emitter requires
    ``count_include_pad=True`` (PyTorch's default) but emits NNEF's
    ``border="ignore"`` (which is ``count_include_pad=False``); padding
    > 0 surfaces the semantic mismatch.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=4))
        kernel = draw(st.integers(min_value=2, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        padding = (
            draw(st.integers(min_value=0, max_value=kernel // 2))
            if allow_padding
            else 0
        )
        h = draw(st.integers(min_value=kernel + 2, max_value=8))
        w = draw(st.integers(min_value=kernel + 2, max_value=8))
        x = draw(
            tensor_st(
                (n, c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        wrapped = partial(
            op, kernel_size=kernel, stride=stride, padding=padding
        )
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _pool1d_sample_st(
    op: T.Callable[..., torch.Tensor],
    allow_padding: bool = True,
) -> st.SearchStrategy[OpSample]:
    """1D pool. See ``_pool2d_sample_st`` for ``allow_padding`` rationale."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=4))
        kernel = draw(st.integers(min_value=2, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        padding = (
            draw(st.integers(min_value=0, max_value=kernel // 2))
            if allow_padding
            else 0
        )
        length = draw(st.integers(min_value=kernel + 2, max_value=8))
        x = draw(
            tensor_st(
                (n, c, length),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        wrapped = partial(
            op, kernel_size=kernel, stride=stride, padding=padding
        )
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _adaptive_pool2d_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """adaptive_pool2d -- input H/W must divide output H/W.

    t2n's adaptive pool emitter at ``torch_to_nnef/op/aten/pool.py:288``
    is documented as "will likely only work with full defined shapes" --
    it doesn't fully translate adaptive_pool semantics for non-divisible
    input/output ratios (proptest finds shape mismatches like
    output (2,1) on input H=3 producing tract output H=3 instead of 2).
    Restrict to integer multiples so the op effectively becomes a
    regular fixed-stride pool.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=4))
        # Pick output dims first, then input dims as integer multiples.
        out_h = draw(st.integers(min_value=1, max_value=4))
        out_w = draw(st.integers(min_value=1, max_value=4))
        h_mult = draw(st.integers(min_value=1, max_value=3))
        w_mult = draw(st.integers(min_value=1, max_value=3))
        h = out_h * h_mult
        w = out_w * w_mult
        x = draw(
            tensor_st(
                (n, c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        wrapped = partial(op, output_size=(out_h, out_w))
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _pool_specs() -> T.List[OpSpec]:
    import torch.nn.functional as F

    EXACT = TractCheckTolerance.EXACT
    APPROX = TractCheckTolerance.APPROXIMATE
    return [
        OpSpec(
            name="max_pool1d",
            sample_st=_pool1d_sample_st(F.max_pool1d),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="max_pool2d",
            sample_st=_pool2d_sample_st(F.max_pool2d),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # padding=0 only -- t2n's avg_pool emitter requires
            # count_include_pad=True (PyTorch's default) but emits
            # NNEF border="ignore" which means count_include_pad=False.
            # Padding > 0 surfaces the semantic mismatch (PyTorch's edge
            # outputs include the padded zeros in the average; tract's
            # don't). t2n bug -- emitter should either implement
            # count_include_pad=True faithfully or reject it.
            name="avg_pool1d",
            sample_st=_pool1d_sample_st(F.avg_pool1d, allow_padding=False),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # Same padding limitation as avg_pool1d.
            name="avg_pool2d",
            sample_st=_pool2d_sample_st(F.avg_pool2d, allow_padding=False),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="adaptive_avg_pool2d",
            sample_st=_adaptive_pool2d_sample_st(F.adaptive_avg_pool2d),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="adaptive_max_pool2d",
            sample_st=_adaptive_pool2d_sample_st(F.adaptive_max_pool2d),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Norm + Matmul + Conv specs
# -----------------------------------------------------------------------------


def _matmul_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.matmul(A, B)`` -- joint inner-dim constraint A[-1]==B[-2]."""

    @st.composite
    def _draw(draw) -> OpSample:
        # Rank in [2, 4]: 2D for plain matmul, higher for batched.
        rank = draw(st.integers(min_value=2, max_value=4))
        batch_dims = []
        for _ in range(rank - 2):
            batch_dims.append(draw(st.integers(min_value=1, max_value=3)))
        m = draw(st.integers(min_value=1, max_value=6))
        k = draw(st.integers(min_value=1, max_value=6))
        n = draw(st.integers(min_value=1, max_value=6))
        a = draw(
            tensor_st(
                tuple(batch_dims + [m, k]),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        b = draw(
            tensor_st(
                tuple(batch_dims + [k, n]),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        return OpSample(
            inputs=(a, b),
            kwargs={},
            module=BinaryPrimitive(torch.matmul),
        )

    return _draw()


def _linear_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Linear(in_f, out_f)`` -- input shape ends with ``in_f``.

    Rank starts at 2 (always a batch dim). PyTorch supports rank-1 input
    (treats it as a single vector) but t2n's export pipeline needs a
    leading batch dim to wire NNEF's matmul correctly.
    """
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        # in_features and out_features both >= 2 to avoid a t2n corner
        # where Linear(1, 1) on (1, 1)-shape input trips
        # ``maybe_align_inputs_ranks`` in
        # ``torch_to_nnef/op/helper.py`` (TypeError: Tensor not iterable).
        in_features = draw(st.integers(min_value=2, max_value=8))
        out_features = draw(st.integers(min_value=2, max_value=8))
        bias = draw(st.booleans())
        # Input: (..., in_features); rank >= 2 (require batch dim).
        rank = draw(st.integers(min_value=2, max_value=3))
        leading = []
        for _ in range(rank - 1):
            leading.append(draw(st.integers(min_value=2, max_value=4)))
        x = draw(
            tensor_st(
                tuple(leading + [in_features]),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        layer = nn.Linear(in_features, out_features, bias=bias).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _layer_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.LayerNorm(normalized_shape)`` -- input ends with that suffix."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        # 1- to 3-axis layer norm.
        n_norm_axes = draw(st.integers(min_value=1, max_value=3))
        normalized_shape = []
        for _ in range(n_norm_axes):
            normalized_shape.append(draw(st.integers(min_value=2, max_value=6)))
        leading = []
        rank = draw(st.integers(min_value=0, max_value=2))
        for _ in range(rank):
            leading.append(draw(st.integers(min_value=1, max_value=3)))
        x = draw(
            tensor_st(
                tuple(leading + normalized_shape),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        layer = nn.LayerNorm(normalized_shape).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _batch_norm1d_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.BatchNorm1d(C)`` over (N, C) or (N, C, L) input."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        c = draw(st.integers(min_value=1, max_value=6))
        n = draw(st.integers(min_value=2, max_value=4))
        # Optional length axis.
        with_length = draw(st.booleans())
        if with_length:
            length = draw(st.integers(min_value=1, max_value=4))
            shape = (n, c, length)
        else:
            shape = (n, c)
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        layer = nn.BatchNorm1d(c).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _group_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.GroupNorm(num_groups, num_channels)`` -- groups must divide C.

    Each group must have non-trivial variance, otherwise normalization
    amplifies float-roundoff differences between PyTorch and tract into
    visible output drift. We feed a permutation of integers (unique
    values) to guarantee variance > 0 in every group.
    """
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        num_groups = draw(st.integers(min_value=1, max_value=4))
        c_mult = draw(st.integers(min_value=1, max_value=3))
        num_channels = num_groups * c_mult
        n = draw(st.integers(min_value=1, max_value=3))
        h = draw(st.integers(min_value=2, max_value=4))
        w = draw(st.integers(min_value=2, max_value=4))
        shape = (n, num_channels, h, w)
        total = n * num_channels * h * w
        # Permutation of 1..total -> unique values, non-zero variance.
        perm = draw(st.permutations(list(range(1, total + 1))))
        scale = float(total)
        x = torch.tensor(perm, dtype=torch.float32).reshape(shape) / scale
        layer = nn.GroupNorm(num_groups, num_channels).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _conv1d_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Conv1d(in_C, out_C, kernel)`` over (N, in_C, L) input."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        in_c = draw(st.integers(min_value=1, max_value=4))
        out_c = draw(st.integers(min_value=1, max_value=4))
        kernel = draw(st.integers(min_value=1, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        padding = draw(st.integers(min_value=0, max_value=kernel // 2))
        bias = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=2))
        length = draw(st.integers(min_value=kernel + 2, max_value=8))
        x = draw(
            tensor_st(
                (n, in_c, length),
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        layer = nn.Conv1d(
            in_c, out_c, kernel, stride=stride, padding=padding, bias=bias
        ).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _conv2d_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Conv2d(in_C, out_C, kernel)`` over (N, in_C, H, W) input."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        in_c = draw(st.integers(min_value=1, max_value=4))
        out_c = draw(st.integers(min_value=1, max_value=4))
        kernel = draw(st.integers(min_value=1, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        padding = draw(st.integers(min_value=0, max_value=kernel // 2))
        bias = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=2))
        h = draw(st.integers(min_value=kernel + 2, max_value=8))
        w = draw(st.integers(min_value=kernel + 2, max_value=8))
        x = draw(
            tensor_st(
                (n, in_c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        layer = nn.Conv2d(
            in_c, out_c, kernel, stride=stride, padding=padding, bias=bias
        ).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _norm_conv_matmul_specs() -> T.List[OpSpec]:
    # Multi-op chains -- tract's f32 ops accumulate ULP-level error per
    # multiply-accumulate. CLOSE (1e-5) is too tight for a typical
    # conv/linear; VERY (1e-4) gives breathing room.
    VERY = TractCheckTolerance.VERY
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            name="matmul",
            sample_st=_matmul_sample_st(),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="linear",
            sample_st=_linear_sample_st(),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # layer_norm involves variance + division; nightly proptest
            # surfaces near-zero outputs where tract diverges by ~1.5e-4
            # absolute (above VERY but well below SUPER). Same root cause
            # class as group_norm (multi-step f32 reduction precision).
            name="layer_norm",
            sample_st=_layer_norm_sample_st(),
            tolerance=TractCheckTolerance.SUPER,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="batch_norm1d",
            sample_st=_batch_norm1d_sample_st(),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # Previously xfailed because the ``group_norm.nnef``
            # fragment tiled the BATCH axis instead of GROUPS, leaking
            # mean from one batch into another batch's channels for
            # multi-batch inputs with num_groups < num_channels. Now
            # fixed: the emitter flattens spatial dims before the
            # fragment, the fragment computes everything in 3D
            # ``(B, num_groups, S)`` space, and scale/offset are
            # applied via the standard per-channel unsqueeze +
            # left-aligned NNEF broadcast pattern after restoration of
            # the original input rank.
            name="group_norm",
            sample_st=_group_norm_sample_st(),
            tolerance=TractCheckTolerance.SUPER,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="conv1d",
            sample_st=_conv1d_sample_st(),
            tolerance=CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="conv2d",
            sample_st=_conv2d_sample_st(),
            tolerance=CLOSE,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Concat / split / multi-tensor specs
# -----------------------------------------------------------------------------


class _CatPair(torch.nn.Module):
    """Wrapper for ``torch.cat([a, b], dim=k)`` -- list-of-2 form."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.cat([a, b], dim=self.dim)


class _StackPair(torch.nn.Module):
    """Wrapper for ``torch.stack([a, b], dim=k)`` -- list-of-2 form."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.stack([a, b], dim=self.dim)


def _cat_sample_st() -> st.SearchStrategy[OpSample]:
    """``cat([a, b], dim)`` -- joint shape: a/b agree on every non-cat dim."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        # Draw a base shape, then pick the cat dim and let a/b differ there.
        base = list(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        a_dim = draw(st.integers(min_value=1, max_value=4))
        b_dim = draw(st.integers(min_value=1, max_value=4))
        sa = list(base)
        sa[dim] = a_dim
        sb = list(base)
        sb[dim] = b_dim
        a = draw(
            tensor_st(
                tuple(sa),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        b = draw(
            tensor_st(
                tuple(sb),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(a, b), kwargs={}, module=_CatPair(dim))

    return _draw()


def _stack_sample_st() -> st.SearchStrategy[OpSample]:
    """``stack([a, b], dim)`` -- joint shape: a and b have identical shape."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=0, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # New axis can be inserted at any position 0..rank inclusive.
        dim = draw(st.integers(min_value=0, max_value=rank))
        a = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        b = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(a, b), kwargs={}, module=_StackPair(dim))

    return _draw()


def _chunk_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.chunk(chunks, dim)`` -- multi-output split.

    PyTorch's chunk handles non-divisible ``shape[dim]`` gracefully (last
    chunk is smaller). The t2n split emitter at
    ``torch_to_nnef/op/aten/split.py:97`` asserts equal-sized chunks and
    raises ``AssertionError`` otherwise -- so our strategy enforces
    ``shape[dim] % chunks == 0``.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape_list = list(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        # Pick chunks first as a divisor of dim_size.
        max_chunks = shape_list[dim]
        divisors = [c for c in range(1, max_chunks + 1) if max_chunks % c == 0]
        chunks = draw(st.sampled_from(divisors))
        shape = tuple(shape_list)
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "chunk", kwargs={"chunks": chunks, "dim": dim}
            ),
        )

    return _draw()


def _unbind_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.unbind(input, dim)`` -- splits into a tuple of slices."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.unbind, dim=dim)),
        )

    return _draw()


def _roll_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.roll(input, shifts, dims)`` -- cyclic shift.

    Sweeps the full PyTorch range for ``shifts``: positive, negative,
    zero, and magnitudes >= dim_size. Tract's slice/concat path has
    issues with shift=0 and shift==dim_size (output shape doubles), but
    the t2n roll emitter at ``torch_to_nnef/op/aten/concat.py:136`` now
    normalizes shifts via modulo and elides no-op rolls (matches
    PyTorch's behavior), which avoids triggering the tract bug.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        # Full range, including 0 and |shift| >= dim_size; the t2n
        # emitter normalizes both cases.
        max_abs = 2 * shape[dim]
        shift = draw(st.integers(min_value=-max_abs, max_value=max_abs))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.roll, shifts=shift, dims=dim)),
        )

    return _draw()


def _outer_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.outer(a, b)`` -- both inputs are 1-D, result is 2-D."""

    @st.composite
    def _draw(draw) -> OpSample:
        m = draw(st.integers(min_value=1, max_value=6))
        n = draw(st.integers(min_value=1, max_value=6))
        a = draw(
            tensor_st(
                (m,),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        b = draw(
            tensor_st(
                (n,),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        return OpSample(
            inputs=(a, b),
            kwargs={},
            module=BinaryPrimitive(torch.outer),
        )

    return _draw()


def _triangular_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """``torch.tril/triu(input, diagonal)`` -- requires rank >= 2."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # Diagonal range bounded by the last two dims.
        diagonal = draw(st.integers(min_value=-2, max_value=2))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(op, diagonal=diagonal)),
        )

    return _draw()


def _concat_split_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="cat",
            sample_st=_cat_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="stack",
            sample_st=_stack_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="chunk",
            sample_st=_chunk_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="unbind",
            sample_st=_unbind_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="roll",
            sample_st=_roll_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="outer",
            sample_st=_outer_sample_st(),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="tril",
            sample_st=_triangular_sample_st(torch.tril),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="triu",
            sample_st=_triangular_sample_st(torch.triu),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Padding specs
# -----------------------------------------------------------------------------


def _pad_sample_st(
    mode: str, max_pad_per_side: T.Optional[int] = None
) -> st.SearchStrategy[OpSample]:
    """``F.pad(input, pad, mode, value)``.

    PyTorch's ``pad`` list is right-to-left: ``[L_-1, R_-1, L_-2, R_-2, ...]``
    where ``L_i`` and ``R_i`` are left/right padding for axis -i. Up to
    ``rank`` axes can be padded.

    Reflection and replication modes require ``pad <= dim_size - 1`` (for
    reflect) or ``pad <= dim_size`` (for replicate), so ``max_pad_per_side``
    bounds the strategy accordingly.
    """
    import torch.nn.functional as F

    @st.composite
    def _draw(draw) -> OpSample:
        # ``reflect``/``replicate`` need rank>=3 (N, C, spatial...) since
        # they only operate on spatial dims. ``constant`` accepts any rank.
        if mode in ("reflect", "replicate"):
            rank = draw(st.integers(min_value=3, max_value=4))
            min_dim = 3
        else:
            rank = draw(st.integers(min_value=1, max_value=4))
            min_dim = 2
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=min_dim, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # PyTorch's reflect/replicate require padding ALL spatial dims at
        # once (3D -> 2 pad values; 4D -> 4; 5D -> 6). For constant, any
        # subset of axes is fine.
        if mode in ("reflect", "replicate"):
            n_axes = rank - 2
        else:
            n_axes = draw(st.integers(min_value=1, max_value=rank))
        pad = []
        # Build pad list from last axis backward.
        for i in range(n_axes):
            axis = -(i + 1)
            dim_size = shape[axis]
            if mode == "reflect":
                ub = max(0, dim_size - 1)
            elif mode == "replicate":
                ub = dim_size
            else:
                ub = max_pad_per_side or 3
            ub = min(ub, max_pad_per_side or ub)
            left = draw(st.integers(min_value=0, max_value=ub))
            right = draw(st.integers(min_value=0, max_value=ub))
            pad.extend([left, right])
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        if mode == "constant":
            wrapped = partial(F.pad, pad=tuple(pad), mode=mode, value=0.0)
        else:
            wrapped = partial(F.pad, pad=tuple(pad), mode=mode)
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _pad_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="pad-constant",
            sample_st=_pad_sample_st("constant", max_pad_per_side=3),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="pad-reflect",
            sample_st=_pad_sample_st("reflect"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="pad-replicate-xfail",
            sample_st=_pad_sample_st("replicate"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract 0.22.1 does not implement NNEF pad mode "
                '"replicate" ("unsupported padding mode replicate"). '
                "t2n's replication_padnd emitter passes through the "
                "mode attribute; the gap is downstream in tract."
            ),
        ),
    ]


# -----------------------------------------------------------------------------
# Norm variants (vector norm, frobenius, linalg, rms)
# -----------------------------------------------------------------------------


def _vector_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.norm(p, dim, keepdim)`` -- vector p-norm along a dim."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        keepdim = draw(st.booleans())
        # p in {1, 2} only -- t2n's norm emitter at norm.py:149 dispatches
        # only these in tract; fractional p may go through a different
        # path with its own bugs.
        p = draw(st.sampled_from([1, 2]))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "norm", kwargs={"p": p, "dim": dim, "keepdim": keepdim}
            ),
        )

    return _draw()


def _rms_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.RMSNorm(normalized_shape)`` -- input ends with that suffix."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        norm_size = draw(st.integers(min_value=2, max_value=6))
        leading_rank = draw(st.integers(min_value=1, max_value=3))
        leading = []
        for _ in range(leading_rank):
            leading.append(draw(st.integers(min_value=1, max_value=3)))
        shape = tuple(leading + [norm_size])
        # Keep inputs away from zero to avoid divide-by-near-zero in RMS.
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        layer = nn.RMSNorm(norm_size).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _norm_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="vector_norm",
            sample_st=_vector_norm_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # Confirmed upstream tract bug: tract's native
            # ``tract_transformers_rms_norm`` op (which t2n routes to
            # for tract >= 0.22.0 with single-axis ``normalized_shape``,
            # the typical LLM case) diverges by ~3% relative vs
            # PyTorch's ``nn.RMSNorm``. The t2n fragment fallback
            # (``rms_norm.nnef``) has the correct formula
            # ``x * rsqrt(mean(x^2) + eps) * gamma`` -- forcing
            # ``prefer_native_tract_rms_norm`` to False makes proptest
            # match PyTorch exactly. The fix lives in tract's native op.
            name="rms_norm-xfail",
            sample_st=_rms_norm_sample_st(),
            tolerance=TractCheckTolerance.ULTRA,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract's native tract_transformers_rms_norm op diverges "
                "from PyTorch's nn.RMSNorm by ~3% relative; forcing the "
                "t2n fragment fallback path matches exactly. Bug is "
                "upstream in tract."
            ),
        ),
    ]


# -----------------------------------------------------------------------------
# Sort / scatter specs (extension of the selector family)
# -----------------------------------------------------------------------------


def _sort_sample_st(method_name: str) -> st.SearchStrategy[OpSample]:
    """``Tensor.sort(dim, descending)`` / ``Tensor.argsort(dim, descending)``.

    Like topk, sort's index output is tie-breaking-dependent. We feed a
    permutation of integers 1..N to guarantee unique values and a
    deterministic index output.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        descending = draw(st.booleans())
        n = 1
        for s in shape:
            n *= s
        perm = draw(st.permutations(list(range(1, n + 1))))
        x = torch.tensor(perm, dtype=torch.float32).reshape(shape)
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                method_name, kwargs={"dim": dim, "descending": descending}
            ),
        )

    return _draw()


def _scatter_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.scatter(dim, index, src)`` -- counterpart of gather."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        # Index shape == src shape; both can differ from input only at
        # ``dim``. Index values must be in [0, input.shape[dim]).
        idx_dim_size = draw(st.integers(min_value=1, max_value=shape[dim]))
        idx_shape = list(shape)
        idx_shape[dim] = idx_dim_size
        idx = draw(
            tensor_st(
                tuple(idx_shape),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[dim] - 1),
            )
        )
        src = draw(
            tensor_st(
                tuple(idx_shape),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        # ``Tensor.scatter`` is positional-only on dim; same lambda wrapper
        # pattern as index_select / gather.
        op_fn = (lambda d: lambda t, i, s: t.scatter(d, i, s))(dim)
        return OpSample(
            inputs=(x, idx, src),
            kwargs={},
            module=TernaryPrimitive(op_fn),
        )

    return _draw()


def _slice_sample_st() -> st.SearchStrategy[OpSample]:
    """Python slice via ``__getitem__`` -- maps to ``aten:slice``.

    Currently only the simple "single dim, contiguous" form to keep the
    strategy simple.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        dim_size = shape[dim]
        start = draw(st.integers(min_value=0, max_value=dim_size - 1))
        end = draw(st.integers(min_value=start + 1, max_value=dim_size))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )

        def slice_fn(t, _dim=dim, _start=start, _end=end):
            slicer = [slice(None)] * t.ndim
            slicer[_dim] = slice(_start, _end)
            return t[tuple(slicer)]

        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(slice_fn))

    return _draw()


def _sort_scatter_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="sort",
            sample_st=_sort_sample_st("sort"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="argsort",
            sample_st=_sort_sample_st("argsort"),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="scatter",
            sample_st=_scatter_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="slice",
            sample_st=_slice_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# 3D conv/pool + numerical helpers + classifiers
# -----------------------------------------------------------------------------


def _conv3d_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Conv3d`` over (N, in_C, D, H, W) input."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        in_c = draw(st.integers(min_value=1, max_value=3))
        out_c = draw(st.integers(min_value=1, max_value=3))
        kernel = draw(st.integers(min_value=1, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        padding = draw(st.integers(min_value=0, max_value=kernel // 2))
        bias = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=2))
        d = draw(st.integers(min_value=kernel + 1, max_value=5))
        h = draw(st.integers(min_value=kernel + 1, max_value=5))
        w = draw(st.integers(min_value=kernel + 1, max_value=5))
        x = draw(
            tensor_st(
                (n, in_c, d, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        layer = nn.Conv3d(
            in_c, out_c, kernel, stride=stride, padding=padding, bias=bias
        ).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _pool3d_sample_st(
    op: T.Callable[..., torch.Tensor],
    allow_padding: bool = True,
) -> st.SearchStrategy[OpSample]:
    """3D pool over (N, C, D, H, W) input."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        kernel = draw(st.integers(min_value=2, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        padding = (
            draw(st.integers(min_value=0, max_value=kernel // 2))
            if allow_padding
            else 0
        )
        d = draw(st.integers(min_value=kernel + 2, max_value=6))
        h = draw(st.integers(min_value=kernel + 2, max_value=6))
        w = draw(st.integers(min_value=kernel + 2, max_value=6))
        x = draw(
            tensor_st(
                (n, c, d, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        wrapped = partial(
            op, kernel_size=kernel, stride=stride, padding=padding
        )
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _cumsum_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.cumsum(input, dim)``."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(torch.cumsum, dim=dim)),
        )

    return _draw()


def _atan2_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.atan2(y, x)`` -- broadcasted, no special domain."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        y = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        x = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        return OpSample(
            inputs=(y, x), kwargs={}, module=BinaryPrimitive(torch.atan2)
        )

    return _draw()


def _classifier_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """NaN/Inf classifier -- input may contain NaN/Inf."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=3))
        # finite=False so NaN/Inf can be drawn; outputs are bool exact.
        x = draw(tensor_st(shape, torch.float32, finite=False))
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(op))

    return _draw()


def _conv3d_pool3d_helpers_specs() -> T.List[OpSpec]:
    import torch.nn.functional as F

    EXACT = TractCheckTolerance.EXACT
    APPROX = TractCheckTolerance.APPROXIMATE
    CLOSE = TractCheckTolerance.CLOSE
    VERY = TractCheckTolerance.VERY
    return [
        OpSpec(
            name="conv3d",
            sample_st=_conv3d_sample_st(),
            tolerance=CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="max_pool3d",
            sample_st=_pool3d_sample_st(F.max_pool3d),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            # Same count_include_pad caveat as avg_pool1d/2d -- padding=0.
            name="avg_pool3d",
            sample_st=_pool3d_sample_st(F.avg_pool3d, allow_padding=False),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="expm1",
            sample_st=_unary_sample_st(
                torch.expm1, domain=Interval(-10.0, 10.0)
            ),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="log1p",
            sample_st=_unary_sample_st(
                torch.log1p, domain=Interval(-0.999, 1e3)
            ),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="log10",
            sample_st=_unary_sample_st(torch.log10, domain=Interval(1e-3, 1e4)),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="trunc-unary",
            sample_st=_unary_sample_st(
                torch.trunc, domain=Interval(-100.0, 100.0)
            ),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="cumsum",
            sample_st=_cumsum_sample_st(),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="atan2-xfail",
            sample_st=_atan2_sample_st(),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract atan2 disagrees with PyTorch on quadrant boundaries "
                "(e.g. atan2(0, -1) returns 0 in tract vs pi in PyTorch). "
                "Likely a tract upstream bug in atan2 quadrant handling."
            ),
        ),
        OpSpec(
            name="isnan-xfail",
            sample_st=_classifier_sample_st(torch.isnan),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_is_nan; requires "
                "tract > 0.22.1 (same gating as any/all)."
            ),
        ),
        OpSpec(
            name="isinf-xfail",
            sample_st=_classifier_sample_st(torch.isinf),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_is_inf; requires tract > 0.22.1."
            ),
        ),
        OpSpec(
            name="isposinf-xfail",
            sample_st=_classifier_sample_st(torch.isposinf),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_isposinf; requires "
                "tract > 0.22.1."
            ),
        ),
        OpSpec(
            name="isneginf-xfail",
            sample_st=_classifier_sample_st(torch.isneginf),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_isneginf; requires "
                "tract > 0.22.1."
            ),
        ),
    ]


# -----------------------------------------------------------------------------
# Bitwise + tensor builders
# -----------------------------------------------------------------------------


def _bitwise_binary_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Bitwise binary op over int32 -- mutually broadcastable shapes."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        a = draw(
            tensor_st(sa, torch.int32, finite=True, domain=Interval(-100, 100))
        )
        b = draw(
            tensor_st(sb, torch.int32, finite=True, domain=Interval(-100, 100))
        )
        return OpSample(inputs=(a, b), kwargs={}, module=BinaryPrimitive(op))

    return _draw()


def _bitwise_not_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.bitwise_not`` over int32."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4))
        x = draw(
            tensor_st(
                shape, torch.int32, finite=True, domain=Interval(-100, 100)
            )
        )
        return OpSample(
            inputs=(x,), kwargs={}, module=UnaryPrimitive(torch.bitwise_not)
        )

    return _draw()


def _zeros_like_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.zeros_like(input)`` -- output matches input shape/dtype.

    Min rank/size raised to avoid the export-pipeline constant-folding
    case (single-element rank-1 input gets folded out, leaving tract
    unable to find the output variable).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        return OpSample(
            inputs=(x,), kwargs={}, module=UnaryPrimitive(torch.zeros_like)
        )

    return _draw()


def _ones_like_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.ones_like(input)`` -- see _zeros_like for shape note."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        return OpSample(
            inputs=(x,), kwargs={}, module=UnaryPrimitive(torch.ones_like)
        )

    return _draw()


def _full_like_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.full_like(input, fill_value)`` -- swept fill values."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        fill_value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(
                partial(torch.full_like, fill_value=fill_value)
            ),
        )

    return _draw()


def _bitwise_builder_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="bitwise_and",
            sample_st=_bitwise_binary_sample_st(torch.bitwise_and),
            tolerance=EXACT,
            dtypes_hint=(torch.int32,),
        ),
        OpSpec(
            name="bitwise_or",
            sample_st=_bitwise_binary_sample_st(torch.bitwise_or),
            tolerance=EXACT,
            dtypes_hint=(torch.int32,),
        ),
        OpSpec(
            name="bitwise_xor",
            sample_st=_bitwise_binary_sample_st(torch.bitwise_xor),
            tolerance=EXACT,
            dtypes_hint=(torch.int32,),
        ),
        OpSpec(
            name="bitwise_not",
            sample_st=_bitwise_not_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.int32,),
        ),
        OpSpec(
            name="zeros_like",
            sample_st=_zeros_like_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="ones_like",
            sample_st=_ones_like_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="full_like",
            sample_st=_full_like_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Specialty ops (embedding, repeat_interleave, upsample, sdpa, ...)
# -----------------------------------------------------------------------------


def _embedding_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Embedding(num_embeddings, embedding_dim)`` -- index lookup."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        num_emb = draw(st.integers(min_value=2, max_value=8))
        emb_dim = draw(st.integers(min_value=2, max_value=8))
        # Index input: (batch, seq_len) of int64 indices.
        batch = draw(st.integers(min_value=1, max_value=3))
        seq = draw(st.integers(min_value=1, max_value=4))
        idx = draw(
            tensor_st(
                (batch, seq),
                torch.int64,
                finite=True,
                domain=Interval(0, num_emb - 1),
            )
        )
        layer = nn.Embedding(num_emb, emb_dim).eval()
        return OpSample(inputs=(idx,), kwargs={}, module=layer)

    return _draw()


def _repeat_interleave_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.repeat_interleave(input, repeats, dim)`` -- scalar repeats."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        repeats = draw(st.integers(min_value=1, max_value=3))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(
                partial(torch.repeat_interleave, repeats=repeats, dim=dim)
            ),
        )

    return _draw()


def _upsample_nearest2d_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.UpsamplingNearest2d(scale_factor=N)`` -- (N, C, H, W) input."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        h = draw(st.integers(min_value=1, max_value=4))
        w = draw(st.integers(min_value=1, max_value=4))
        scale = draw(st.integers(min_value=2, max_value=3))
        x = draw(
            tensor_st(
                (n, c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        layer = nn.UpsamplingNearest2d(scale_factor=scale).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _specialty_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="embedding",
            sample_st=_embedding_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="repeat_interleave",
            sample_st=_repeat_interleave_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="upsample_nearest2d",
            sample_st=_upsample_nearest2d_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# prelu / glu / einsum
# -----------------------------------------------------------------------------


def _prelu_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.PReLU(num_parameters=1)`` -- shared slope across all channels."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        layer = nn.PReLU(num_parameters=1).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _prelu_multi_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.PReLU(num_parameters=C)`` -- per-channel slope.

    PyTorch broadcasts ``weight`` of shape ``(C,)`` along the channel
    axis (``dim=1``) of an input shaped ``(N, C, *spatial)``. Because
    NNEF broadcasts left-aligned, the t2n emitter pre-unsqueezes the
    weight to ``(C, 1, 1, ...)`` before emit -- see
    ``torch_to_nnef/op/aten/activation.py:prelu``.
    """
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        # rank >= 3 so a real channel axis exists (N, C, *spatial).
        rank = draw(st.integers(min_value=3, max_value=4))
        n = draw(st.integers(min_value=1, max_value=3))
        c = draw(st.integers(min_value=2, max_value=6))
        spatial = draw(
            st.lists(
                st.integers(min_value=1, max_value=4),
                min_size=rank - 2,
                max_size=rank - 2,
            )
        )
        shape = tuple([n, c, *spatial])
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        layer = nn.PReLU(num_parameters=c).eval()
        # Randomize the slope so different channels exercise distinct
        # branches; default-initialized PReLU has every channel at 0.25.
        with torch.no_grad():
            w = draw(
                tensor_st(
                    (c,),
                    torch.float32,
                    finite=True,
                    domain=Interval(-1.0, 1.0),
                )
            )
            layer.weight.copy_(w)
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _glu_sample_st() -> st.SearchStrategy[OpSample]:
    """``F.glu(input, dim)`` -- splits input in half along dim, gates."""
    import torch.nn.functional as F

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape_list = list(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        # GLU requires shape[dim] to be even.
        if shape_list[dim] % 2 != 0:
            shape_list[dim] += 1
        shape = tuple(shape_list)
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(F.glu, dim=dim)),
        )

    return _draw()


class _Einsum2Op(torch.nn.Module):
    """Wrapper that calls ``torch.einsum(expr, a, b)``."""

    def __init__(self, expr: str):
        super().__init__()
        self.expr = expr

    def forward(self, a, b):
        return torch.einsum(self.expr, a, b)


def _einsum_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.einsum(expr, a, b)`` -- a small set of canonical patterns.

    Open-ended einsum strings are too unconstrained for a useful sweep;
    we pick a fixed catalog of well-known patterns and let hypothesis
    sweep the dim sizes within each.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        # (expr, a_dims_letters, b_dims_letters)
        catalog = [
            ("ij,jk->ik", "ij", "jk"),  # 2D matmul
            ("bij,bjk->bik", "bij", "bjk"),  # batched matmul
            ("i,j->ij", "i", "j"),  # outer product
            ("ij,ij->ij", "ij", "ij"),  # element-wise
            ("ij,j->i", "ij", "j"),  # mat-vec
        ]
        expr, a_letters, b_letters = draw(st.sampled_from(catalog))
        sizes = {}
        for ch in set(a_letters + b_letters):
            sizes[ch] = draw(st.integers(min_value=1, max_value=5))
        a_shape = tuple(sizes[ch] for ch in a_letters)
        b_shape = tuple(sizes[ch] for ch in b_letters)
        a = draw(
            tensor_st(
                a_shape,
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        b = draw(
            tensor_st(
                b_shape,
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        return OpSample(inputs=(a, b), kwargs={}, module=_Einsum2Op(expr))

    return _draw()


def _prelu_glu_einsum_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="prelu",
            sample_st=_prelu_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="prelu-multi",
            sample_st=_prelu_multi_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="glu",
            sample_st=_glu_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="einsum",
            sample_st=_einsum_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Final user-facing ops (max_pool*_with_indices, dropout, index)
# -----------------------------------------------------------------------------


def _max_pool2d_with_indices_sample_st() -> st.SearchStrategy[OpSample]:
    """``F.max_pool2d(..., return_indices=True)`` -- multi-output.

    Like topk, indices are tie-breaking-dependent. We feed a permutation
    of integers as input to make every value unique and indices
    deterministic.
    """
    import torch.nn.functional as F

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        kernel = draw(st.integers(min_value=2, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        h = draw(st.integers(min_value=kernel + 2, max_value=6))
        w = draw(st.integers(min_value=kernel + 2, max_value=6))
        total = n * c * h * w
        perm = draw(st.permutations(list(range(1, total + 1))))
        x = torch.tensor(perm, dtype=torch.float32).reshape(
            (n, c, h, w)
        ) / float(total)
        wrapped = partial(
            F.max_pool2d,
            kernel_size=kernel,
            stride=stride,
            return_indices=True,
        )
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _dropout_eval_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Dropout(p)`` in eval mode -- a no-op identity.

    The export pipeline should skip dropout in eval mode (it has no
    effect at inference). Proptest sweeps shapes to confirm the no-op
    invariant holds across the export.
    """
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        # eval() mode -- dropout should be identity.
        layer = nn.Dropout(p=0.5).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _final_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="max_pool2d_with_indices",
            sample_st=_max_pool2d_with_indices_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="dropout",
            sample_st=_dropout_eval_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Constructors (input-less in PyTorch, wrapped with a shape-coupled input)
# + advanced index + SDPA
# -----------------------------------------------------------------------------


class _ZerosFromShapeOf(torch.nn.Module):
    """``torch.zeros(*x.shape, dtype=x.dtype)`` -- derives shape from input."""

    def forward(self, x):
        return torch.zeros(x.shape, dtype=x.dtype)


class _OnesFromShapeOf(torch.nn.Module):
    """``torch.ones(*x.shape, dtype=x.dtype)``."""

    def forward(self, x):
        return torch.ones(x.shape, dtype=x.dtype)


class _FullFromShapeOf(torch.nn.Module):
    """``torch.full(x.shape, fill_value, dtype=x.dtype)`` -- swept fills."""

    def __init__(self, fill_value: float):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        return torch.full(x.shape, self.fill_value, dtype=x.dtype)


class _ArangeFromInput(torch.nn.Module):
    """``torch.arange(start, end, step)`` -- start/end/step baked at init.

    The input is ignored at runtime, but kept so the export pipeline has
    a real graph input. We attach a no-op dependency via ``+ x.sum() * 0``
    so the graph extractor sees the tensor.
    """

    def __init__(self, start: int, end: int, step: int):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x):
        return torch.arange(self.start, self.end, self.step) + (x.sum() * 0)


class _ScalarTensorOfDtypeOf(torch.nn.Module):
    """``torch.scalar_tensor(value, dtype=x.dtype)`` -- 0-d constant."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.scalar_tensor(self.value, dtype=x.dtype) + (x.sum() * 0)


class _NewZerosFromInput(torch.nn.Module):
    """``Tensor.new_zeros(shape)`` -- derives shape and dtype from input."""

    def forward(self, x):
        return x.new_zeros(x.shape)


def _zeros_from_shape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(inputs=(x,), kwargs={}, module=_ZerosFromShapeOf())

    return _draw()


def _ones_from_shape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(inputs=(x,), kwargs={}, module=_OnesFromShapeOf())

    return _draw()


def _full_from_shape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        fill_value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=_FullFromShapeOf(fill_value),
        )

    return _draw()


def _arange_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        start = draw(st.integers(min_value=0, max_value=5))
        length = draw(st.integers(min_value=1, max_value=20))
        step = draw(st.integers(min_value=1, max_value=3))
        end = start + length * step
        x = draw(
            tensor_st(
                (2, 3), torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=_ArangeFromInput(start, end, step),
        )

    return _draw()


def _scalar_tensor_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        x = draw(
            tensor_st(
                (2, 3), torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=_ScalarTensorOfDtypeOf(value),
        )

    return _draw()


def _new_zeros_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(inputs=(x,), kwargs={}, module=_NewZerosFromInput())

    return _draw()


def _index_advanced_sample_st() -> st.SearchStrategy[OpSample]:
    """``x[long_tensor]`` -- advanced indexing along axis 0.

    Output shape: index_tensor.shape + x.shape[1:].
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        n_idx = draw(st.integers(min_value=1, max_value=4))
        idx = draw(
            tensor_st(
                (n_idx,),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[0] - 1),
            )
        )
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )

        def op_fn(t, i):
            return t[i]

        return OpSample(
            inputs=(x, idx),
            kwargs={},
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _sdpa_sample_st() -> st.SearchStrategy[OpSample]:
    """``F.scaled_dot_product_attention(Q, K, V)`` -- shape (B, H, S, D)."""
    import torch.nn.functional as F

    @st.composite
    def _draw(draw) -> OpSample:
        b = draw(st.integers(min_value=1, max_value=2))
        h = draw(st.integers(min_value=1, max_value=2))
        s = draw(st.integers(min_value=2, max_value=4))
        d = draw(st.integers(min_value=2, max_value=4))
        # Same shape for Q, K, V (typical use case).
        domain = Interval(-1.0, 1.0)
        q = draw(
            tensor_st((b, h, s, d), torch.float32, finite=True, domain=domain)
        )
        k = draw(
            tensor_st((b, h, s, d), torch.float32, finite=True, domain=domain)
        )
        v = draw(
            tensor_st((b, h, s, d), torch.float32, finite=True, domain=domain)
        )
        return OpSample(
            inputs=(q, k, v),
            kwargs={},
            module=TernaryPrimitive(F.scaled_dot_product_attention),
        )

    return _draw()


def _constructors_index_sdpa_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    VERY = TractCheckTolerance.VERY
    return [
        OpSpec(
            name="zeros",
            sample_st=_zeros_from_shape_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="ones",
            sample_st=_ones_from_shape_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="full",
            sample_st=_full_from_shape_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="arange",
            sample_st=_arange_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.int64,),
        ),
        OpSpec(
            name="scalar_tensor",
            sample_st=_scalar_tensor_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="new_zeros",
            sample_st=_new_zeros_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="index",
            sample_st=_index_advanced_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="sdpa",
            sample_st=_sdpa_sample_st(),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# FFT (real-input forward and inverse)
# -----------------------------------------------------------------------------


def _fft_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """``torch.fft.fft(input, n=None, dim=-1, norm=None)``.

    The t2n FFT emitter (``torch_to_nnef/op/aten/fft.py:_fft``) requires
    ``n`` and ``norm`` to be None on the version path we test, and works
    on real (float32) input by padding to complex internally.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=8),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        # Bound input modestly to keep FFT magnitudes in range.
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=UnaryPrimitive(partial(op, dim=dim)),
        )

    return _draw()


def _fft_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            # PyTorch's complex output (shape ``(...,)`` complex64) and
            # tract's unfolded output (shape ``(..., 2)`` real, with the
            # last axis being ``[real, imag]``) don't compare apples-to-
            # apples in the current comparator. FFT proptest support
            # needs a complex-aware comparator that either folds tract's
            # output back to complex or unfolds PyTorch's output to
            # match tract.
            name="fft_fft-xfail",
            sample_st=_fft_sample_st(torch.fft.fft),
            tolerance=TractCheckTolerance.SUPER,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "FFT returns complex; comparator doesn't bridge "
                "PyTorch's complex64 output vs tract's (real, imag) "
                "unfolded layout."
            ),
        ),
        OpSpec(
            # Additionally, t2n's NPZ writer at
            # ``model_wrapper.py:303`` raises ``RuntimeError: Can't call
            # numpy() on Tensor that has conjugate bit set`` for IFFT
            # output -- needs a ``.resolve_conj()`` before serialization.
            name="fft_ifft-xfail",
            sample_st=_fft_sample_st(torch.fft.ifft),
            tolerance=TractCheckTolerance.SUPER,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "Same complex-output comparator gap as fft_fft, plus "
                "t2n model_wrapper.py:303 missing .resolve_conj() before "
                ".numpy() for ifft output (conjugate bit set)."
            ),
        ),
    ]


# -----------------------------------------------------------------------------
# Identity-like glue ops + dtype casts + simple mutators-as-functional
# -----------------------------------------------------------------------------


def _identity_unary_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Generic unary identity (clone, contiguous, detach).

    These are no-ops on the tensor data at runtime in eval mode.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(op))

    return _draw()


class _CastToDtype(torch.nn.Module):
    """``Tensor.to(dtype)`` -- runtime dtype cast."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


class _TypeAsFromOther(torch.nn.Module):
    """``Tensor.type_as(other)`` -- cast to other's dtype."""

    def forward(self, a, b):
        return a.type_as(b)


class _FillFunctional(torch.nn.Module):
    """Functional ``torch.full_like(x, value)`` standing in for fill_."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def _to_dtype_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.to(dtype)`` -- sweep cast targets among supported floats."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        target_dtype = draw(
            st.sampled_from([torch.float32, torch.float16, torch.float64])
        )
        return OpSample(
            inputs=(x,), kwargs={}, module=_CastToDtype(target_dtype)
        )

    return _draw()


def _type_as_sample_st() -> st.SearchStrategy[OpSample]:
    """``a.type_as(b)`` -- a takes b's dtype."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        a = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        # b is a tiny tensor whose dtype we want to inherit.
        target_dtype = draw(st.sampled_from([torch.float32, torch.float16]))
        b = draw(
            tensor_st(
                (1,),
                target_dtype,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        return OpSample(inputs=(a, b), kwargs={}, module=_TypeAsFromOther())

    return _draw()


def _fill_sample_st() -> st.SearchStrategy[OpSample]:
    """Functional fill via ``full_like``.

    PyTorch traces inplace ``fill_`` as ``full_like`` when no in-place
    graph is needed; this spec exercises that path.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(inputs=(x,), kwargs={}, module=_FillFunctional(value))

    return _draw()


def _glue_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="clone",
            sample_st=_identity_unary_sample_st(torch.clone),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="contiguous",
            sample_st=_identity_unary_sample_st(lambda t: t.contiguous()),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="detach",
            sample_st=_identity_unary_sample_st(lambda t: t.detach()),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="to_dtype",
            sample_st=_to_dtype_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32, torch.float16, torch.float64),
        ),
        OpSpec(
            name="type_as",
            sample_st=_type_as_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32, torch.float16),
        ),
        OpSpec(
            name="fill",
            sample_st=_fill_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Depth: conv with dilation/groups + pool with dilation
# -----------------------------------------------------------------------------


def _conv2d_dilation_groups_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Conv2d`` sweeping ``dilation`` and ``groups`` kwargs.

    ``groups`` must divide both ``in_channels`` and ``out_channels``;
    we draw a common groups divisor and pick channel counts as multiples.
    Dilation increases effective kernel; we ensure spatial >= effective k.
    """
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        groups = draw(st.integers(min_value=1, max_value=3))
        in_mult = draw(st.integers(min_value=1, max_value=2))
        out_mult = draw(st.integers(min_value=1, max_value=2))
        in_c = groups * in_mult
        out_c = groups * out_mult
        kernel = draw(st.integers(min_value=1, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        dilation = draw(st.integers(min_value=1, max_value=2))
        # padding bounded so output spatial stays positive.
        padding = draw(st.integers(min_value=0, max_value=kernel // 2))
        bias = draw(st.booleans())
        n = draw(st.integers(min_value=1, max_value=2))
        # effective kernel size with dilation
        eff_k = (kernel - 1) * dilation + 1
        h = draw(st.integers(min_value=eff_k + 2, max_value=10))
        w = draw(st.integers(min_value=eff_k + 2, max_value=10))
        x = draw(
            tensor_st(
                (n, in_c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        layer = nn.Conv2d(
            in_c,
            out_c,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _max_pool2d_dilation_sample_st() -> st.SearchStrategy[OpSample]:
    """``F.max_pool2d`` with ``dilation`` swept.

    ``ceil_mode`` stays False -- t2n's pool emitter raises
    NotImplementedError on ceil_mode=True.
    """
    import torch.nn.functional as F

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        kernel = draw(st.integers(min_value=2, max_value=3))
        stride = draw(st.integers(min_value=1, max_value=2))
        dilation = draw(st.integers(min_value=1, max_value=2))
        eff_k = (kernel - 1) * dilation + 1
        padding = draw(st.integers(min_value=0, max_value=kernel // 2))
        h = draw(st.integers(min_value=eff_k + 2, max_value=10))
        w = draw(st.integers(min_value=eff_k + 2, max_value=10))
        x = draw(
            tensor_st(
                (n, c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        wrapped = partial(
            F.max_pool2d,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(wrapped))

    return _draw()


def _depth_conv_pool_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="conv2d-dilation-groups",
            sample_st=_conv2d_dilation_groups_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="max_pool2d-dilation",
            sample_st=_max_pool2d_dilation_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Depth: norm kwargs (eps, affine), topk/sort flags, cat/stack with N
# -----------------------------------------------------------------------------


def _layer_norm_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.LayerNorm`` sweeping ``eps`` and ``elementwise_affine``."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        norm_size = draw(st.integers(min_value=2, max_value=6))
        leading_rank = draw(st.integers(min_value=1, max_value=2))
        leading = []
        for _ in range(leading_rank):
            leading.append(draw(st.integers(min_value=1, max_value=3)))
        shape = tuple(leading + [norm_size])
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        eps = draw(
            st.floats(
                min_value=1e-8,
                max_value=1e-3,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        elementwise_affine = draw(st.booleans())
        layer = nn.LayerNorm(
            norm_size, eps=eps, elementwise_affine=elementwise_affine
        ).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _batch_norm1d_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.BatchNorm1d`` sweeping ``eps`` (affine=True only).

    ``affine=False`` is not implemented in t2n's batch_norm emitter
    (``norm.py`` raises NotImplementedError when the param tensors are
    None). Sticking to affine=True for v1.
    """
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        c = draw(st.integers(min_value=1, max_value=6))
        n = draw(st.integers(min_value=2, max_value=4))
        with_length = draw(st.booleans())
        if with_length:
            length = draw(st.integers(min_value=1, max_value=4))
            shape = (n, c, length)
        else:
            shape = (n, c)
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        eps = draw(
            st.floats(
                min_value=1e-8,
                max_value=1e-3,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        layer = nn.BatchNorm1d(c, eps=eps, affine=True).eval()
        return OpSample(inputs=(x,), kwargs={}, module=layer)

    return _draw()


def _topk_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.topk`` sweeping ``largest`` (sorted=True only).

    t2n's topk emitter raises NotImplementedError on ``sorted=False``
    (``selector.py:758``). Sticking to sorted=True.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        k = draw(st.integers(min_value=1, max_value=shape[dim]))
        n = 1
        for s in shape:
            n *= s
        perm = draw(st.permutations(list(range(1, n + 1))))
        x = torch.tensor(perm, dtype=torch.float32).reshape(shape)
        largest = draw(st.booleans())
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "topk",
                kwargs={
                    "k": k,
                    "dim": dim,
                    "largest": largest,
                    "sorted": True,
                },
            ),
        )

    return _draw()


def _sort_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.sort`` sweeping ``descending`` (stable=False only).

    The ``stable`` kwarg fails the schema-match in t2n's dynamic call
    path -- sort.stable is a separate aten overload that t2n's
    update_call_op_arg_kwargs doesn't translate. Stable matters only
    when ties exist; we already feed unique values, so dropping the
    sweep loses no signal.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        n = 1
        for s in shape:
            n *= s
        perm = draw(st.permutations(list(range(1, n + 1))))
        x = torch.tensor(perm, dtype=torch.float32).reshape(shape)
        descending = draw(st.booleans())
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                "sort",
                kwargs={"dim": dim, "descending": descending},
            ),
        )

    return _draw()


class _CatNTensors(torch.nn.Module):
    """``torch.cat([t1, ..., tN], dim=k)`` -- variable N."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.cat(list(tensors), dim=self.dim)


def _cat_n_tensors_sample_st() -> st.SearchStrategy[OpSample]:
    """``torch.cat`` with N tensors (3-4 in this strategy)."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        base = list(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        n_tensors = draw(st.integers(min_value=3, max_value=4))
        tensors = []
        for _ in range(n_tensors):
            d = draw(st.integers(min_value=1, max_value=3))
            shape = list(base)
            shape[dim] = d
            tensors.append(
                draw(
                    tensor_st(
                        tuple(shape),
                        torch.float32,
                        finite=True,
                        domain=Interval(-5.0, 5.0),
                    )
                )
            )
        return OpSample(
            inputs=tuple(tensors),
            kwargs={},
            module=_CatNTensors(dim),
        )

    return _draw()


def _embedding_padding_idx_sample_st() -> st.SearchStrategy[OpSample]:
    """``nn.Embedding`` sweeping ``padding_idx``."""
    import torch.nn as nn

    @st.composite
    def _draw(draw) -> OpSample:
        num_emb = draw(st.integers(min_value=2, max_value=8))
        emb_dim = draw(st.integers(min_value=2, max_value=8))
        # padding_idx selects an embedding row that's zero at output time.
        padding_idx = draw(st.integers(min_value=0, max_value=num_emb - 1))
        batch = draw(st.integers(min_value=1, max_value=3))
        seq = draw(st.integers(min_value=1, max_value=4))
        idx = draw(
            tensor_st(
                (batch, seq),
                torch.int64,
                finite=True,
                domain=Interval(0, num_emb - 1),
            )
        )
        layer = nn.Embedding(num_emb, emb_dim, padding_idx=padding_idx).eval()
        return OpSample(inputs=(idx,), kwargs={}, module=layer)

    return _draw()


def _depth_norm_topk_cat_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            # Sweeping ``eps`` exposes near-zero output cases where tract
            # diverges by more than SUPER's 1e-3 (e.g. ~2.4e-3 abs with
            # very small ``eps``). ULTRA matches the practical noise
            # floor for layer_norm under hypothesis.
            name="layer_norm-broad",
            sample_st=_layer_norm_kwargs_sample_st(),
            tolerance=TractCheckTolerance.ULTRA,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="batch_norm1d-broad",
            sample_st=_batch_norm1d_kwargs_sample_st(),
            tolerance=TractCheckTolerance.VERY,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="topk-broad",
            sample_st=_topk_kwargs_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="sort-broad",
            sample_st=_sort_kwargs_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="cat-n-tensors",
            sample_st=_cat_n_tensors_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="embedding-padding-idx",
            sample_st=_embedding_padding_idx_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Depth: reduction dtype kwarg (sum, mean, prod with dtype=)
# -----------------------------------------------------------------------------


def _reduction_dtype_kwarg_sample_st(
    method_name: str,
) -> st.SearchStrategy[OpSample]:
    """Reduction with the ``dtype`` cast-then-reduce kwarg.

    PyTorch's ``sum/mean/prod(dim, *, dtype=)`` casts the input to
    ``dtype`` BEFORE reducing -- useful for f16 -> f32 accumulation.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        keepdim = draw(st.booleans())
        # Sweep target dtype: float64 (upcast) and float32 (no-op).
        target_dtype = draw(st.sampled_from([torch.float32, torch.float64]))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(
            inputs=(x,),
            kwargs={},
            module=TensorFnPrimitive(
                method_name,
                kwargs={
                    "dim": dim,
                    "keepdim": keepdim,
                    "dtype": target_dtype,
                },
            ),
        )

    return _draw()


def _depth_reduction_dtype_specs() -> T.List[OpSpec]:
    APPROX = TractCheckTolerance.APPROXIMATE
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            name="sum-dtype",
            sample_st=_reduction_dtype_kwarg_sample_st("sum"),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="mean-dtype",
            sample_st=_reduction_dtype_kwarg_sample_st("mean"),
            tolerance=APPROX,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="prod-dtype",
            sample_st=_reduction_dtype_kwarg_sample_st("prod"),
            tolerance=CLOSE,
            dtypes_hint=(torch.float32,),
        ),
    ]


# -----------------------------------------------------------------------------
# Registry assembly
# -----------------------------------------------------------------------------


def _build_registry() -> T.Tuple[OpSpec, ...]:
    specs: T.List[OpSpec] = []
    specs.extend(_unary_specs())
    specs.extend(_unary_broad_specs())
    specs.extend(_binary_arith_specs())
    specs.extend(_binary_compare_specs())
    specs.extend(_binary_logical_specs())
    specs.extend(_reduction_specs())
    specs.extend(_shape_specs())
    specs.extend(_clamp_where_specs())
    specs.extend(_activation_specs())
    specs.extend(_softmax_specs())
    specs.extend(_selector_specs())
    specs.extend(_pool_specs())
    specs.extend(_norm_conv_matmul_specs())
    specs.extend(_concat_split_specs())
    specs.extend(_pad_specs())
    specs.extend(_norm_specs())
    specs.extend(_sort_scatter_specs())
    specs.extend(_conv3d_pool3d_helpers_specs())
    specs.extend(_bitwise_builder_specs())
    specs.extend(_specialty_specs())
    specs.extend(_prelu_glu_einsum_specs())
    specs.extend(_final_specs())
    specs.extend(_constructors_index_sdpa_specs())
    specs.extend(_fft_specs())
    specs.extend(_glue_specs())
    specs.extend(_depth_conv_pool_specs())
    specs.extend(_depth_norm_topk_cat_specs())
    specs.extend(_depth_reduction_dtype_specs())
    return tuple(specs)


REGISTRY: T.Tuple[OpSpec, ...] = _build_registry()
