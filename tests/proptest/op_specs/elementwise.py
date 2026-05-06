"""Spec builders for the elementwise op group."""

import typing as T
from functools import partial

import torch
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    BinaryPrimitive,
    TernaryPrimitive,
    UnaryPrimitive,
)
from ..inputs import Interval, dtype_st, tensor_st
from ..shapes import (
    binary_broadcast_shapes_st,
    shape_st,
    ternary_broadcast_shapes_st,
)
from ._common import OpSample, OpSpec, _unary_sample_st

# Domain bounds chosen to keep outputs in a numerically meaningful range and
# avoid trivial saturation while still exercising edge cases.
_UNARY_TRIG_DOMAIN = Interval(-6.283, 6.283)  # ~ +/- 2*pi
_UNARY_TAN_DOMAIN = Interval(-1.4, 1.4)  # avoid tan(pi/2) explosion
_UNARY_EXP_DOMAIN = Interval(-30.0, 30.0)
_UNARY_LOG_DOMAIN = Interval(1e-3, 1e4)
_UNARY_FINITE_DOMAIN = Interval(-1e4, 1e4)
_UNARY_SQRT_DOMAIN = Interval(0.0, 1e4)
_UNARY_RSQRT_DOMAIN = Interval(1e-3, 1e4)
# Reciprocal is positive-only here: the strategy doesn't yet support
# disjoint intervals, and a single straddle interval would generate
# values arbitrarily close to zero.
_UNARY_RECIP_DOMAIN = Interval(1e-2, 1e3)
_UNARY_TANH_DOMAIN = Interval(-30.0, 30.0)
# Inverse-trig (asin, acos): input in [-1, 1].
_UNARY_INVTRIG_DOMAIN = Interval(-1.0, 1.0)
# acosh: input in [1, inf).
_UNARY_ACOSH_DOMAIN = Interval(1.0, 1e3)
# atanh: input in (-1, 1) strict; epsilon margin avoids the singularities.
_UNARY_ATANH_DOMAIN = Interval(-0.999, 0.999)
# Hyperbolic sinh/cosh: bounded to avoid overflow at f32.
_UNARY_HYP_DOMAIN = Interval(-30.0, 30.0)


def _binary_broadcast_sample_st(
    op: T.Callable[..., torch.Tensor],
    dtype: torch.dtype = torch.float32,
    domain: T.Optional[Interval] = None,
    finite: bool = True,
) -> st.SearchStrategy[OpSample]:
    """Binary-op sample strategy with mutually broadcastable shapes."""

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
    """Binary-op sample sweeping a list of float dtypes.

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


def _unary_specs() -> T.List[OpSpec]:
    # Transcendentals get VERY tolerance (rtol/atol 1e-4) since tract's f32
    # implementation typically diverges from torch by 1-2 ULPs (~1e-6 relative)
    # and our CLOSE level (1e-5) trips on edge cases (e.g. sin near pi).
    cases: T.List[T.Tuple[str, T.Callable, TractCheckTolerance, Interval]] = [
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


# Binary specs (5 arithmetic + 6 compare + 3 logical = 14)

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


# Broadened specs derived from PyTorch op signatures.
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


# Reduction specs (4)


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


# Specialty ops (embedding, repeat_interleave, upsample, sdpa, ...)
