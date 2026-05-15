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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

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
    drawn dtype: broadcasting is independent.
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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

    return _draw()


def _binary_pow_int_exp_sample_st() -> st.SearchStrategy[OpSample]:
    """Pow with integer-valued exponent tensors.

    Integer exponents go through a different code path in tract (a
    repeated-multiply or sqr/rsqr fragment for small constants: see
    `torch_to_nnef/op/aten/math.py:_pow`). Cover small absolute values
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
        # Rounding ops: exact integer outputs, no tolerance needed.
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
        # Tier-A2 trivial unaries.
        (
            "positive",
            torch.positive,
            TractCheckTolerance.EXACT,
            _UNARY_FINITE_DOMAIN,
        ),
        (
            "deg2rad",
            torch.deg2rad,
            TractCheckTolerance.VERY,
            _UNARY_FINITE_DOMAIN,
        ),
        (
            "rad2deg",
            torch.rad2deg,
            TractCheckTolerance.VERY,
            _UNARY_FINITE_DOMAIN,
        ),
    ]
    return [
        OpSpec(
            name=name,
            sample_st=_unary_sample_st(op, domain=domain),
            tolerance=tol,
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
        return OpSample(inputs=(x,), module=UnaryPrimitive(op))

    return _draw()


def _unary_broad_specs() -> T.List[OpSpec]:
    """Multi-dtype broadening for the highest-value unary ops.

    f16 has a tighter representable range; we shrink the per-op domain
    accordingly. We don't broaden every unary op: the goal is to surface
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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

    return _draw()


# Broadened specs derived from PyTorch op signatures.
# `add`/`sub` accept `alpha` (multiplier for `other`) per
# https://pytorch.org/docs/stable/generated/torch.add.html and the t2n
# emitter at `torch_to_nnef/op/aten/math.py` exports it. We sweep
# alpha values plus multi-dtype (f32 + f16). Domain bounds for f16 are
# tighter to keep results within f16's representable range.
_F16_BINARY_DOMAIN = Interval(-50.0, 50.0)


def _add_or_sub_multi_dtype_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Sweep dtype (f32 + f16) for `torch.add` / `torch.sub`.

    Note: `alpha` (the second documented parameter of these ops) is NOT
    swept here: see `_add_or_sub_alpha_sample_st` and the corresponding
    `add-alpha-xfail` / `sub-alpha-xfail` registry entries for that
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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

    return _draw()


def _add_or_sub_alpha_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Sweep non-default `alpha` for `torch.add` / `torch.sub`.

    PyTorch's `torch.add(a, b, alpha=k)` computes `a + k*b` and
    `torch.sub(a, b, alpha=k)` computes `a - k*b`. Originally proptest
    found that the alpha attribute was silently dropped at export: two
    bugs combined: `ir_helpers._prepare_arguments` truncated aten:add /
    aten:sub inputs to the first two, and `unary.generic_unary` (which
    these ops were routed through) ignores attributes. Both fixed in this
    same change set: dedicated emitters live at
    `torch_to_nnef/op/aten/math.py` (search `_add_or_sub_with_alpha`)
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
            module=BinaryPrimitive(partial(op, alpha=alpha)),
        )

    return _draw()


def _div_explicit_none_sample_st() -> st.SearchStrategy[OpSample]:
    """Div called with explicit `rounding_mode=None`.

    Originally proptest found that the t2n div emitter cast the output to
    int64 whenever `len(node.inputs) == 3`, even when `rounding_mode`
    was the literal `None` (which PyTorch documents as equivalent to
    `/` true division). Now fixed: the emitter checks
    `rounding_mode is not None` before applying the cast (see
    `torch_to_nnef/op/aten/math.py`).
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
            module=BinaryPrimitive(partial(torch.div, rounding_mode=None)),
        )

    return _draw()


def _div_rounding_sample_st() -> st.SearchStrategy[OpSample]:
    """Div with `rounding_mode in {"trunc", "floor"}`.

    **Tract upstream precision bug: this spec stays xfailed pending a
    tract fix.** The original t2n-side issues are fixed:

    1. `div(float, float, rounding_mode="trunc")` previously returned
       int64; now returns float32 to match PyTorch (the emitter only
       casts to int64 when the traced output dtype is integer).

    2. The remaining failure is a tract precision issue: tract's float
       division for some specific value pairs (e.g. `11.75 / 11.75`)
       returns ~0.99999994 instead of 1.0 (off by ~0.5 ULP of f32
       epsilon), so `trunc(0.99999994) = 0` rather than `trunc(1.0)
       = 1`. Reproduced directly with a plain `div` (no rounding).
       The trunc/floor NNEF fragments at `torch_to_nnef/op/fragment/`
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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(wrapped))

    return _draw()


def _binary_arith_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="add",
            sample_st=_binary_broadcast_sample_st(
                torch.add, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
        ),
        OpSpec(
            name="add-broad",
            sample_st=_add_or_sub_multi_dtype_sample_st(torch.add),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="add-alpha",
            sample_st=_add_or_sub_alpha_sample_st(torch.add),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="sub",
            sample_st=_binary_broadcast_sample_st(
                torch.sub, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
        ),
        OpSpec(
            name="sub-broad",
            sample_st=_add_or_sub_multi_dtype_sample_st(torch.sub),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="sub-alpha",
            sample_st=_add_or_sub_alpha_sample_st(torch.sub),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="mul",
            sample_st=_binary_broadcast_sample_st(
                torch.mul, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
        ),
        OpSpec(
            name="div",
            sample_st=_div_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="div-explicit-none",
            sample_st=_div_explicit_none_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="div-rounding-xfail",
            sample_st=_div_rounding_sample_st(),
            tolerance=TractCheckTolerance.VERY,
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
        ),
        OpSpec(
            name="pow-int-exp",
            sample_st=_binary_pow_int_exp_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="mul-broad",
            sample_st=_binary_multi_dtype_sample_st(
                torch.mul,
                domain_f32=_BINARY_ARITH_DOMAIN,
                domain_f16=_F16_BINARY_DOMAIN,
            ),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="minimum",
            sample_st=_binary_broadcast_sample_st(
                torch.minimum, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="maximum",
            sample_st=_binary_broadcast_sample_st(
                torch.maximum, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            # Element-wise `torch.min(a, b)` (binary form).
            # Distinct from the dim-reduction in `min-dim`.
            name="min-elementwise",
            sample_st=_binary_broadcast_sample_st(
                torch.min, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="max-elementwise",
            sample_st=_binary_broadcast_sample_st(
                torch.max, domain=_BINARY_ARITH_DOMAIN
            ),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="floor_divide-xfail",
            sample_st=_binary_broadcast_sample_st(
                torch.floor_divide,
                domain=_BINARY_DIV_NUM_DOMAIN,
            ),
            tolerance=TractCheckTolerance.VERY,
            xfail_reason=(
                "Same tract upstream precision bug as div-rounding-xfail: "
                "tract's div(x, x) returns ~0.99999994 instead of 1.0, so "
                "floor(div(107, 107)) = 0 instead of 1. Fix is upstream in "
                "tract's f32 division algorithm."
            ),
        ),
        OpSpec(
            # remainder is implemented as `a - floor(a/b) * b` (see
            # `torch_to_nnef/op/fragment/remainder.nnef`). The fragment
            # is mathematically correct, but it depends on tract's f32
            # `div` which has the precision bug noted in
            # `div-rounding-xfail` (`div(x, x)` returns ~0.99999994).
            # That makes `floor(div(x, x)) = 0` instead of 1, and
            # `remainder(x, x) = x` instead of 0.
            name="remainder-xfail",
            sample_st=_div_like_sample_st(torch.remainder),
            tolerance=TractCheckTolerance.VERY,
            xfail_reason=(
                "Same tract upstream div precision bug propagates through "
                "the remainder fragment `a - floor(a/b) * b`: "
                "remainder(205.375, 205.375) returns 205.375 in tract vs "
                "0 in PyTorch."
            ),
        ),
        OpSpec(
            # fmod is implemented as `a - trunc(a/b) * b` (see
            # `torch_to_nnef/op/fragment/fmod.nnef`). Same upstream
            # tract div bug as remainder.
            name="fmod-xfail",
            sample_st=_div_like_sample_st(torch.fmod),
            tolerance=TractCheckTolerance.VERY,
            xfail_reason=(
                "Same tract upstream div precision bug propagates through "
                "the fmod fragment `a - trunc(a/b) * b`."
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
        )
        for name, op in cases
    ]


# Reduction specs (4)


def _clamp_sample_st() -> st.SearchStrategy[OpSample]:
    """Clamp over the full PyTorch surface.

    Historically narrowed for two distinct reasons that turned out to be
    the SAME `if X.data:` Python falsy bug in t2n's clamp emitter (now
    fixed in `torch_to_nnef/op/aten/activation.py`):

    - `min/max == 0.0` was silently skipped (truthy check on the bound
      value), letting tract output the unclamped input.
    - `min == max == 0.0` plus matching input tripped
      `KeyError: 'output_0'` because BOTH conditional branches were
      skipped, leaving the output node unwired.

    Both go away with the explicit `is None` check. This strategy now
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
            module=TernaryPrimitive(torch.where),
        )

    return _draw()


def _clamp_where_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="clamp",
            sample_st=_clamp_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="where",
            sample_st=_where_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


def _bitwise_binary_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Bitwise binary op over int32: mutually broadcastable shapes."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        a = draw(
            tensor_st(sa, torch.int32, finite=True, domain=Interval(-100, 100))
        )
        b = draw(
            tensor_st(sb, torch.int32, finite=True, domain=Interval(-100, 100))
        )
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

    return _draw()


def _bitwise_not_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.bitwise_not` over int32."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4))
        x = draw(
            tensor_st(
                shape, torch.int32, finite=True, domain=Interval(-100, 100)
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.bitwise_not))

    return _draw()


def _bitwise_shift_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Bitwise shift over int32: non-negative data, shift counts in [0, 30].

    The data interval excludes negatives because right-shift on signed
    integers is implementation-defined in C++ and tract follows the
    arithmetic-shift convention; sticking to non-negative inputs keeps
    the comparison crisp. Shift counts above 30 are UB on 32-bit ints,
    so we cap them at 30.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        a = draw(
            tensor_st(sa, torch.int32, finite=True, domain=Interval(0, 1024))
        )
        b = draw(
            tensor_st(sb, torch.int32, finite=True, domain=Interval(0, 30))
        )
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

    return _draw()


def _zeros_like_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.zeros_like(input)`: output matches input shape/dtype.

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
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.zeros_like))

    return _draw()


def _ones_like_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.ones_like(input)`: see _zeros_like for shape note."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.ones_like))

    return _draw()


def _full_like_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.full_like(input, fill_value)`: swept fill values."""

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
        ),
        OpSpec(
            name="bitwise_or",
            sample_st=_bitwise_binary_sample_st(torch.bitwise_or),
            tolerance=EXACT,
        ),
        OpSpec(
            name="bitwise_xor",
            sample_st=_bitwise_binary_sample_st(torch.bitwise_xor),
            tolerance=EXACT,
        ),
        OpSpec(
            name="bitwise_not",
            sample_st=_bitwise_not_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="bitwise_left_shift",
            sample_st=_bitwise_shift_sample_st(torch.bitwise_left_shift),
            tolerance=EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="bitwise_right_shift",
            sample_st=_bitwise_shift_sample_st(torch.bitwise_right_shift),
            tolerance=EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="zeros_like",
            sample_st=_zeros_like_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="ones_like",
            sample_st=_ones_like_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="full_like",
            sample_st=_full_like_sample_st(),
            tolerance=EXACT,
        ),
    ]


# Recently-shipped aten op handlers (PRs around early 2026):
# `exp2`, `sinc`, `frac`, `tanhshrink`, `erfc`, `signbit`, `logaddexp`,
# `logaddexp2`, `copysign`, `hypot`, `xlogy`, `fmax`, `fmin`, `ldexp`,
# `heaviside`, `isclose`, `addcdiv`. Grouped together so the elementwise
# module stays the home for primitive arithmetic / comparison ops.

# Domains chosen to keep f32 outputs representable. `exp2` saturates at
# 2^128 around 128, so we keep the input < 60.
_UNARY_EXP2_DOMAIN = Interval(-60.0, 60.0)
# `sinc(x) = sin(pi x)/(pi x)`: bound to avoid huge-pi-x cancellation,
# which would amplify tract / torch ULP drift past VERY tolerance.
_UNARY_SINC_DOMAIN = Interval(-10.0, 10.0)
# `frac` straddles the integer boundary; bound modestly so trunc has
# room without overflowing f32 mantissa precision.
_UNARY_FRAC_DOMAIN = Interval(-1e3, 1e3)
# Hypot, copysign etc. -- a single comfortable f32 finite range.
_BINARY_FINITE_DOMAIN = Interval(-1e3, 1e3)
# logaddexp(2): inputs already get exp'd internally, so bound similar
# to the unary exp surface.
_LOGADDEXP_DOMAIN = Interval(-30.0, 30.0)
# ldexp: 2nd arg is the integer exponent. PyTorch casts it to float
# under the hood. Keep magnitudes modest to avoid 2**exp blowing past
# f32 range.
_LDEXP_EXP_DOMAIN = Interval(-30, 30)
# xlogy: y must be >= 0 (log domain). PyTorch returns 0 when x==0
# regardless of y (even y=0). We sweep x including zero, y in (0, inf).
_XLOGY_X_DOMAIN = Interval(-1e3, 1e3)
_XLOGY_Y_DOMAIN = Interval(1e-3, 1e3)


def _xlogy_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.xlogy(x, y)` with `y > 0` and explicit `x == 0` rows.

    The special branch `xlogy(0, y) -> 0` (including `y == 0`) is one
    of the two reasons xlogy exists separately from `x * log(y)`; we
    inject a small number of guaranteed-zero rows in `x` so each draw
    has a non-negligible chance of exercising it.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        x = draw(
            tensor_st(sa, torch.float32, finite=True, domain=_XLOGY_X_DOMAIN)
        )
        y = draw(
            tensor_st(sb, torch.float32, finite=True, domain=_XLOGY_Y_DOMAIN)
        )
        # Force ~25% of entries in x to exactly 0 so the special branch
        # gets stable coverage even on small draws.
        if x.numel() > 0:
            mask = draw(
                tensor_st(
                    tuple(x.shape),
                    torch.float32,
                    finite=True,
                    domain=Interval(0.0, 1.0),
                )
            )
            x = torch.where(mask < 0.25, torch.zeros_like(x), x)
        return OpSample(inputs=(x, y), module=BinaryPrimitive(torch.xlogy))

    return _draw()


def _binary_nan_friendly_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Binary op whose semantics depend on NaN inputs: fmax / fmin / isclose.

    Drawn with `finite=False` so NaN/Inf show up; magnitude bound to
    keep non-special values in a representable f32 range.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        # finite=False lets NaN / Inf in. We deliberately don't pass a
        # domain since hypothesis would otherwise filter them out.
        a = draw(tensor_st(sa, torch.float32, finite=False))
        b = draw(tensor_st(sb, torch.float32, finite=False))
        # Clamp magnitudes of finite entries so non-special arithmetic
        # stays comparable: keep `a, b` in [-1e3, 1e3] when finite.
        a = torch.where(torch.isfinite(a), torch.clamp(a, -1e3, 1e3), a)
        b = torch.where(torch.isfinite(b), torch.clamp(b, -1e3, 1e3), b)
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op))

    return _draw()


def _ldexp_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.ldexp(x, exp)` with integer-valued exponent (as float)."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb = draw(binary_broadcast_shapes_st(max_rank=4, max_dim=6))
        x = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=Interval(-1e3, 1e3)
            )
        )
        # PyTorch's ldexp accepts a float `exp` but truncates to int
        # semantics. Draw integer-valued floats in a safe range.
        exp_int = draw(
            tensor_st(
                sb,
                torch.float32,
                finite=True,
                domain=_LDEXP_EXP_DOMAIN,
            )
        ).round()
        return OpSample(
            inputs=(x, exp_int), module=BinaryPrimitive(torch.ldexp)
        )

    return _draw()


def _addcdiv_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.addcdiv(self, t1, t2, value)`: 3 broadcast inputs + scalar.

    `t2` (divisor) excludes near-zero so the division stays numerically
    stable on both sides; the `value` scalar sweeps a small range
    including the default 1.0 case.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb, sc = draw(ternary_broadcast_shapes_st(max_rank=3, max_dim=5))
        inp = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=Interval(-1e2, 1e2)
            )
        )
        t1 = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=Interval(-1e2, 1e2)
            )
        )
        t2 = draw(
            tensor_st(
                sc,
                torch.float32,
                finite=True,
                domain=_BINARY_DIV_DEN_DOMAIN,
            )
        )
        value = draw(
            st.floats(
                min_value=-2.0,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(inp, t1, t2),
            module=TernaryPrimitive(partial(torch.addcdiv, value=value)),
        )

    return _draw()


def _recent_elementwise_specs() -> T.List[OpSpec]:
    """Specs for the elementwise ops shipped in the recent PR cluster."""
    APPROX = TractCheckTolerance.APPROXIMATE
    VERY = TractCheckTolerance.VERY
    EXACT = TractCheckTolerance.EXACT
    return [
        # --- Unary real -> real ---
        OpSpec(
            name="exp2",
            sample_st=_unary_sample_st(torch.exp2, domain=_UNARY_EXP2_DOMAIN),
            tolerance=VERY,
        ),
        OpSpec(
            name="sinc",
            sample_st=_unary_sample_st(torch.sinc, domain=_UNARY_SINC_DOMAIN),
            tolerance=VERY,
        ),
        OpSpec(
            name="frac",
            sample_st=_unary_sample_st(torch.frac, domain=_UNARY_FRAC_DOMAIN),
            # frac decomposes as `x - trunc(x)` which inherits tract's
            # div / trunc precision quirks near integer boundaries.
            tolerance=VERY,
        ),
        OpSpec(
            name="tanhshrink",
            sample_st=_unary_sample_st(
                torch.nn.functional.tanhshrink, domain=_UNARY_TANH_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="erfc",
            sample_st=_unary_sample_st(torch.erfc, domain=Interval(-5.0, 5.0)),
            # erfc lowers via 1 - erf; both sides accumulate ULP error.
            tolerance=VERY,
        ),
        # --- Unary real -> bool ---
        OpSpec(
            # Bool comparator is exact; the divergence with torch is
            # `signbit(-0.0)` which the NNEF `x < 0` lowering can't see
            # (documented in `torch_to_nnef/op/aten/math.py:signbit`).
            # Hypothesis hits -0.0 reliably under a finite-zero-spanning
            # domain (NumPy's signed-zero is emitted as part of the
            # f32 float pool), so the spec stays xfail until the
            # generator filters them out or the fragment learns the
            # IEEE-754 sign bit.
            name="signbit-xfail",
            sample_st=_unary_sample_st(
                torch.signbit, domain=_UNARY_FINITE_DOMAIN
            ),
            tolerance=EXACT,
            xfail_reason=(
                "NNEF `x < 0` can't see the IEEE-754 sign bit so "
                "`signbit(-0.0)` returns False vs torch's True. "
                "Falsifying example: tensor([-0.])."
            ),
        ),
        # --- Binary real -> real ---
        OpSpec(
            name="logaddexp",
            sample_st=_binary_broadcast_sample_st(
                torch.logaddexp, domain=_LOGADDEXP_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="logaddexp2",
            sample_st=_binary_broadcast_sample_st(
                torch.logaddexp2, domain=_LOGADDEXP_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="copysign",
            sample_st=_binary_broadcast_sample_st(
                torch.copysign, domain=_BINARY_FINITE_DOMAIN
            ),
            # Same -0.0 caveat as signbit; our generator avoids -0.0.
            tolerance=EXACT,
        ),
        OpSpec(
            name="hypot",
            sample_st=_binary_broadcast_sample_st(
                torch.hypot, domain=_BINARY_FINITE_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="xlogy",
            sample_st=_xlogy_sample_st(),
            tolerance=VERY,
        ),
        OpSpec(
            # fmax / fmin propagate non-NaN over NaN; the tract `fmax` /
            # `fmin` fragments implement this via `where(isnan(b), a, ...)`.
            name="fmax",
            sample_st=_binary_nan_friendly_sample_st(torch.fmax),
            tolerance=APPROX,
        ),
        OpSpec(
            name="fmin",
            sample_st=_binary_nan_friendly_sample_st(torch.fmin),
            tolerance=APPROX,
        ),
        OpSpec(
            name="ldexp",
            sample_st=_ldexp_sample_st(),
            tolerance=VERY,
        ),
        OpSpec(
            name="heaviside",
            sample_st=_binary_broadcast_sample_st(
                torch.heaviside, domain=_BINARY_FINITE_DOMAIN
            ),
            tolerance=EXACT,
        ),
        # --- Binary real -> bool ---
        OpSpec(
            # NaN coverage matters: `isclose(NaN, NaN, equal_nan=False)`
            # returns False; we don't pass equal_nan so the default path
            # is exercised.
            name="isclose-xfail",
            sample_st=_binary_nan_friendly_sample_st(torch.isclose),
            tolerance=EXACT,
            xfail_reason=(
                "isclose `|a - b| <= atol + rtol*|b|` evaluates True for "
                "infinity inputs in the NNEF lowering: with a=0, b=inf "
                "the math becomes inf <= inf, which is True in NNEF "
                "stdlib but False in torch (torch checks "
                "`a == b or (finite and within tol)`). The fragment at "
                "`torch_to_nnef/op/fragment/isclose.nnef` would need a "
                "finite-input guard. See `_binary_nan_friendly_sample_st` "
                "for the falsifying example: isclose(0, inf)."
            ),
        ),
        # --- Ternary ---
        OpSpec(
            name="addcdiv",
            sample_st=_addcdiv_sample_st(),
            tolerance=VERY,
        ),
    ]


# Special functions (Bessel I0, lgamma)


# `i0` / `i0e`: tested across both polynomial branches (|x| < 3.75
# small-series and the |x| >= 3.75 asymptotic). VERY tolerance --
# polynomial approximations are bounded to ~1e-7 relative error but
# the exp / sqrt chain in the large branch amplifies that.
_UNARY_I0_DOMAIN = Interval(-15.0, 15.0)

# `lgamma`: only valid for `x > 0.5` in our Lanczos-only impl. Bound
# above 0.6 to stay clear of the reflection branch.
_UNARY_LGAMMA_DOMAIN = Interval(0.6, 50.0)


def _special_function_specs() -> T.List[OpSpec]:
    VERY = TractCheckTolerance.VERY
    return [
        OpSpec(
            name="i0",
            sample_st=_unary_sample_st(
                torch.special.i0, domain=_UNARY_I0_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="special_i0e",
            sample_st=_unary_sample_st(
                torch.special.i0e, domain=_UNARY_I0_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="lgamma",
            sample_st=_unary_sample_st(
                torch.lgamma, domain=_UNARY_LGAMMA_DOMAIN
            ),
            tolerance=VERY,
        ),
        # `frexp`: returns (mantissa in [0.5, 1.0), exponent: int32).
        # Sample range stays in the normal-float regime (well clear of
        # FLT_MIN ~= 1.18e-38); subnormals lose an exponent bit through
        # tract's `log2` and are documented as out-of-domain in the
        # fragment.
        OpSpec(
            name="frexp",
            sample_st=_unary_sample_st(
                torch.frexp, domain=Interval(-1e30, 1e30)
            ),
            tolerance=VERY,
        ),
        # `special_i1` / `special_i1e`: same polynomial branches as
        # `i0` / `i0e` (A&S 9.8.3 / 9.8.4).
        OpSpec(
            name="special_i1",
            sample_st=_unary_sample_st(
                torch.special.i1, domain=_UNARY_I0_DOMAIN
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="special_i1e",
            sample_st=_unary_sample_st(
                torch.special.i1e, domain=_UNARY_I0_DOMAIN
            ),
            tolerance=VERY,
        ),
        # `digamma`: asymptotic series after a fixed 6-shift; valid for
        # `x > 0`. Sample above 0.1 to keep the leading `1/x` finite.
        OpSpec(
            name="digamma",
            sample_st=_unary_sample_st(
                torch.digamma, domain=Interval(0.1, 50.0)
            ),
            tolerance=VERY,
        ),
        # `special_entr(x) = -x * log(x)`. Sample non-negative so torch's
        # reference stays finite (negative inputs return -inf which our
        # eps-clamped fragment doesn't try to match exactly).
        OpSpec(
            name="special_entr",
            sample_st=_unary_sample_st(
                torch.special.entr, domain=Interval(0.0, 100.0)
            ),
            tolerance=VERY,
            dynamic_axes_compatible=True,
        ),
    ]


def _xlog1py_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.special.xlog1py(x, y)` -- domain `y > -1`.

    Shapes are kept equal-rank so the broadcast goes through tract's
    pointwise binary path (rank-mismatch broadcast hits an unrelated
    `pow` limitation in t2n).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        # `y > -1` to keep `log(1 + y)` finite in torch's reference.
        y = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-0.99, 100.0),
            )
        )
        return OpSample(
            inputs=(x, y),
            module=BinaryPrimitive(torch.special.xlog1py),
        )

    return _draw()


def _ravel_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e3, 1e3),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.ravel))

    return _draw()


def _diff_sample_st(n: int) -> st.SearchStrategy[OpSample]:
    """`torch.diff(x, n=n, dim=-1)` with input axis size > n."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        # Trailing axis must be larger than `n` so the result is non-empty.
        sizes = draw(
            st.lists(
                st.integers(min_value=1, max_value=4),
                min_size=rank,
                max_size=rank,
            )
        )
        sizes[-1] = draw(st.integers(min_value=n + 1, max_value=n + 5))
        x = draw(
            tensor_st(
                tuple(sizes),
                torch.float32,
                finite=True,
                domain=Interval(-100.0, 100.0),
            )
        )
        op_fn = (lambda nn: lambda t: torch.diff(t, n=nn, dim=-1))(n)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _float_power_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.float_power(x, y)` with positive base + small exponent.

    Equal-rank shapes only: rank-mismatch broadcast trips t2n's `pow`
    handler (unrelated to float_power).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4))
        base = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(0.1, 100.0),
            )
        )
        exp_t = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        return OpSample(
            inputs=(base, exp_t),
            module=BinaryPrimitive(torch.float_power),
        )

    return _draw()


def _trapezoid_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.trapezoid(y, dx=, dim=-1)` with static dx."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        sizes = draw(
            st.lists(
                st.integers(min_value=2, max_value=5),
                min_size=rank,
                max_size=rank,
            )
        )
        dx = draw(st.sampled_from([1.0, 0.5, 2.0]))
        y = draw(
            tensor_st(
                tuple(sizes),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        op_fn = (lambda d: lambda t: torch.trapezoid(t, dx=d, dim=-1))(dx)
        return OpSample(inputs=(y,), module=UnaryPrimitive(op_fn))

    return _draw()


def _tier_a2_specs() -> T.List[OpSpec]:
    APPROX = TractCheckTolerance.APPROXIMATE
    VERY = TractCheckTolerance.VERY
    # All of these emit pure pointwise / slice-shape primitives whose
    # axis arguments stay independent of the dynamic axis 0, so the
    # dyn-axes proptest passes.
    return [
        OpSpec(
            name="ravel",
            sample_st=_ravel_sample_st(),
            tolerance=APPROX,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="float_power",
            sample_st=_float_power_sample_st(),
            tolerance=VERY,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="diff_n1",
            sample_st=_diff_sample_st(1),
            tolerance=VERY,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="diff_n2",
            sample_st=_diff_sample_st(2),
            tolerance=VERY,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="trapezoid",
            sample_st=_trapezoid_sample_st(),
            tolerance=VERY,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="special_xlog1py",
            sample_st=_xlog1py_sample_st(),
            tolerance=VERY,
            dynamic_axes_compatible=True,
        ),
    ]


# Specialty ops (embedding, repeat_interleave, upsample, sdpa, ...)

SPECS = (
    *_unary_specs(),
    *_unary_broad_specs(),
    *_binary_arith_specs(),
    *_binary_compare_specs(),
    *_binary_logical_specs(),
    *_clamp_where_specs(),
    *_bitwise_builder_specs(),
    *_recent_elementwise_specs(),
    *_special_function_specs(),
    *_tier_a2_specs(),
)
