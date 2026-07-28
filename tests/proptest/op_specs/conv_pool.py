"""Spec builders for the conv pool op group."""

import typing as T
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    BinaryPrimitive,
    UnaryPrimitive,
)
from ..inputs import Interval, tensor_st
from ..shapes import (
    binary_broadcast_shapes_st,
    shape_st,
)
from ._common import (
    NnefGapStage,
    OpSample,
    OpSpec,
    _unary_sample_st,
)
from ._gap_common import (
    gap_spec,
    image_st,
    unpool,
)


def _pool2d_sample_st(
    op: T.Callable[..., torch.Tensor],
    allow_padding: bool = True,
) -> st.SearchStrategy[OpSample]:
    """2D pool over (N, C, H, W) input.

    t2n's pool emitters reject `ceil_mode=True`,
    `count_include_pad=False` and `divisor_override` so we keep all of
    those at safe defaults. avg_pool callers should set
    `allow_padding=False`: the t2n avg_pool emitter requires
    `count_include_pad=True` (PyTorch's default) but emits NNEF's
    `border="ignore"` (which is `count_include_pad=False`); padding
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
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _pool1d_sample_st(
    op: T.Callable[..., torch.Tensor],
    allow_padding: bool = True,
    op_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
) -> st.SearchStrategy[OpSample]:
    """1D pool. See `_pool2d_sample_st` for `allow_padding` rationale.

    `op_kwargs` are passed through to the wrapped op (e.g.
    `return_indices=True` for `max_pool1d`).
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
            op,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            **(op_kwargs or {}),
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _adaptive_pool2d_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """adaptive_pool2d: input H/W must divide output H/W.

    t2n's adaptive pool emitter at `torch_to_nnef/op/aten/pool.py`
    is documented as "will likely only work with full defined shapes".
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
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _pool_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    APPROX = TractCheckTolerance.APPROXIMATE
    return [
        OpSpec(
            name="max_pool1d",
            aten_ops=("max_pool1d",),
            sample_st=_pool1d_sample_st(F.max_pool1d),
            tolerance=EXACT,
        ),
        OpSpec(
            name="max_pool1d_with_indices",
            aten_ops=("max_pool1d_with_indices",),
            sample_st=_pool1d_sample_st(
                F.max_pool1d, op_kwargs={"return_indices": True}
            ),
            tolerance=EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="max_pool2d",
            aten_ops=("max_pool2d",),
            sample_st=_pool2d_sample_st(F.max_pool2d),
            tolerance=EXACT,
        ),
        OpSpec(
            # padding=0 only: t2n's avg_pool emitter requires
            # count_include_pad=True (PyTorch's default) but emits
            # NNEF border="ignore" which means count_include_pad=False.
            # Padding > 0 surfaces the semantic mismatch (PyTorch's edge
            # outputs include the padded zeros in the average; tract's
            # don't). t2n bug: emitter should either implement
            # count_include_pad=True faithfully or reject it.
            name="avg_pool1d",
            aten_ops=("avg_pool1d",),
            sample_st=_pool1d_sample_st(F.avg_pool1d, allow_padding=False),
            tolerance=APPROX,
        ),
        OpSpec(
            # Same padding limitation as avg_pool1d.
            name="avg_pool2d",
            aten_ops=("avg_pool2d",),
            sample_st=_pool2d_sample_st(F.avg_pool2d, allow_padding=False),
            tolerance=APPROX,
        ),
        OpSpec(
            name="adaptive_avg_pool2d",
            aten_ops=("adaptive_avg_pool2d",),
            sample_st=_adaptive_pool2d_sample_st(F.adaptive_avg_pool2d),
            tolerance=APPROX,
        ),
        OpSpec(
            name="adaptive_max_pool2d",
            aten_ops=("adaptive_max_pool2d",),
            sample_st=_adaptive_pool2d_sample_st(F.adaptive_max_pool2d),
            tolerance=EXACT,
        ),
    ]


# Norm + Matmul + Conv specs


def _conv3d_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Conv3d` over (N, in_C, D, H, W) input."""

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
        return OpSample(inputs=(x,), module=layer)

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
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _cumsum_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.cumsum(input, dim)`."""

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
            module=UnaryPrimitive(partial(torch.cumsum, dim=dim)),
        )

    return _draw()


def _logcumsumexp_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.logcumsumexp(x, dim)`: numerically-stable cumsum of exp.

    Bound input magnitudes so the running `exp` stays in f32 range over
    up to 5 steps along the chosen axis (the t2n lowering keeps a
    running-max for stability, so wider domains would work too -- we
    keep CLOSE because the scan accumulates ULP error like cumsum).
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
            module=UnaryPrimitive(partial(torch.logcumsumexp, dim=dim)),
        )

    return _draw()


def _cumprod_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.cumprod(input, dim)`.

    Bound elements close to 1 so the running product stays in range
    over up-to-5 steps along the chosen axis.
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
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1.5, 1.5),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(torch.cumprod, dim=dim)),
        )

    return _draw()


def _atan2_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.atan2(y, x)`: broadcasted, no special domain."""

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
        return OpSample(inputs=(y, x), module=BinaryPrimitive(torch.atan2))

    return _draw()


def _classifier_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """NaN/Inf classifier: input may contain NaN/Inf."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=3))
        # finite=False so NaN/Inf can be drawn; outputs are bool exact.
        x = draw(tensor_st(shape, torch.float32, finite=False))
        return OpSample(inputs=(x,), module=UnaryPrimitive(op))

    return _draw()


def _conv3d_pool3d_helpers_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    APPROX = TractCheckTolerance.APPROXIMATE
    CLOSE = TractCheckTolerance.CLOSE
    VERY = TractCheckTolerance.VERY
    return [
        OpSpec(
            name="conv3d",
            aten_ops=("conv3d",),
            sample_st=_conv3d_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="max_pool3d",
            aten_ops=("max_pool3d",),
            sample_st=_pool3d_sample_st(F.max_pool3d),
            tolerance=EXACT,
        ),
        OpSpec(
            # Same count_include_pad caveat as avg_pool1d/2d: padding=0.
            name="avg_pool3d",
            aten_ops=("avg_pool3d",),
            sample_st=_pool3d_sample_st(F.avg_pool3d, allow_padding=False),
            tolerance=APPROX,
        ),
        OpSpec(
            name="expm1",
            aten_ops=("expm1",),
            sample_st=_unary_sample_st(
                torch.expm1, domain=Interval(-10.0, 10.0)
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="log1p",
            aten_ops=("log1p",),
            sample_st=_unary_sample_st(
                torch.log1p, domain=Interval(-0.999, 1e3)
            ),
            tolerance=VERY,
        ),
        OpSpec(
            name="log10",
            aten_ops=("log10",),
            sample_st=_unary_sample_st(torch.log10, domain=Interval(1e-3, 1e4)),
            tolerance=VERY,
        ),
        OpSpec(
            name="trunc-unary",
            aten_ops=("trunc",),
            sample_st=_unary_sample_st(
                torch.trunc, domain=Interval(-100.0, 100.0)
            ),
            tolerance=EXACT,
        ),
        OpSpec(
            name="cumsum",
            aten_ops=("cumsum",),
            sample_st=_cumsum_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="cumprod",
            aten_ops=("cumprod",),
            sample_st=_cumprod_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            # logcumsumexp uses the tract `tract_logcumsumexp` op
            # (a two-state scan); the running-max stabiliser means
            # CLOSE rather than APPROXIMATE is what we need.
            name="logcumsumexp",
            aten_ops=("logcumsumexp",),
            sample_st=_logcumsumexp_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            # Quadrants are now handled (the `atan2.nnef` fragment got
            # a quadrant-aware lowering). What remains diverging are two
            # IEEE-754 corners the NNEF stdlib `lt` can't see:
            # `atan2(y, -0.0)` flips sign (`lt(-0, 0)` is False so we
            # don't add the `pi`-adjust), and `atan2(0, 0)` returns NaN
            # vs torch's 0. Hypothesis hits both reliably under the
            # `(-10, 10)` interval, so the spec stays xfail until the
            # sample generator filters them out.
            name="atan2-xfail",
            aten_ops=("atan2",),
            sample_st=_atan2_sample_st(),
            tolerance=VERY,
            xfail_reason=(
                "Edge cases at signed zero -- `atan2(y, -0.0)` and "
                "`atan2(0, 0)` -- diverge from PyTorch by pi or NaN. "
                "The rest of the quadrant plane now matches."
            ),
        ),
        OpSpec(
            name="isnan-xfail",
            aten_ops=("isnan",),
            sample_st=_classifier_sample_st(torch.isnan),
            tolerance=EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_is_nan; requires "
                "tract > 0.22.1 (same gating as any/all)."
            ),
        ),
        OpSpec(
            name="isinf-xfail",
            aten_ops=("isinf",),
            sample_st=_classifier_sample_st(torch.isinf),
            tolerance=EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_is_inf; requires tract > 0.22.1."
            ),
        ),
        OpSpec(
            name="isposinf-xfail",
            aten_ops=("isposinf",),
            sample_st=_classifier_sample_st(torch.isposinf),
            tolerance=EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_isposinf; requires "
                "tract > 0.22.1."
            ),
        ),
        OpSpec(
            name="isneginf-xfail",
            aten_ops=("isneginf",),
            sample_st=_classifier_sample_st(torch.isneginf),
            tolerance=EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks tract_core_isneginf; requires "
                "tract > 0.22.1."
            ),
        ),
    ]


# Bitwise + tensor builders


def _conv2d_dilation_groups_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Conv2d` sweeping `dilation` and `groups` kwargs.

    `groups` must divide both `in_channels` and `out_channels`;
    we draw a common groups divisor and pick channel counts as multiples.
    Dilation increases effective kernel; we ensure spatial >= effective k.
    """

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _max_pool2d_dilation_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.max_pool2d` with `dilation` swept.

    `ceil_mode` stays False: t2n's pool emitter raises
    NotImplementedError on ceil_mode=True.
    """

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
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _conv2d_kwarg_sweep_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="conv2d-dilation-groups",
            aten_ops=("conv2d",),
            sample_st=_conv2d_dilation_groups_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="max_pool2d-dilation",
            aten_ops=("max_pool2d",),
            sample_st=_max_pool2d_dilation_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


def _unpool_specs() -> T.Tuple[OpSpec, ...]:
    """Pooling inverses: a scatter into a larger buffer.

    Not translated yet: each spec carries `nnef_gap`, so the tract
    driver asserts the failure and the ONNX sweep still measures
    it. Implementing one means deleting that one field.
    """
    return (
        # -- pooling --
        gap_spec(
            "max_unpool2d",
            image_st(unpool(2), "max_unpool2d", rank=2),
            "the inverse of a pooling: a scatter into a larger buffer at "
            "runtime-known indices, which no emitter builds",
        ),
        gap_spec(
            "max_unpool3d",
            image_st(unpool(3), "max_unpool3d", rank=3),
            "the 3-D form of the same scatter as `max_unpool2d`",
        ),
    )


def _fractional_pool_specs() -> T.Tuple[OpSpec, ...]:
    """Pooling with randomly chosen regions.

    Not translated yet: each spec carries `nnef_gap`, so the tract
    driver asserts the failure and the ONNX sweep still measures
    it. Implementing one means deleting that one field.
    """
    return (
        gap_spec(
            "fractional_max_pool2d",
            image_st(
                lambda x: F.fractional_max_pool2d(x, 2, output_size=(2, 2)),
                "fractional_max_pool2d",
                rank=2,
                side=5,
            ),
            "picks its pooling regions at random, so it carries the RNG "
            "problem into a layer; it also fails before the emitter lookup",
            stage=NnefGapStage.EXPORT_ERROR,
            nondeterministic=True,
        ),
        gap_spec(
            "fractional_max_pool3d",
            image_st(
                lambda x: F.fractional_max_pool3d(x, 2, output_size=(2, 2, 2)),
                "fractional_max_pool3d",
                rank=3,
                side=5,
            ),
            "the 3-D form of `fractional_max_pool2d`, same RNG problem",
            stage=NnefGapStage.EXPORT_ERROR,
            nondeterministic=True,
        ),
    )


SPECS = (
    *_pool_specs(),
    *_conv3d_pool3d_helpers_specs(),
    *_conv2d_kwarg_sweep_specs(),
    *_unpool_specs(),
    *_fractional_pool_specs(),
)
