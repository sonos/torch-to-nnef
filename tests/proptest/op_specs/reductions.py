"""Spec builders for the reductions op group."""

import typing as T

import torch
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    TensorFnPrimitive,
    UnaryPrimitive,
)
from ..inputs import Interval, tensor_st
from ..joint import (
    reduction_dim_st,
)
from ._common import (
    NnefGapStage,
    OpSample,
    OpSpec,
)
from ._gap_common import (
    REASON_DATA_BINS,
    REASON_SORT,
    gap_spec,
    matrix_rows_st,
    small_int_st,
    unary_st,
    vector_st,
)


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
            module=TensorFnPrimitive(method_name, kwargs=kwargs),
        )

    return _draw()


def _sum_full_or_multi_dim_sample_st() -> st.SearchStrategy[OpSample]:
    """Sum with the full dim surface per torch.sum doc.

    PyTorch's `torch.sum` accepts `dim` as `None` (reduce all),
    a single int, or a tuple/list of ints (multi-axis reduction). The t2n
    reducer at `torch_to_nnef/op/aten/reducer.py` handles all
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
        # Keep dim sizes small (<=4): product of many values quickly
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
            module=TensorFnPrimitive(
                "prod", kwargs={"dim": dim, "keepdim": keepdim}
            ),
        )

    return _draw()


def _var_std_sample_st(method_name: str) -> st.SearchStrategy[OpSample]:
    """`var` / `std` reduction sweeping `correction` and `keepdim`.

    The refactored t2n emitter (math.py: `_emit_var_or_std_with_optional_mean`)
    handles arbitrary `correction` values and both keepdim modes via an
    explicit `mean_reduce + sub + sqr + reduce` pipeline, so this spec
    sweeps the full surface (correction in {0, 1, 2}, keepdim in {F, T},
    rank 1..4, single dim).
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
        keepdim = draw(st.booleans())
        # Bound correction by the smallest reduced-axis size minus 1
        # (denom > 0 is required by torch and the t2n emitter).
        if isinstance(dim, int):
            min_axis = shape[dim]
        elif dim is None:
            min_axis = 1
            for d in shape:
                min_axis *= d
        else:
            min_axis = min(shape[d] for d in dim)
        correction = draw(st.integers(min_value=0, max_value=min_axis - 1))
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
            module=TensorFnPrimitive(
                method_name,
                kwargs={
                    "dim": dim,
                    "keepdim": keepdim,
                    "correction": correction,
                },
            ),
        )

    return _draw()


def _minmax_full_reduction_sample_st(
    method_name: str,
) -> st.SearchStrategy[OpSample]:
    """`x.max()` / `x.min()` full reduction (no dim).

    The full-reduction overload is `aten::max(input) -> Tensor`, a single-input
    node with no dim/keepdim, which the reducer must map to a reduce over every
    axis. Regression for the crash exporting Qwen2.5/3-VL vision towers
    (`grid_thw.max()`).
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
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        return OpSample(inputs=(x,), module=TensorFnPrimitive(method_name))

    return _draw()


def _reduction_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            # Even single-axis reductions can hit catastrophic cancellation
            # under adversarial inputs (hypothesis happily draws `[31, 31,
            # ..., 31, -64, -64.92]` whose sum cancels to ~-5 with absolute
            # f32 error around a few ULPs). Same rationale as
            # `sum-dim-broad` / `sum-dtype`; APPROXIMATE's 1e-6 atol is too
            # tight.
            name="sum-dim",
            aten_ops=("sum",),
            sample_st=_reduction_sample_st("sum"),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="sum-dim-broad",
            aten_ops=("sum",),
            sample_st=_sum_full_or_multi_dim_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            # See `sum-dim`: post-cancellation absolute error exceeds
            # APPROXIMATE for the mean as well.
            name="mean-dim",
            aten_ops=("mean",),
            sample_st=_reduction_sample_st("mean"),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="max-dim",
            aten_ops=("max",),
            sample_st=_reduction_sample_st("max"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="min-dim",
            aten_ops=("min",),
            sample_st=_reduction_sample_st("min"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="max-full",
            aten_ops=("max",),
            sample_st=_minmax_full_reduction_sample_st("max"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="min-full",
            aten_ops=("min",),
            sample_st=_minmax_full_reduction_sample_st("min"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="amax-dim",
            aten_ops=("amax",),
            sample_st=_reduction_sample_st("amax"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="amin-dim",
            aten_ops=("amin",),
            sample_st=_reduction_sample_st("amin"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        # Argmax / argmin return int64 indices: the comparator's exact
        # int path catches any divergence. Pure index ops, no tolerance
        # needed.
        OpSpec(
            name="argmax-dim",
            aten_ops=("argmax",),
            sample_st=_reduction_sample_st("argmax"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="argmin-dim",
            aten_ops=("argmin",),
            sample_st=_reduction_sample_st("argmin"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        # any / all are bool reductions (input bool, output bool).
        # tract 0.22.1 (latest in TractNNEF.OFFICIAL_SUPPORTED_VERSIONS)
        # does NOT define `any_reduce` / `all_reduce` operators: they
        # were added in tract > 0.22.1. The curated test at
        # `tests/test_primitive.py` skips these via
        # `cond_tract_gt_0_22_0`. Xfail until the supported version set
        # bumps past 0.22.1.
        OpSpec(
            name="any-dim-xfail",
            aten_ops=("any",),
            sample_st=_bool_reduction_sample_st("any"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks any_reduce; introduced in tract > "
                "0.22.1. Bumping TractNNEF.OFFICIAL_SUPPORTED_VERSIONS "
                "will flip this back to a normal pass."
            ),
        ),
        OpSpec(
            name="all-dim-xfail",
            aten_ops=("all",),
            sample_st=_bool_reduction_sample_st("all"),
            tolerance=TractCheckTolerance.EXACT,
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
            aten_ops=("prod",),
            sample_st=_prod_dim_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        # var / std sweep the full (dim, correction, keepdim) surface
        # against the refactored t2n emitter. Not opted into the
        # dyn-axes variant: the kept-dim intermediates created by
        # `_make_intermediate_ntensor` declare concrete numeric shapes,
        # which clash with tract's symbolic-dim resolution
        # ("d_axis0_sizeN should be equal to N"). Lifting that needs
        # the intermediate-shape builder to track dyn-axis symbols.
        OpSpec(
            name="var-dim",
            aten_ops=("var",),
            sample_st=_var_std_sample_st("var"),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_skip_reason=(
                "var family kept-dim intermediates declare concrete "
                "shapes; symbolic-dim threading needs follow-up."
            ),
        ),
        OpSpec(
            name="std-dim",
            aten_ops=("std",),
            sample_st=_var_std_sample_st("std"),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_skip_reason=(
                "var family kept-dim intermediates declare concrete "
                "shapes; symbolic-dim threading needs follow-up."
            ),
        ),
    ]


# Shape op specs (5)


def _reduction_dtype_kwarg_sample_st(
    method_name: str,
) -> st.SearchStrategy[OpSample]:
    """Reduction with the `dtype` cast-then-reduce kwarg.

    PyTorch's `sum/mean/prod(dim, *, dtype=)` casts the input to
    `dtype` BEFORE reducing: useful for f16 -> f32 accumulation.
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


def _reduction_dtype_kwarg_specs() -> T.List[OpSpec]:
    APPROX = TractCheckTolerance.APPROXIMATE
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            # The strategy domain `[-1e2, 1e2]` can draw values that
            # cancel down to ~1e-1 (e.g. `34 - 31 - 1.37 - 1.37`). The
            # post-cancellation absolute error is then ~`N * eps * max|x|`
            # which is a few ULPs at f32 eps and exceeds the APPROXIMATE
            # 1e-6 atol. Match `sum-dim-broad`'s rationale (accumulated
            # float error needs CLOSE).
            name="sum-dtype",
            aten_ops=("sum",),
            sample_st=_reduction_dtype_kwarg_sample_st("sum"),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="mean-dtype",
            aten_ops=("mean",),
            sample_st=_reduction_dtype_kwarg_sample_st("mean"),
            tolerance=APPROX,
        ),
        OpSpec(
            name="prod-dtype",
            aten_ops=("prod",),
            sample_st=_reduction_dtype_kwarg_sample_st("prod"),
            tolerance=CLOSE,
        ),
    ]


def _aminmax_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.aminmax(x, dim, keepdim)` returns a `(min, max)` tuple."""

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
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        op_fn = (lambda d, k: lambda t: torch.aminmax(t, dim=d, keepdim=k))(
            dim, keepdim
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _aminmax_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="aminmax",
            aten_ops=("aminmax",),
            sample_st=_aminmax_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            # Routes through `reducer_helper` like amax/amin -- the
            # static-axes path and the tract_core_shape_of dynamic-axes
            # path are shared, so dynamic axis 0 works out of the box.
            dynamic_axes_compatible=True,
        ),
    ]


def _var_std_mean_sample_st(
    method_name: str,
) -> st.SearchStrategy[OpSample]:
    """`var_mean` / `std_mean` -- multi-output forms.

    Same surface as `_var_std_sample_st`; the wrapped op returns a
    `(var_or_std, mean)` tuple which the comparator handles
    transparently via the multi-output NPZ path.
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
        keepdim = draw(st.booleans())
        if isinstance(dim, int):
            min_axis = shape[dim]
        elif dim is None:
            min_axis = 1
            for d in shape:
                min_axis *= d
        else:
            min_axis = min(shape[d] for d in dim)
        correction = draw(st.integers(min_value=0, max_value=min_axis - 1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1e2, 1e2),
            )
        )
        op_fn = (
            lambda d, c, k: (
                lambda t: getattr(torch, method_name)(
                    t, dim=d, correction=c, keepdim=k
                )
            )
        )(dim, correction, keepdim)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _var_std_mean_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="var_mean",
            aten_ops=("var_mean",),
            sample_st=_var_std_mean_sample_st("var_mean"),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_skip_reason=(
                "var family kept-dim intermediates declare concrete "
                "shapes; symbolic-dim threading needs follow-up."
            ),
        ),
        OpSpec(
            name="std_mean",
            aten_ops=("std_mean",),
            sample_st=_var_std_mean_sample_st("std_mean"),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_skip_reason=(
                "var family kept-dim intermediates declare concrete "
                "shapes; symbolic-dim threading needs follow-up."
            ),
        ),
    ]


# Registry assembly


def _order_statistic_specs() -> T.Tuple[OpSpec, ...]:
    """Rank-based reductions: a sort plus a gather, uncomposed.

    Not translated yet: each spec carries `nnef_gap`, so the tract
    driver asserts the failure and the ONNX sweep still measures
    it. Implementing one means deleting that one field.
    """
    return (
        # -- order statistics --
        gap_spec("median", unary_st(torch.median, "median"), REASON_SORT),
        gap_spec(
            "nanmedian", unary_st(torch.nanmedian, "nanmedian"), REASON_SORT
        ),
        gap_spec(
            "msort", unary_st(torch.msort, "msort", min_rank=2), REASON_SORT
        ),
        gap_spec(
            "kthvalue",
            unary_st(
                lambda x: torch.kthvalue(x, 1, dim=-1)[0],
                "kthvalue",
                min_rank=2,
            ),
            REASON_SORT,
        ),
        gap_spec(
            "quantile",
            unary_st(lambda x: torch.quantile(x, 0.5), "quantile"),
            REASON_SORT,
        ),
        gap_spec(
            "nanquantile",
            unary_st(lambda x: torch.nanquantile(x, 0.5), "nanquantile"),
            REASON_SORT,
        ),
        gap_spec(
            "mode",
            small_int_st(
                lambda x: torch.mode(x, dim=-1)[0], "mode", min_rank=2
            ),
            "needs a count-and-compare over equal values, which has no "
            "NNEF idiom",
        ),
    )


def _histogram_specs() -> T.Tuple[OpSpec, ...]:
    """Counting reductions whose output extent follows the data.

    Not translated yet: each spec carries `nnef_gap`, so the tract
    driver asserts the failure and the ONNX sweep still measures
    it. Implementing one means deleting that one field.
    """
    return (
        # -- histograms and counting --
        gap_spec(
            "bincount",
            vector_st(
                torch.bincount,
                "bincount",
                dtype=torch.int64,
                domain=Interval(0, 6),
            ),
            "output length is `max(input) + 1`, so it is data-dependent",
        ),
        gap_spec(
            "histc",
            vector_st(lambda x: torch.histc(x, bins=4), "histc"),
            REASON_DATA_BINS,
        ),
        gap_spec(
            "histogram",
            vector_st(lambda x: torch.histogram(x, bins=4)[0], "histogram"),
            REASON_DATA_BINS,
        ),
        gap_spec(
            "histogramdd",
            # `bins` must have one entry per innermost axis, so the column
            # count is fixed rather than drawn.
            matrix_rows_st(
                lambda x: torch.histogramdd(x, bins=[3, 3])[0],
                "histogramdd",
                min_cols=2,
                max_cols=2,
            ),
            f"{REASON_DATA_BINS}; also fails before the emitter lookup",
            stage=NnefGapStage.EXPORT_ERROR,
        ),
    )


def _correlation_specs() -> T.Tuple[OpSpec, ...]:
    """Second-order statistics over an axis.

    Not translated yet: each spec carries `nnef_gap`, so the tract
    driver asserts the failure and the ONNX sweep still measures
    it. Implementing one means deleting that one field.
    """
    return (
        # -- statistics --
        gap_spec(
            "corrcoef",
            matrix_rows_st(torch.corrcoef, "corrcoef"),
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
    )


SPECS = (
    *_reduction_specs(),
    *_reduction_dtype_kwarg_specs(),
    *_aminmax_specs(),
    *_var_std_mean_specs(),
    *_order_statistic_specs(),
    *_histogram_specs(),
    *_correlation_specs(),
)
