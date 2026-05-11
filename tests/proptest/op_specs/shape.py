"""Spec builders for the shape op group."""

import typing as T
from functools import partial

import torch
import torch.nn.functional as F
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    BinaryPrimitive,
    TensorFnPrimitive,
    TernaryPrimitive,
    UnaryPrimitive,
)
from ..inputs import Interval, tensor_st
from ..joint import (
    permutation_st,
    reshape_target_st,
    transpose_dims_st,
)
from ..shapes import (
    shape_st,
    ternary_broadcast_shapes_st,
)
from ._common import (
    OpSample,
    OpSpec,
)


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
            module=UnaryPrimitive(partial(torch.permute, dims=perm)),
        )

    return _draw()


def _numpy_T_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.T` -- reverse-all-axes transpose."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=0, max_value=4))
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
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(lambda t: t.T),
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
            module=UnaryPrimitive(partial(torch.unsqueeze, dim=dim)),
        )

    return _draw()


def _squeeze_sample_st() -> st.SearchStrategy[OpSample]:
    """Squeeze on a dim that is guaranteed to have size 1.

    PyTorch's `squeeze(dim=k)` on a non-1 dim is a no-op, but tract rejects
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
            module=UnaryPrimitive(partial(torch.squeeze, dim=squeeze_dim)),
        )

    return _draw()


def _view_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.view(*shape)`: like reshape but requires contiguous input."""

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
            module=TensorFnPrimitive("view", args=tuple(target_shape)),
        )

    return _draw()


def _flatten_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.flatten(start_dim, end_dim)` over a random valid range."""

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
            module=TensorFnPrimitive(
                "flatten",
                kwargs={"start_dim": start_dim, "end_dim": end_dim},
            ),
        )

    return _draw()


def _narrow_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.narrow(x, dim, start, length)` with start+length <= dim size."""

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
            module=UnaryPrimitive(
                partial(torch.narrow, dim=dim, start=start, length=length)
            ),
        )

    return _draw()


def _expand_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.expand(*sizes)`: each source dim must be 1 or equal."""

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
            module=TensorFnPrimitive("expand", args=tuple(target)),
        )

    return _draw()


def _repeat_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.repeat(*sizes)`: repeats the tensor along each dim.

    PyTorch allows `len(sizes) >= rank`, with the source treated as if
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
            module=TensorFnPrimitive("repeat", args=(list(sizes),)),
        )

    return _draw()


def _t_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.t()`: rank<=2, transposes 2-D, no-op for 0-D / 1-D."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=0, max_value=2))
        if rank == 0:
            shape = ()
        elif rank == 1:
            shape = (draw(st.integers(min_value=1, max_value=4)),)
        else:
            shape = (
                draw(st.integers(min_value=1, max_value=4)),
                draw(st.integers(min_value=1, max_value=4)),
            )
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
            module=UnaryPrimitive(lambda t: t.t()),
        )

    return _draw()


def _square_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.square))

    return _draw()


def _dot_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=6))
        a = draw(
            tensor_st(
                (n,),
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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(torch.dot))

    return _draw()


def _mv_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        m = draw(st.integers(min_value=1, max_value=4))
        n = draw(st.integers(min_value=1, max_value=5))
        a = draw(
            tensor_st(
                (m, n),
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
        return OpSample(inputs=(a, b), module=BinaryPrimitive(torch.mv))

    return _draw()


def _eye_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.eye(n)` (or `(n, m)`) added to a passthrough input.

    The op has no tensor inputs, so we tie ``x`` into the graph via a
    `0 * x.sum()` term: it always evaluates to 0, leaving the output
    equal to `eye(n, m)`, while keeping the proptest harness's
    expectation of at least one declared input.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=5))
        m = draw(st.integers(min_value=1, max_value=5))
        x = draw(
            tensor_st(
                (1,),
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        op_fn = (lambda nn, mm: lambda t: torch.eye(nn, mm) + 0.0 * t.sum())(
            n, m
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _expand_as_sample_st() -> st.SearchStrategy[OpSample]:
    """`x.expand_as(y)` with `x` carrying size-1 dims that match `y`."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        target = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # Source has rank==target rank; each axis is either 1 or
        # equal to target.
        source = tuple((1 if draw(st.booleans()) else d) for d in target)
        x = draw(
            tensor_st(
                source,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        y = draw(
            tensor_st(
                target,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x, y),
            module=BinaryPrimitive(lambda a, b: a.expand_as(b)),
        )

    return _draw()


def _reshape_as_sample_st() -> st.SearchStrategy[OpSample]:
    """`x.reshape_as(y)`: x and y have the same total element count."""

    @st.composite
    def _draw(draw) -> OpSample:
        source = draw(shape_st(min_rank=1, max_rank=4, min_dim=1, max_dim=4))
        target = draw(reshape_target_st(source, max_rank=4))
        x = draw(
            tensor_st(
                source,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        y = draw(
            tensor_st(
                tuple(target),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x, y),
            module=BinaryPrimitive(lambda a, b: a.reshape_as(b)),
        )

    return _draw()


def _broadcast_to_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
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
                domain=Interval(-10.0, 10.0),
            )
        )
        op_fn = (lambda t: lambda a: torch.broadcast_to(a, t))(target)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _atleast_sample_st(n: int) -> st.SearchStrategy[OpSample]:
    """`torch.atleast_{n}d(x)` for ranks 0..3."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=0, max_value=3))
        if rank == 0:
            shape = ()
        else:
            shape = tuple(
                draw(
                    st.lists(
                        st.integers(min_value=1, max_value=3),
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
                domain=Interval(-10.0, 10.0),
            )
        )
        op = {1: torch.atleast_1d, 2: torch.atleast_2d, 3: torch.atleast_3d}[n]
        return OpSample(inputs=(x,), module=UnaryPrimitive(op))

    return _draw()


def _tile_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.tile(x, dims)` covers `len(dims) <`, `==`, `>` x.dim()."""

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
        n_dims = draw(st.integers(min_value=1, max_value=rank + 1))
        dims = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=3),
                    min_size=n_dims,
                    max_size=n_dims,
                )
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
        op_fn = (lambda d: lambda a: torch.tile(a, d))(dims)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _floor_divide_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=3, min_dim=1, max_dim=4))
        a = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        # Avoid zeros in divisor (and don't probe the very-small-magnitude
        # band where rounding direction near zero crossings can flip).
        b_pos = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(0.5, 5.0),
            )
        )
        sign = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        b = torch.where(sign >= 0, b_pos, -b_pos)
        return OpSample(
            inputs=(a, b),
            module=BinaryPrimitive(torch.floor_divide),
        )

    return _draw()


def _nan_to_num_sample_st() -> st.SearchStrategy[OpSample]:
    """`nan_to_num(x, nan=0, posinf=+M, neginf=-M)` over a mix of values."""

    @st.composite
    def _draw(draw) -> OpSample:
        # Build a tensor of 8 elements: a few NaN/+inf/-inf and finite
        # numbers, so the comparator exercises every branch.
        finite_count = draw(st.integers(min_value=2, max_value=5))
        finite = draw(
            tensor_st(
                (finite_count,),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        specials = torch.tensor(
            [float("nan"), float("inf"), -float("inf")],
            dtype=torch.float32,
        )
        x = torch.cat([finite, specials])
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                partial(torch.nan_to_num, nan=0.0, posinf=1e6, neginf=-1e6)
            ),
        )

    return _draw()


def _cosine_similarity_sample_st() -> st.SearchStrategy[OpSample]:
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
        # Domain bounded away from 0 to keep the safe-denom path stable.
        a = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        b = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        op_fn = (lambda d: lambda x, y: torch.cosine_similarity(x, y, dim=d))(
            dim
        )
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _shape_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="reshape",
            sample_st=_reshape_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="transpose",
            sample_st=_transpose_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="permute",
            sample_st=_permute_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="numpy_T",
            sample_st=_numpy_T_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="unsqueeze",
            sample_st=_unsqueeze_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="squeeze",
            sample_st=_squeeze_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="view",
            sample_st=_view_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="flatten",
            sample_st=_flatten_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="narrow",
            sample_st=_narrow_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="expand",
            sample_st=_expand_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="repeat",
            sample_st=_repeat_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="t",
            sample_st=_t_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="square",
            sample_st=_square_sample_st(),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="dot",
            sample_st=_dot_sample_st(),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="mv",
            sample_st=_mv_sample_st(),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="eye",
            sample_st=_eye_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="expand_as",
            sample_st=_expand_as_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=False,
            dynamic_axes_skip_reason=(
                "expand_as still routes through `_emit_static_expand` "
                "and asserts `all(int)` on `other.shape`; the dynamic "
                "path needs a refactor of `expand.py::_append_repeats_"
                "on_existing_dims` so the helper can be shared."
            ),
        ),
        OpSpec(
            name="reshape_as",
            sample_st=_reshape_as_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=False,
            dynamic_axes_skip_reason=(
                "reshape_as feeds the second input's runtime shape "
                "into NNEF reshape; tract's symbolic-dim checker "
                "can't verify `prod(target_dims) == prod(source_dims)` "
                "when both sides involve different dynamic axes "
                "(e.g. `d_axis0_sizeM == d_axis0_sizeN * literal`). "
                "The op is dynamic-axes-correct in real models where "
                "the trace pins the relationship; only the proptest "
                "harness's same-rank-different-shape draws trip it."
            ),
        ),
        OpSpec(
            name="broadcast_to",
            sample_st=_broadcast_to_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=False,
            dynamic_axes_skip_reason=(
                "Strategy generates a literal target shape; under "
                "dynamic-axes the source's axis 0 is symbolic and "
                "tract can't prove the broadcast rule "
                "(symbolic == literal). The aliased `aten::expand` "
                "handler itself works fine when target dims also "
                "come from runtime tensors."
            ),
        ),
        OpSpec(
            name="atleast_1d",
            sample_st=_atleast_sample_st(1),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="atleast_2d",
            sample_st=_atleast_sample_st(2),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="atleast_3d",
            sample_st=_atleast_sample_st(3),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="tile",
            sample_st=_tile_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="floor_divide",
            sample_st=_floor_divide_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="nan_to_num",
            sample_st=_nan_to_num_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="cosine_similarity",
            sample_st=_cosine_similarity_sample_st(),
            tolerance=TractCheckTolerance.APPROXIMATE,
            dynamic_axes_compatible=True,
        ),
    ]


# clamp + where (2)


def _select_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.select(dim, index)`: pick a single slice along dim.

    Output rank = input rank - 1; index must be in `[0, dim_size)`.
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
            module=TensorFnPrimitive(
                "select", kwargs={"dim": dim, "index": index}
            ),
        )

    return _draw()


def _index_select_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.index_select(input, dim, index_tensor)`.

    The index tensor has int64 dtype and 1-D shape; values in
    `[0, dim_size)`. Output replaces the selected dim with the index
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
        # `torch.index_select` is positional-only on dim
        # (`index_select(input, dim, index)`). `partial(..., dim=dim)`
        # would inject dim as a kwarg, which the schema rejects.
        op_fn = (lambda d: lambda t, ix: torch.index_select(t, d, ix))(dim)
        return OpSample(
            inputs=(x, idx),
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _gather_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.gather(input, dim, index)`: index has same rank as input.

    For each output position, the value is
    `input[i_0, ..., index[i], ..., i_{n-1}]` along the gather dim.
    The index tensor's shape can differ from input only in the gather
    dim; values in `[0, input.shape[dim])`.
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
        # Index shape == input shape, except at `dim` where it can vary.
        idx_dim_size = draw(st.integers(min_value=1, max_value=4))
        idx_shape = list(shape)
        idx_shape[dim] = idx_dim_size
        # Build idx values in valid range: via hypothesis (not
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
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _masked_fill_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.masked_fill(mask, value)`: bool mask, scalar value."""

    @st.composite
    def _draw(draw) -> OpSample:
        # `masked_fill` accepts any broadcastable mask; this strategy
        # keeps them same-shape for simplicity.
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
            module=BinaryPrimitive(lambda t, m: t.masked_fill(m, value)),
        )

    return _draw()


def _topk_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.topk(input, k, dim)` returns (values, indices).

    Tie-breaking on equal values is implementation-defined; PyTorch and
    tract pick different indices when the input has duplicates. To make
    the indices output well-defined, we feed a permutation of integers
    1..N as the input: every value is unique so the index output is
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
        # Draw a permutation of 1..n then reshape: unique values, no ties.
        perm = draw(st.permutations(list(range(1, n + 1))))
        x = torch.tensor(perm, dtype=torch.float32).reshape(shape)
        return OpSample(
            inputs=(x,),
            module=TensorFnPrimitive("topk", kwargs={"k": k, "dim": dim}),
        )

    return _draw()


def _selector_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="select",
            sample_st=_select_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="index_select",
            sample_st=_index_select_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="gather",
            sample_st=_gather_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="masked_fill",
            sample_st=_masked_fill_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="topk",
            sample_st=_topk_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


# Pooling specs


class _CatPair(torch.nn.Module):
    """Wrapper for `torch.cat([a, b], dim=k)`: list-of-2 form."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.cat([a, b], dim=self.dim)


class _StackPair(torch.nn.Module):
    """Wrapper for `torch.stack([a, b], dim=k)`: list-of-2 form."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.stack([a, b], dim=self.dim)


def _cat_sample_st() -> st.SearchStrategy[OpSample]:
    """`cat([a, b], dim)`: joint shape: a/b agree on every non-cat dim."""

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
        return OpSample(inputs=(a, b), module=_CatPair(dim))

    return _draw()


def _stack_sample_st() -> st.SearchStrategy[OpSample]:
    """`stack([a, b], dim)`: joint shape: a and b have identical shape."""

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
        return OpSample(inputs=(a, b), module=_StackPair(dim))

    return _draw()


def _chunk_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.chunk(chunks, dim)`: multi-output split.

    PyTorch's chunk handles non-divisible `shape[dim]` gracefully (last
    chunk is smaller). The t2n split emitter at
    `torch_to_nnef/op/aten/split.py` asserts equal-sized chunks and
    raises `AssertionError` otherwise: so our strategy enforces
    `shape[dim] % chunks == 0`.
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
            module=TensorFnPrimitive(
                "chunk", kwargs={"chunks": chunks, "dim": dim}
            ),
        )

    return _draw()


def _split_int_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.split(t, split_size, dim)` -- int form: chunks + remainder."""

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
        split_size = draw(
            st.integers(min_value=1, max_value=max(1, shape_list[dim]))
        )
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
            module=TensorFnPrimitive(
                "split", kwargs={"split_size": split_size, "dim": dim}
            ),
        )

    return _draw()


def _unfold_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.unfold(dim, size, step)`: sliding-window view."""

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
        size = draw(st.integers(min_value=1, max_value=shape_list[dim]))
        step = draw(st.integers(min_value=1, max_value=max(1, shape_list[dim])))
        shape = tuple(shape_list)
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )

        class _Unfold(torch.nn.Module):
            def forward(self, t):
                return t.unfold(dim, size, step)

        return OpSample(inputs=(x,), module=_Unfold())

    return _draw()


def _im2col_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.unfold(input, kernel_size, dilation, padding, stride)`.

    Lowers to `aten::im2col`; rank-4 inputs `(N, C, H, W)` only.
    Sample sizes / kernels / strides / dilations / paddings are kept
    small but exercise both square and asymmetric configurations, and
    cover the padding > 0 / dilation > 1 branches.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        kh = draw(st.integers(min_value=1, max_value=3))
        kw = draw(st.integers(min_value=1, max_value=3))
        dh = draw(st.integers(min_value=1, max_value=2))
        dw = draw(st.integers(min_value=1, max_value=2))
        ph = draw(st.integers(min_value=0, max_value=1))
        pw = draw(st.integers(min_value=0, max_value=1))
        sh = draw(st.integers(min_value=1, max_value=2))
        sw = draw(st.integers(min_value=1, max_value=2))
        rcpt_h = dh * (kh - 1) + 1
        rcpt_w = dw * (kw - 1) + 1
        h = draw(st.integers(min_value=rcpt_h, max_value=rcpt_h + 4))
        w = draw(st.integers(min_value=rcpt_w, max_value=rcpt_w + 4))
        x = draw(
            tensor_st(
                (n, c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )

        class _Im2Col(torch.nn.Module):
            def forward(self, t):
                return torch.nn.functional.unfold(
                    t,
                    kernel_size=(kh, kw),
                    dilation=(dh, dw),
                    padding=(ph, pw),
                    stride=(sh, sw),
                )

        return OpSample(inputs=(x,), module=_Im2Col())

    return _draw()


def _col2im_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.fold(input, output_size, kernel_size, dilation, padding, stride)`.

    Lowers to `aten::col2im`. Samples a rank-4 image `(N, C, H, W)`,
    runs `F.unfold` to get the col representation, then `F.fold`
    inverts it; the proptest checks that the t2n emitter reproduces
    torch's output (which is the per-position sum of overlapping
    kernel contributions, scaled by overlap count).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        kh = draw(st.integers(min_value=1, max_value=3))
        kw = draw(st.integers(min_value=1, max_value=3))
        dh = draw(st.integers(min_value=1, max_value=2))
        dw = draw(st.integers(min_value=1, max_value=2))
        ph = draw(st.integers(min_value=0, max_value=1))
        pw = draw(st.integers(min_value=0, max_value=1))
        sh = draw(st.integers(min_value=1, max_value=2))
        sw = draw(st.integers(min_value=1, max_value=2))
        rcpt_h = dh * (kh - 1) + 1
        rcpt_w = dw * (kw - 1) + 1
        out_h = draw(st.integers(min_value=rcpt_h, max_value=rcpt_h + 4))
        out_w = draw(st.integers(min_value=rcpt_w, max_value=rcpt_w + 4))
        padded_h = out_h + 2 * ph
        padded_w = out_w + 2 * pw
        n_h = (padded_h - rcpt_h) // sh + 1
        n_w = (padded_w - rcpt_w) // sw + 1
        # F.fold input is (N, C*kH*kW, L).
        x = draw(
            tensor_st(
                (n, c * kh * kw, n_h * n_w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )

        class _Col2Im(torch.nn.Module):
            def forward(self, t):
                return torch.nn.functional.fold(
                    t,
                    output_size=(out_h, out_w),
                    kernel_size=(kh, kw),
                    dilation=(dh, dw),
                    padding=(ph, pw),
                    stride=(sh, sw),
                )

        return OpSample(inputs=(x,), module=_Col2Im())

    return _draw()


def _unbind_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.unbind(input, dim)`: splits into a tuple of slices."""

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
            module=UnaryPrimitive(partial(torch.unbind, dim=dim)),
        )

    return _draw()


def _roll_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.roll(input, shifts, dims)`: cyclic shift.

    Sweeps the full PyTorch range for `shifts`: positive, negative,
    zero, and magnitudes >= dim_size. Tract's slice/concat path has
    issues with shift=0 and shift==dim_size (output shape doubles), but
    the t2n roll emitter at `torch_to_nnef/op/aten/concat.py` now
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
            module=UnaryPrimitive(partial(torch.roll, shifts=shift, dims=dim)),
        )

    return _draw()


def _outer_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.outer(a, b)`: both inputs are 1-D, result is 2-D."""

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
            module=BinaryPrimitive(torch.outer),
        )

    return _draw()


def _triangular_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """`torch.tril/triu(input, diagonal)`: requires rank >= 2."""

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
            module=UnaryPrimitive(partial(op, diagonal=diagonal)),
        )

    return _draw()


def _matrix_transpose_sample_st(
    op_fn,
) -> st.SearchStrategy[OpSample]:
    """`Tensor.mT` / `Tensor.mH` -- swap the last two axes (rank >= 2)."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
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
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _fliplr_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
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
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.fliplr))

    return _draw()


def _flipud_sample_st() -> st.SearchStrategy[OpSample]:
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
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.flipud))

    return _draw()


def _rot90_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.rot90(input, k, dims)`: rotate by 90 * k degrees."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        k = draw(st.integers(min_value=0, max_value=3))
        d0 = draw(st.integers(min_value=0, max_value=rank - 1))
        d1 = draw(
            st.integers(min_value=0, max_value=rank - 1).filter(
                lambda v: v != d0
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
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(torch.rot90, k=k, dims=(d0, d1))),
        )

    return _draw()


def _flip_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.flip(input, dims)`: reverse along a unique subset of dims."""

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
        n_dims = draw(st.integers(min_value=1, max_value=rank))
        dims = draw(
            st.lists(
                st.integers(min_value=0, max_value=rank - 1),
                min_size=n_dims,
                max_size=n_dims,
                unique=True,
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
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(torch.flip, dims=tuple(dims))),
        )

    return _draw()


def _diagonal_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.diagonal(input, offset, dim1, dim2)`.

    Shapes on `dim1` / `dim2` are drawn independently to exercise the
    non-square slice path. `offset` is drawn in
    `[-(shape[dim1] - 1), shape[dim2] - 1]` so the resulting diagonal
    has length >= 1 (the t2n emitter rejects empty-diagonal cases).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
        shape = list(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim1 = draw(st.integers(min_value=0, max_value=rank - 1))
        candidates = [d for d in range(rank) if d != dim1]
        dim2 = draw(st.sampled_from(candidates))
        s1 = shape[dim1]
        s2 = shape[dim2]
        offset = draw(st.integers(min_value=-(s1 - 1), max_value=s2 - 1))
        x = draw(
            tensor_st(
                tuple(shape),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                partial(
                    torch.diagonal,
                    offset=offset,
                    dim1=dim1,
                    dim2=dim2,
                )
            ),
        )

    return _draw()


def _concat_split_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="cat",
            sample_st=_cat_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="stack",
            sample_st=_stack_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="chunk",
            sample_st=_chunk_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="split-int",
            sample_st=_split_int_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="unfold",
            sample_st=_unfold_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="im2col",
            sample_st=_im2col_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="col2im",
            sample_st=_col2im_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="unbind",
            sample_st=_unbind_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="roll",
            sample_st=_roll_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="outer",
            sample_st=_outer_sample_st(),
            tolerance=TractCheckTolerance.APPROXIMATE,
        ),
        OpSpec(
            name="tril",
            sample_st=_triangular_sample_st(torch.tril),
            tolerance=EXACT,
        ),
        OpSpec(
            name="triu",
            sample_st=_triangular_sample_st(torch.triu),
            tolerance=EXACT,
        ),
        OpSpec(
            name="flip",
            sample_st=_flip_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="mT",
            sample_st=_matrix_transpose_sample_st(lambda x: x.mT),
            tolerance=EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="mH",
            sample_st=_matrix_transpose_sample_st(lambda x: x.mH),
            tolerance=EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="fliplr",
            sample_st=_fliplr_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="flipud",
            sample_st=_flipud_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="rot90",
            sample_st=_rot90_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="diagonal",
            sample_st=_diagonal_sample_st(),
            tolerance=EXACT,
        ),
    ]


# Padding specs


def _pad_sample_st(
    mode: str, max_pad_per_side: T.Optional[int] = None
) -> st.SearchStrategy[OpSample]:
    """`F.pad(input, pad, mode, value)`.

    PyTorch's `pad` list is right-to-left: `[L_-1, R_-1, L_-2, R_-2, ...]`
    where `L_i` and `R_i` are left/right padding for axis -i. Up to
    `rank` axes can be padded.

    Reflection and replication modes require `pad <= dim_size - 1` (for
    reflect) or `pad <= dim_size` (for replicate), so `max_pad_per_side`
    bounds the strategy accordingly.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        # `reflect`/`replicate` need rank>=3 (N, C, spatial...) since
        # they only operate on spatial dims. `constant` accepts any rank.
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
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _pad_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="pad-constant",
            sample_st=_pad_sample_st("constant", max_pad_per_side=3),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="pad-reflect",
            sample_st=_pad_sample_st("reflect"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="pad-replicate-xfail",
            sample_st=_pad_sample_st("replicate"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 does not implement NNEF pad mode "
                '"replicate" ("unsupported padding mode replicate"). '
                "t2n's replication_padnd emitter passes through the "
                "mode attribute; the gap is downstream in tract."
            ),
        ),
    ]


# Norm variants (vector norm, frobenius, linalg, rms)


def _sort_sample_st(method_name: str) -> st.SearchStrategy[OpSample]:
    """`Tensor.sort(dim, descending)` / `Tensor.argsort(dim, descending)`.

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
            module=TensorFnPrimitive(
                method_name, kwargs={"dim": dim, "descending": descending}
            ),
        )

    return _draw()


def _scatter_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.scatter(dim, index, src)`: counterpart of gather."""

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
        # `dim`. Index values must be in [0, input.shape[dim]).
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
        # `Tensor.scatter` is positional-only on dim; same lambda wrapper
        # pattern as index_select / gather.
        op_fn = (lambda d: lambda t, i, s: t.scatter(d, i, s))(dim)
        return OpSample(
            inputs=(x, idx, src),
            module=TernaryPrimitive(op_fn),
        )

    return _draw()


def _slice_sample_st() -> st.SearchStrategy[OpSample]:
    """Python slice via `__getitem__`: maps to `aten:slice`.

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

        return OpSample(inputs=(x,), module=UnaryPrimitive(slice_fn))

    return _draw()


def _scatter_add_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.scatter_add(dim, index, src)`: in-place add via scatter."""

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
        op_fn = (lambda d: lambda t, i, s: t.scatter_add(d, i, s))(dim)
        return OpSample(inputs=(x, idx, src), module=TernaryPrimitive(op_fn))

    return _draw()


def _scatter_reduce_sample_st(
    reduce_mode: str,
) -> st.SearchStrategy[OpSample]:
    """`Tensor.scatter_reduce(dim, index, src, reduce, include_self=True)`."""

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
        op_fn = (
            lambda d, r: (
                lambda t, i, s: t.scatter_reduce(
                    d, i, s, reduce=r, include_self=True
                )
            )
        )(dim, reduce_mode)
        return OpSample(inputs=(x, idx, src), module=TernaryPrimitive(op_fn))

    return _draw()


def _select_scatter_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.select_scatter(input, src, dim, index)`: write src at index.

    Restricted to rank >= 2 because `tensor_st(())` produces shape `[1]`
    (1-D), not 0-D, so a rank-1 input + rank-0 src case can't be drawn
    cleanly. The op itself supports rank 1, but the strategy doesn't.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=3))
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
        index = draw(st.integers(min_value=0, max_value=shape[dim] - 1))
        # `src` has rank = input.rank - 1 (drops `dim`).
        src_shape = shape[:dim] + shape[dim + 1 :]
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        src = draw(
            tensor_st(
                src_shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        op_fn = (lambda d, i: lambda t, s: torch.select_scatter(t, s, d, i))(
            dim, index
        )
        return OpSample(inputs=(x, src), module=BinaryPrimitive(op_fn))

    return _draw()


def _slice_scatter_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.slice_scatter(input, src, dim, start, end, step=1)`."""

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
        dim_size = shape[dim]
        start = draw(st.integers(min_value=0, max_value=dim_size - 1))
        end = draw(st.integers(min_value=start + 1, max_value=dim_size))
        # `src` matches input shape except at `dim` where it's (end-start).
        src_shape = list(shape)
        src_shape[dim] = end - start
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        src = draw(
            tensor_st(
                tuple(src_shape),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        op_fn = (
            lambda d, st_, en: (
                lambda t, s: torch.slice_scatter(
                    t, s, dim=d, start=st_, end=en, step=1
                )
            )
        )(dim, start, end)
        return OpSample(inputs=(x, src), module=BinaryPrimitive(op_fn))

    return _draw()


def _sort_scatter_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="sort",
            sample_st=_sort_sample_st("sort"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="argsort",
            sample_st=_sort_sample_st("argsort"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="scatter",
            sample_st=_scatter_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        # scatter reduction landed in tract 0.23.0-dev.4 (#2109). The
        # CI runtime is the published 0.22.1, which silently ignores the
        # NNEF `reduction` attribute and runs overwrite (the t2n
        # emitter raises T2NErrorNotImplemented under that version).
        # Flip these to non-xfail once tract 0.23 ships stable.
        OpSpec(
            name="scatter_add-xfail",
            sample_st=_scatter_add_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="scatter_reduce-sum-xfail",
            sample_st=_scatter_reduce_sample_st("sum"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="scatter_reduce-prod-xfail",
            sample_st=_scatter_reduce_sample_st("prod"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="scatter_reduce-amax-xfail",
            sample_st=_scatter_reduce_sample_st("amax"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="scatter_reduce-amin-xfail",
            sample_st=_scatter_reduce_sample_st("amin"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="select_scatter",
            sample_st=_select_scatter_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="slice_scatter",
            sample_st=_slice_scatter_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="slice",
            sample_st=_slice_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


def _index_fill_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.index_fill(dim, index, value)`: scalar-fill at indices."""

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
        idx_len = draw(st.integers(min_value=1, max_value=shape[dim]))
        idx = draw(
            tensor_st(
                (idx_len,),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[dim] - 1),
            )
        )
        value = draw(st.floats(min_value=-10.0, max_value=10.0))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        op_fn = (lambda d, v: lambda t, i: t.index_fill(d, i, v))(dim, value)
        return OpSample(inputs=(x, idx), module=BinaryPrimitive(op_fn))

    return _draw()


def _index_copy_sample_st(reduction: str) -> st.SearchStrategy[OpSample]:
    """`Tensor.index_copy / index_add(dim, index, source)`."""

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
        idx_len = draw(st.integers(min_value=1, max_value=shape[dim]))
        idx = draw(
            tensor_st(
                (idx_len,),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[dim] - 1),
            )
        )
        # index_copy requires unique indices (otherwise undefined which
        # write wins); regenerate until unique.
        idx_unique = torch.unique(idx)
        if reduction == "none" and idx_unique.numel() != idx.numel():
            idx = idx_unique
            idx_len = int(idx.numel())
        src_shape = list(shape)
        src_shape[dim] = idx_len
        src = draw(
            tensor_st(
                tuple(src_shape),
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
        if reduction == "none":
            op_fn = (lambda d: lambda t, i, s: t.index_copy(d, i, s))(dim)
        else:
            op_fn = (lambda d: lambda t, i, s: t.index_add(d, i, s))(dim)
        return OpSample(inputs=(x, idx, src), module=TernaryPrimitive(op_fn))

    return _draw()


def _take_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.take(self, index)`: gather from the flat view of `self`."""

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
        numel = 1
        for d in shape:
            numel *= d
        idx_len = draw(st.integers(min_value=1, max_value=numel))
        idx = draw(
            tensor_st(
                (idx_len,),
                torch.int64,
                finite=True,
                domain=Interval(0, numel - 1),
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
        return OpSample(inputs=(x, idx), module=BinaryPrimitive(torch.take))

    return _draw()


def _pixel_shuffle_sample_st(*, downscale: bool) -> st.SearchStrategy[OpSample]:
    """`torch.pixel_(un)shuffle(x, factor)` over rank-4 / rank-3 inputs."""

    @st.composite
    def _draw(draw) -> OpSample:
        leading_rank = draw(st.integers(min_value=0, max_value=1))
        r = draw(st.integers(min_value=2, max_value=3))
        if downscale:
            h_units = draw(st.integers(min_value=1, max_value=3))
            w_units = draw(st.integers(min_value=1, max_value=3))
            c_in = draw(st.integers(min_value=1, max_value=3))
            spatial = (c_in, h_units * r, w_units * r)
        else:
            c = draw(st.integers(min_value=1, max_value=3))
            h = draw(st.integers(min_value=1, max_value=4))
            w = draw(st.integers(min_value=1, max_value=4))
            spatial = (c * r * r, h, w)
        leading = tuple(
            draw(st.integers(min_value=1, max_value=2))
            for _ in range(leading_rank)
        )
        shape = leading + spatial
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        op = torch.pixel_unshuffle if downscale else torch.pixel_shuffle
        op_fn = (lambda f: lambda t: op(t, f))(r)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _index_pixel_specs() -> T.List[OpSpec]:
    return [
        # The index_* family lowers to `tract_core_scatter_elements` with
        # a `reduction` attribute that landed in tract 0.23.0-dev.4
        # (#2109). Same xfail story as scatter_add / scatter_reduce: the
        # CI runtime is 0.22.1, the t2n emitter hard-errors below
        # 0.23.0-dev.4. Flip these once tract 0.23 ships stable.
        OpSpec(
            name="index_fill-xfail",
            sample_st=_index_fill_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="index_copy-xfail",
            sample_st=_index_copy_sample_st("none"),
            tolerance=TractCheckTolerance.EXACT,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        OpSpec(
            name="index_add-xfail",
            sample_st=_index_copy_sample_st("add"),
            tolerance=TractCheckTolerance.APPROXIMATE,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4."
            ),
        ),
        # `take` uses `tract_core_gather`, no scatter reduction, so it
        # passes on 0.22.1. The flatten reshape uses `[-1]` (deferred
        # to tract / NNEF), so dyn-axes works without special-casing.
        OpSpec(
            name="take",
            sample_st=_take_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="pixel_shuffle",
            sample_st=_pixel_shuffle_sample_st(downscale=False),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="pixel_unshuffle",
            sample_st=_pixel_shuffle_sample_st(downscale=True),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


def _broadcast_tensors_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.broadcast_tensors(t0, t1, t2)` over 3 broadcastable inputs."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa, sb, sc = draw(ternary_broadcast_shapes_st(max_rank=3, max_dim=4))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        c = draw(
            tensor_st(
                sc, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )

        def op_fn(x, y, z):
            return torch.broadcast_tensors(x, y, z)

        return OpSample(inputs=(a, b, c), module=TernaryPrimitive(op_fn))

    return _draw()


def _meshgrid_sample_st(indexing: str) -> st.SearchStrategy[OpSample]:
    """`torch.meshgrid(a, b, indexing=...)` over rank-1 inputs."""

    @st.composite
    def _draw(draw) -> OpSample:
        sa = draw(st.integers(min_value=1, max_value=4))
        sb = draw(st.integers(min_value=1, max_value=4))
        a = draw(
            tensor_st(
                (sa,), torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        b = draw(
            tensor_st(
                (sb,), torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        op_fn = (lambda idx: lambda x, y: torch.meshgrid(x, y, indexing=idx))(
            indexing
        )
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _tensor_split_sample_st(mode: str) -> st.SearchStrategy[OpSample]:
    """`torch.tensor_split(x, sections_or_indices, dim)`.

    `mode='int'` sweeps the integer-sections form; `mode='list'` sweeps
    the boundary-indices form. Indices form is constrained so that
    boundaries are strictly increasing and within `[1, dim_size - 1]`.
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
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        if mode == "int":
            sections = draw(st.integers(min_value=1, max_value=shape[dim]))
            op_fn = (lambda s, d: lambda t: torch.tensor_split(t, s, dim=d))(
                sections, dim
            )
        else:
            n_indices = draw(
                st.integers(min_value=1, max_value=max(1, shape[dim] - 1))
            )
            indices = sorted(
                draw(
                    st.lists(
                        st.integers(min_value=1, max_value=shape[dim] - 1),
                        min_size=n_indices,
                        max_size=n_indices,
                        unique=True,
                    )
                )
            )
            op_fn = (
                lambda idxs, d: lambda t: torch.tensor_split(t, idxs, dim=d)
            )(indices, dim)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _shape_utility_specs() -> T.List[OpSpec]:
    return [
        # `tract_core_broadcast` is shape-only and rank-preserving, so
        # dyn-axes works out of the box.
        OpSpec(
            name="broadcast_tensors",
            sample_st=_broadcast_tensors_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        # `resolve_attr_axis_size` threads each input's axis-0 dim
        # through the reshape + broadcast attrs as a runtime
        # identifier, so dyn-axes works.
        OpSpec(
            name="meshgrid_ij",
            sample_st=_meshgrid_sample_st("ij"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="meshgrid_xy",
            sample_st=_meshgrid_sample_st("xy"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        # tensor_split bakes slice boundaries from the trace-time
        # `dim_size`. Static-axes proptest passes; runtime correctness
        # across different dyn-axis sizes is a known follow-up.
        OpSpec(
            name="tensor_split-int",
            sample_st=_tensor_split_sample_st("int"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_skip_reason=(
                "tensor_split bakes boundaries from trace-time dim_size."
            ),
        ),
        OpSpec(
            name="tensor_split-indices",
            sample_st=_tensor_split_sample_st("list"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_skip_reason=(
                "tensor_split bakes boundaries from trace-time dim_size."
            ),
        ),
    ]


class _StackList(torch.nn.Module):
    """Wrapper for `torch.{vstack,hstack,dstack}([a, b])`."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b):
        return self.fn([a, b])


def _axis_stack_sample_st(fn, *, min_rank: int) -> st.SearchStrategy[OpSample]:
    """Joint shape: two tensors of identical shape, rank>=min_rank.

    `vstack` / `hstack` / `dstack` accept matching shapes (after torch's
    upstream rank-promotion); we feed them with already-promoted ranks
    to match what the t2n emitter sees.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=min_rank, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
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
        return OpSample(inputs=(a, b), module=_StackList(fn))

    return _draw()


def _axis_split_sample_st(
    fn, *, dim: int, min_rank: int
) -> st.SearchStrategy[OpSample]:
    """`vsplit/hsplit/dsplit(x, sections)`: int sections only.

    Torch's `*split` (unlike `tensor_split`) requires the dim to be
    divisible by `sections`. The strategy enforces that by drawing
    `sections` then `multiplier` and setting `shape[dim] = sections *
    multiplier`.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=min_rank, max_value=4))
        shape = list(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        sections = draw(st.integers(min_value=1, max_value=3))
        mult = draw(st.integers(min_value=1, max_value=3))
        shape[dim] = sections * mult
        x = draw(
            tensor_st(
                tuple(shape),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(fn, sections=sections)),
        )

    return _draw()


def _count_nonzero_sample_st(*, all_dims: bool) -> st.SearchStrategy[OpSample]:
    """`torch.count_nonzero(x, dim?)`: half-density mask via {-1, 0, 1}."""

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
        # Sparse-ish input with a real mix of zeros so the count
        # actually exercises the reduce, not just `numel`.
        x = draw(
            tensor_st(shape, torch.int64, finite=True, domain=Interval(-1, 1))
        )
        if all_dims:
            op_fn = lambda t: torch.count_nonzero(t)  # noqa: E731
        else:
            dim = draw(st.integers(min_value=0, max_value=rank - 1))
            op_fn = (lambda d: lambda t: torch.count_nonzero(t, dim=d))(dim)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _alias_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="dstack",
            sample_st=_axis_stack_sample_st(torch.dstack, min_rank=3),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="vsplit",
            sample_st=_axis_split_sample_st(torch.vsplit, dim=0, min_rank=2),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="hsplit",
            sample_st=_axis_split_sample_st(torch.hsplit, dim=1, min_rank=2),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="dsplit",
            sample_st=_axis_split_sample_st(torch.dsplit, dim=2, min_rank=3),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="count_nonzero-dim",
            sample_st=_count_nonzero_sample_st(all_dims=False),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="count_nonzero-all",
            sample_st=_count_nonzero_sample_st(all_dims=True),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
    ]


def _index_put_sample_st(accumulate: bool) -> st.SearchStrategy[OpSample]:
    """`out[idx] = src` (or `+=` when `accumulate=True`) along axis 0.

    Only the single-axis form is exercised -- the t2n emitter rejects
    multi-axis / mask indices with `NotImplementedError`. Indices are
    drawn unique for the overwrite case (torch's `index_put_(...,
    accumulate=False)` with duplicate indices has undefined ordering).
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
        idx_len = draw(st.integers(min_value=1, max_value=shape[0]))
        idx = draw(
            tensor_st(
                (idx_len,),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[0] - 1),
            )
        )
        idx_unique = torch.unique(idx)
        if not accumulate and idx_unique.numel() != idx.numel():
            idx = idx_unique
            idx_len = int(idx.numel())
        src_shape = list(shape)
        src_shape[0] = idx_len
        src = draw(
            tensor_st(
                tuple(src_shape),
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
        op_fn = (
            lambda acc: (
                lambda t, i, s: t.clone().index_put_((i,), s, accumulate=acc)
            )
        )(accumulate)
        return OpSample(inputs=(x, idx, src), module=TernaryPrimitive(op_fn))

    return _draw()


def _bucketize_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.bucketize(input, boundaries)` over a 1-D sorted boundary."""

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
        n_b = draw(st.integers(min_value=1, max_value=4))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        # boundaries must be sorted
        raw_b = draw(
            tensor_st(
                (n_b,),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        b, _ = torch.sort(raw_b)
        right = draw(st.booleans())
        op_fn = (lambda r: lambda xx, bb: torch.bucketize(xx, bb, right=r))(
            right
        )
        return OpSample(inputs=(x, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _searchsorted_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.searchsorted(sorted_seq, values)`: args swapped vs bucketize."""

    @st.composite
    def _draw(draw) -> OpSample:
        n_seq = draw(st.integers(min_value=1, max_value=4))
        raw_seq = draw(
            tensor_st(
                (n_seq,),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        seq, _ = torch.sort(raw_seq)
        rank = draw(st.integers(min_value=1, max_value=3))
        vals_shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        vals = draw(
            tensor_st(
                vals_shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        right = draw(st.booleans())
        op_fn = (lambda r: lambda s, v: torch.searchsorted(s, v, right=r))(
            right
        )
        return OpSample(inputs=(seq, vals), module=BinaryPrimitive(op_fn))

    return _draw()


def _bucketize_searchsorted_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="bucketize",
            sample_st=_bucketize_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="searchsorted",
            sample_st=_searchsorted_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        # index_put(accumulate=False) just overwrites -- the
        # `reduction` NNEF attr is ignored on tract 0.22.1 (default
        # path is overwrite), so it works on stable.
        OpSpec(
            name="index_put",
            sample_st=_index_put_sample_st(accumulate=False),
            tolerance=TractCheckTolerance.EXACT,
        ),
        # accumulate=True needs the ScatterReduction support which
        # landed in tract 0.23.0-dev.4; same xfail as the rest of the
        # scatter family.
        OpSpec(
            name="index_put-accum-xfail",
            sample_st=_index_put_sample_st(accumulate=True),
            tolerance=TractCheckTolerance.APPROXIMATE,
            xfail_reason=(
                "tract 0.22.1 lacks ScatterReduction; t2n hard-errors "
                "below 0.23.0-dev.4 when accumulate=True."
            ),
        ),
    ]


# 3D conv/pool + numerical helpers + classifiers

SPECS = (
    *_shape_specs(),
    *_concat_split_specs(),
    *_pad_specs(),
    *_sort_scatter_specs(),
    *_selector_specs(),
    *_index_pixel_specs(),
    *_shape_utility_specs(),
    *_alias_specs(),
    *_bucketize_searchsorted_specs(),
)
