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
    """`Tensor.view(*shape)` -- like reshape but requires contiguous input."""

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
    """`Tensor.expand(*sizes)` -- each source dim must be 1 or equal."""

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
    """`Tensor.repeat(*sizes)` -- repeats the tensor along each dim.

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
    ]


# clamp + where (2)


def _select_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.select(dim, index)` -- pick a single slice along dim.

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
    """`torch.gather(input, dim, index)` -- index has same rank as input.

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
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _masked_fill_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.masked_fill(mask, value)` -- bool mask, scalar value."""

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
    """Wrapper for `torch.cat([a, b], dim=k)` -- list-of-2 form."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.cat([a, b], dim=self.dim)


class _StackPair(torch.nn.Module):
    """Wrapper for `torch.stack([a, b], dim=k)` -- list-of-2 form."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.stack([a, b], dim=self.dim)


def _cat_sample_st() -> st.SearchStrategy[OpSample]:
    """`cat([a, b], dim)` -- joint shape: a/b agree on every non-cat dim."""

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
    """`stack([a, b], dim)` -- joint shape: a and b have identical shape."""

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
    """`Tensor.chunk(chunks, dim)` -- multi-output split.

    PyTorch's chunk handles non-divisible `shape[dim]` gracefully (last
    chunk is smaller). The t2n split emitter at
    `torch_to_nnef/op/aten/split.py` asserts equal-sized chunks and
    raises `AssertionError` otherwise -- so our strategy enforces
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


def _unbind_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.unbind(input, dim)` -- splits into a tuple of slices."""

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
    """`torch.roll(input, shifts, dims)` -- cyclic shift.

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
    """`torch.outer(a, b)` -- both inputs are 1-D, result is 2-D."""

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
    """`torch.tril/triu(input, diagonal)` -- requires rank >= 2."""

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
    """`Tensor.scatter(dim, index, src)` -- counterpart of gather."""

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
    """Python slice via `__getitem__` -- maps to `aten:slice`.

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
        OpSpec(
            name="slice",
            sample_st=_slice_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


# 3D conv/pool + numerical helpers + classifiers

SPECS = (
    *_shape_specs(),
    *_concat_split_specs(),
    *_pad_specs(),
    *_sort_scatter_specs(),
    *_selector_specs(),
)
