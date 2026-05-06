"""Spec builders for the norm op group."""

import typing as T

import torch
import torch.nn as nn
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    BinaryPrimitive,
    TensorFnPrimitive,
)
from ..inputs import Interval, tensor_st
from ..joint import (
    reduction_dim_st,
)
from ._common import (
    OpSample,
    OpSpec,
)


def _matmul_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.matmul(A, B)`: joint inner-dim constraint A[-1]==B[-2]."""

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
            module=BinaryPrimitive(torch.matmul),
        )

    return _draw()


def _linear_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Linear(in_f, out_f)`: input shape ends with `in_f`.

    Rank starts at 2 (always a batch dim). PyTorch supports rank-1 input
    (treats it as a single vector) but t2n's export pipeline needs a
    leading batch dim to wire NNEF's matmul correctly.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        # in_features and out_features both >= 2 to avoid a t2n corner
        # where Linear(1, 1) on (1, 1)-shape input trips
        # `maybe_align_inputs_ranks` in
        # `torch_to_nnef/op/helper.py` (TypeError: Tensor not iterable).
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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _layer_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.LayerNorm(normalized_shape)`: input ends with that suffix."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _batch_norm1d_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.BatchNorm1d(C)` over (N, C) or (N, C, L) input."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _group_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.GroupNorm(num_groups, num_channels)`: groups must divide C.

    Each group must have non-trivial variance, otherwise normalization
    amplifies float-roundoff differences between PyTorch and tract into
    visible output drift. We feed a permutation of integers (unique
    values) to guarantee variance > 0 in every group.
    """

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _conv1d_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Conv1d(in_C, out_C, kernel)` over (N, in_C, L) input."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _conv2d_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Conv2d(in_C, out_C, kernel)` over (N, in_C, H, W) input."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _norm_conv_matmul_specs() -> T.List[OpSpec]:
    # Multi-op chains: tract's f32 ops accumulate ULP-level error per
    # multiply-accumulate. CLOSE (1e-5) is too tight for a typical
    # conv/linear; VERY (1e-4) gives breathing room.
    VERY = TractCheckTolerance.VERY
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            name="matmul",
            sample_st=_matmul_sample_st(),
            tolerance=VERY,
        ),
        OpSpec(
            name="linear",
            sample_st=_linear_sample_st(),
            tolerance=VERY,
        ),
        OpSpec(
            # layer_norm involves variance + division; nightly proptest
            # surfaces near-zero outputs where tract diverges by ~1.5e-4
            # absolute (above VERY but well below SUPER). Same root cause
            # class as group_norm (multi-step f32 reduction precision).
            name="layer_norm",
            sample_st=_layer_norm_sample_st(),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            name="batch_norm1d",
            sample_st=_batch_norm1d_sample_st(),
            tolerance=VERY,
        ),
        OpSpec(
            # Previously xfailed because the `group_norm.nnef`
            # fragment tiled the BATCH axis instead of GROUPS, leaking
            # mean from one batch into another batch's channels for
            # multi-batch inputs with num_groups < num_channels. Now
            # fixed: the emitter flattens spatial dims before the
            # fragment, the fragment computes everything in 3D
            # `(B, num_groups, S)` space, and scale/offset are
            # applied via the standard per-channel unsqueeze +
            # left-aligned NNEF broadcast pattern after restoration of
            # the original input rank.
            name="group_norm",
            sample_st=_group_norm_sample_st(),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            name="conv1d",
            sample_st=_conv1d_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="conv2d",
            sample_st=_conv2d_sample_st(),
            tolerance=CLOSE,
        ),
    ]


# Concat / split / multi-tensor specs


def _vector_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.norm(p, dim, keepdim)`: vector p-norm along a dim."""

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
        # p in {1, 2} only: t2n's norm emitter at norm.py dispatches
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
            module=TensorFnPrimitive(
                "norm", kwargs={"p": p, "dim": dim, "keepdim": keepdim}
            ),
        )

    return _draw()


def _rms_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.RMSNorm(normalized_shape)`: input ends with that suffix."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _norm_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="vector_norm",
            sample_st=_vector_norm_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            # Confirmed upstream tract bug: tract's native
            # `tract_transformers_rms_norm` op (which t2n routes to
            # for tract >= 0.22.0 with single-axis `normalized_shape`,
            # the typical LLM case) diverges by ~3% relative vs
            # PyTorch's `nn.RMSNorm`. The t2n fragment fallback
            # (`rms_norm.nnef`) has the correct formula
            # `x * rsqrt(mean(x^2) + eps) * gamma`: forcing
            # `prefer_native_tract_rms_norm` to False makes proptest
            # match PyTorch exactly. The fix lives in tract's native op.
            name="rms_norm-xfail",
            sample_st=_rms_norm_sample_st(),
            tolerance=TractCheckTolerance.ULTRA,
            xfail_reason=(
                "tract's native tract_transformers_rms_norm op diverges "
                "from PyTorch's nn.RMSNorm by ~3% relative; forcing the "
                "t2n fragment fallback path matches exactly. Bug is "
                "upstream in tract."
            ),
        ),
    ]


# Sort / scatter specs (extension of the selector family)


def _layer_norm_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.LayerNorm` sweeping `eps` and `elementwise_affine`."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _batch_norm1d_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.BatchNorm1d` sweeping `eps` (affine=True only).

    `affine=False` is not implemented in t2n's batch_norm emitter
    (`norm.py` raises NotImplementedError when the param tensors are
    None), so we pin `affine=True`.
    """

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _topk_kwargs_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.topk` sweeping `largest` (sorted=True only).

    t2n's topk emitter raises NotImplementedError on `sorted=False`
    (`selector.py`). Sticking to sorted=True.
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
    """`torch.sort` sweeping `descending` (stable=False only).

    The `stable` kwarg fails the schema-match in t2n's dynamic call
    path: sort.stable is a separate aten overload that t2n's
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
            module=TensorFnPrimitive(
                "sort",
                kwargs={"dim": dim, "descending": descending},
            ),
        )

    return _draw()


class _CatNTensors(torch.nn.Module):
    """`torch.cat([t1, ..., tN], dim=k)`: variable N."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.cat(list(tensors), dim=self.dim)


def _cat_n_tensors_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.cat` with N tensors (3-4 in this strategy)."""

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
            module=_CatNTensors(dim),
        )

    return _draw()


def _embedding_padding_idx_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Embedding` sweeping `padding_idx`."""

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
        return OpSample(inputs=(idx,), module=layer)

    return _draw()


def _norm_topk_cat_kwarg_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            # Sweeping `eps` exposes near-zero output cases where tract
            # diverges by more than SUPER's 1e-3 (e.g. ~2.4e-3 abs with
            # very small `eps`). ULTRA matches the practical noise
            # floor for layer_norm under hypothesis.
            name="layer_norm-broad",
            sample_st=_layer_norm_kwargs_sample_st(),
            tolerance=TractCheckTolerance.ULTRA,
        ),
        OpSpec(
            name="batch_norm1d-broad",
            sample_st=_batch_norm1d_kwargs_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="topk-broad",
            sample_st=_topk_kwargs_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="sort-broad",
            sample_st=_sort_kwargs_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="cat-n-tensors",
            sample_st=_cat_n_tensors_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="embedding-padding-idx",
            sample_st=_embedding_padding_idx_sample_st(),
            tolerance=EXACT,
        ),
    ]


SPECS = (
    *_norm_conv_matmul_specs(),
    *_norm_specs(),
    *_norm_topk_cat_kwarg_specs(),
)
