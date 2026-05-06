"""Spec builders for the specialty op group."""

import typing as T
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    UnaryPrimitive,
)
from ..inputs import Interval, tensor_st
from ..shapes import (
    shape_st,
)
from ._common import (
    OpSample,
    OpSpec,
)


def _embedding_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Embedding(num_embeddings, embedding_dim)` -- index lookup."""

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
        return OpSample(inputs=(idx,), module=layer)

    return _draw()


def _repeat_interleave_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.repeat_interleave(input, repeats, dim)` -- scalar repeats."""

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
            module=UnaryPrimitive(
                partial(torch.repeat_interleave, repeats=repeats, dim=dim)
            ),
        )

    return _draw()


def _upsample_nearest2d_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.UpsamplingNearest2d(scale_factor=N)` -- (N, C, H, W) input."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _specialty_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="embedding",
            sample_st=_embedding_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="repeat_interleave",
            sample_st=_repeat_interleave_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="upsample_nearest2d",
            sample_st=_upsample_nearest2d_sample_st(),
            tolerance=EXACT,
        ),
    ]


# prelu / glu / einsum


def _prelu_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.PReLU(num_parameters=1)` -- shared slope across all channels."""

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _prelu_multi_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.PReLU(num_parameters=C)` -- per-channel slope.

    PyTorch broadcasts `weight` of shape `(C,)` along the channel
    axis (`dim=1`) of an input shaped `(N, C, *spatial)`. Because
    NNEF broadcasts left-aligned, the t2n emitter pre-unsqueezes the
    weight to `(C, 1, 1, ...)` before emit -- see
    `torch_to_nnef/op/aten/activation.py:prelu`.
    """

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _glu_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.glu(input, dim)` -- splits input in half along dim, gates."""

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
            module=UnaryPrimitive(partial(F.glu, dim=dim)),
        )

    return _draw()


class _Einsum2Op(torch.nn.Module):
    """Wrapper that calls `torch.einsum(expr, a, b)`."""

    def __init__(self, expr: str):
        super().__init__()
        self.expr = expr

    def forward(self, a, b):
        return torch.einsum(self.expr, a, b)


def _einsum_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.einsum(expr, a, b)` -- a small set of canonical patterns.

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
        return OpSample(inputs=(a, b), module=_Einsum2Op(expr))

    return _draw()


def _prelu_glu_einsum_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="prelu",
            sample_st=_prelu_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="prelu-multi",
            sample_st=_prelu_multi_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="glu",
            sample_st=_glu_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="einsum",
            sample_st=_einsum_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
    ]


# Final user-facing ops (max_pool*_with_indices, dropout, index)


def _max_pool2d_with_indices_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.max_pool2d(..., return_indices=True)` -- multi-output.

    Like topk, indices are tie-breaking-dependent. We feed a permutation
    of integers as input to make every value unique and indices
    deterministic.
    """

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
        return OpSample(inputs=(x,), module=UnaryPrimitive(wrapped))

    return _draw()


def _dropout_eval_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.Dropout(p)` in eval mode -- a no-op identity.

    The export pipeline should skip dropout in eval mode (it has no
    effect at inference). Proptest sweeps shapes to confirm the no-op
    invariant holds across the export.
    """

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
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _max_pool_dropout_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="max_pool2d_with_indices",
            sample_st=_max_pool2d_with_indices_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="dropout",
            sample_st=_dropout_eval_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
    ]


# Constructors (input-less in PyTorch, wrapped with a shape-coupled input)
# + advanced index + SDPA

SPECS = (
    *_specialty_specs(),
    *_prelu_glu_einsum_specs(),
    *_max_pool_dropout_specs(),
)
