"""Spec builders for the specialty op group."""

import typing as T
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import assume
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    BinaryPrimitive,
    TernaryPrimitive,
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
    """`nn.Embedding(num_embeddings, embedding_dim)`: index lookup."""

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
    """`torch.repeat_interleave(input, repeats, dim)`: scalar repeats."""

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
    """`nn.UpsamplingNearest2d(scale_factor=N)`: (N, C, H, W) input."""

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


def _upsample_nearest_nd_sample_st(
    spatial_rank: int,
) -> st.SearchStrategy[OpSample]:
    """`F.interpolate(mode='nearest')` for 1-D / 3-D inputs."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        spatial = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=spatial_rank,
                    max_size=spatial_rank,
                )
            )
        )
        scale = draw(st.integers(min_value=2, max_value=3))
        x = draw(
            tensor_st(
                (n, c) + spatial,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )

        class _Up(nn.Module):
            def forward(self, t):
                return torch.nn.functional.interpolate(
                    t, scale_factor=float(scale), mode="nearest"
                )

        return OpSample(inputs=(x,), module=_Up().eval())

    return _draw()


def _interpolate_nd_sample_st(
    spatial_rank: int,
    mode: str,
    align_corners: T.Optional[bool],
) -> st.SearchStrategy[OpSample]:
    """`F.interpolate(mode=...)` for linear / bicubic / nearest-exact."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        spatial = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=4),
                    min_size=spatial_rank,
                    max_size=spatial_rank,
                )
            )
        )
        scale = draw(st.integers(min_value=2, max_value=3))
        x = draw(
            tensor_st(
                (n, c) + spatial,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )

        class _Up(nn.Module):
            def forward(self, t):
                return torch.nn.functional.interpolate(
                    t,
                    scale_factor=float(scale),
                    mode=mode,
                    align_corners=align_corners,
                )

        return OpSample(inputs=(x,), module=_Up().eval())

    return _draw()


def _grid_sample_sample_st(
    mode: str,
    padding_mode: str,
    align_corners: bool,
) -> st.SearchStrategy[OpSample]:
    """`F.grid_sample` on a 2-D feature map with a random sampling grid."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        h = draw(st.integers(min_value=2, max_value=4))
        w = draw(st.integers(min_value=2, max_value=4))
        out_h = draw(st.integers(min_value=2, max_value=4))
        out_w = draw(st.integers(min_value=2, max_value=4))
        x = draw(
            tensor_st(
                (n, c, h, w),
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        grid = draw(
            tensor_st(
                (n, out_h, out_w, 2),
                torch.float32,
                finite=True,
                domain=Interval(-1.5, 1.5),
            )
        )

        class _Grid(nn.Module):
            def forward(self, t, g):
                return torch.nn.functional.grid_sample(
                    t,
                    g,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )

        return OpSample(inputs=(x, grid), module=_Grid().eval())

    return _draw()


# Drafted against tract PR sonos/tract#2363 (tract_core_resize /
# tract_core_grid_sample). xfail until that lands in a released tract that the
# proptest harness downloads; then bump RESIZE_MIN_TRACT_VERSION and drop these.
_RESIZE_XFAIL = "needs released tract with tract_core_resize (sonos/tract#2363)"


def _specialty_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    APPROX = TractCheckTolerance.APPROXIMATE
    return [
        OpSpec(
            name="embedding",
            aten_ops=("embedding",),
            sample_st=_embedding_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="repeat_interleave",
            aten_ops=("repeat_interleave",),
            sample_st=_repeat_interleave_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="upsample_nearest2d",
            aten_ops=("upsample_nearest2d",),
            sample_st=_upsample_nearest2d_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="upsample_nearest1d",
            aten_ops=("upsample_nearest1d",),
            sample_st=_upsample_nearest_nd_sample_st(spatial_rank=1),
            tolerance=EXACT,
        ),
        OpSpec(
            name="upsample_nearest3d",
            aten_ops=("upsample_nearest3d",),
            sample_st=_upsample_nearest_nd_sample_st(spatial_rank=3),
            tolerance=EXACT,
        ),
        OpSpec(
            name="upsample_nearest_exact2d",
            aten_ops=("_upsample_nearest_exact2d",),
            sample_st=_interpolate_nd_sample_st(2, "nearest-exact", None),
            tolerance=EXACT,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="upsample_linear1d",
            aten_ops=("upsample_linear1d",),
            sample_st=_interpolate_nd_sample_st(1, "linear", False),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="upsample_bilinear2d",
            aten_ops=("upsample_bilinear2d",),
            sample_st=_interpolate_nd_sample_st(2, "bilinear", False),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="upsample_bilinear2d_align_corners",
            aten_ops=("upsample_bilinear2d",),
            sample_st=_interpolate_nd_sample_st(2, "bilinear", True),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="upsample_trilinear3d",
            aten_ops=("upsample_trilinear3d",),
            sample_st=_interpolate_nd_sample_st(3, "trilinear", False),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="upsample_bicubic2d",
            aten_ops=("upsample_bicubic2d",),
            sample_st=_interpolate_nd_sample_st(2, "bicubic", False),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="grid_sample_bilinear_zeros",
            aten_ops=("grid_sampler",),
            sample_st=_grid_sample_sample_st("bilinear", "zeros", False),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="grid_sample_nearest_border",
            aten_ops=("grid_sampler",),
            sample_st=_grid_sample_sample_st("nearest", "border", True),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
        OpSpec(
            name="grid_sample_bicubic_reflection",
            aten_ops=("grid_sampler",),
            sample_st=_grid_sample_sample_st("bicubic", "reflection", False),
            tolerance=APPROX,
            xfail_reason=_RESIZE_XFAIL,
        ),
    ]


# prelu / glu / einsum


def _prelu_sample_st() -> st.SearchStrategy[OpSample]:
    """`nn.PReLU(num_parameters=1)`: shared slope across all channels."""

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
    """`nn.PReLU(num_parameters=C)`: per-channel slope.

    PyTorch broadcasts `weight` of shape `(C,)` along the channel
    axis (`dim=1`) of an input shaped `(N, C, *spatial)`. Because
    NNEF broadcasts left-aligned, the t2n emitter pre-unsqueezes the
    weight to `(C, 1, 1, ...)` before emit: see
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
    """`F.glu(input, dim)`: splits input in half along dim, gates."""

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


def _einsum_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.einsum(expr, a, b)`: a small set of canonical patterns.

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
        return OpSample(
            inputs=(a, b),
            module=BinaryPrimitive(partial(torch.einsum, expr)),
        )

    return _draw()


def _prelu_glu_einsum_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="prelu",
            aten_ops=("prelu",),
            sample_st=_prelu_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="prelu-multi",
            aten_ops=("prelu",),
            sample_st=_prelu_multi_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="glu",
            aten_ops=("glu",),
            sample_st=_glu_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="einsum",
            aten_ops=("einsum",),
            sample_st=_einsum_sample_st(),
            tolerance=TractCheckTolerance.VERY,
        ),
    ]


def _max_pool2d_with_indices_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.max_pool2d(..., return_indices=True)`: multi-output.

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


def _dropout_eval_sample_st(layer_cls) -> st.SearchStrategy[OpSample]:
    """Dropout-family `layer_cls` in eval mode: a no-op identity.

    The export pipeline should skip these in eval mode (no effect at
    inference). Proptest sweeps shapes to confirm the no-op invariant
    holds across the export.
    """
    # Dropout2d zeroes whole 2D channels and wants the canonical
    # (N, C, H, W) layout (rank 4); FeatureAlphaDropout wants rank>=3.
    if layer_cls is nn.Dropout2d:
        min_rank = 4
    elif layer_cls is nn.FeatureAlphaDropout:
        min_rank = 3
    else:
        min_rank = 1

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=min_rank, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        layer = layer_cls(p=0.5).eval()
        return OpSample(inputs=(x,), module=layer)

    return _draw()


def _resolve_identity_sample_st(op_name: str) -> st.SearchStrategy[OpSample]:
    """`torch.resolve_conj` / `resolve_neg` / `conj_physical` on real input."""
    fn = getattr(torch, op_name)

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=1))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(fn))

    return _draw()


def _max_pool_dropout_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="max_pool2d_with_indices",
            aten_ops=("max_pool2d_with_indices",),
            sample_st=_max_pool2d_with_indices_sample_st(),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="dropout",
            aten_ops=("dropout",),
            sample_st=_dropout_eval_sample_st(nn.Dropout),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="alpha_dropout",
            aten_ops=("alpha_dropout",),
            sample_st=_dropout_eval_sample_st(nn.AlphaDropout),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="feature_dropout",
            aten_ops=("feature_dropout",),
            sample_st=_dropout_eval_sample_st(nn.Dropout2d),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="feature_alpha_dropout",
            aten_ops=("feature_alpha_dropout",),
            sample_st=_dropout_eval_sample_st(nn.FeatureAlphaDropout),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="resolve_conj",
            aten_ops=("resolve_conj",),
            sample_st=_resolve_identity_sample_st("resolve_conj"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="resolve_neg",
            aten_ops=("resolve_neg",),
            sample_st=_resolve_identity_sample_st("resolve_neg"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="conj_physical",
            aten_ops=("conj_physical",),
            sample_st=_resolve_identity_sample_st("conj_physical"),
            tolerance=TractCheckTolerance.EXACT,
            dynamic_axes_compatible=True,
        ),
    ]


def _pairwise_distance_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.pairwise_distance(a, b, p, eps, keepdim)` over rank-2/3 inputs."""

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
        a = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        b = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        p = draw(st.sampled_from([1.0, 2.0, 3.0]))
        keepdim = draw(st.booleans())
        op_fn = (
            lambda pp, kd: (
                lambda x, y: F.pairwise_distance(x, y, p=pp, keepdim=kd)
            )
        )(p, keepdim)
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _cross_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.cross(a, b, dim)` for rank 1..3 with size-3 along `dim`."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        # The cross axis must have size 3; other axes drawn freely.
        shape = []
        for _ in range(rank):
            shape.append(draw(st.integers(min_value=1, max_value=4)))
        dim = draw(st.integers(min_value=0, max_value=rank - 1))
        shape[dim] = 3
        shape = tuple(shape)
        a = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        b = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-5.0, 5.0)
            )
        )
        op_fn = (lambda d: lambda x, y: torch.cross(x, y, dim=d))(dim)
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _tensordot_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.tensordot(a, b, dims)`: contract one or two axes.

    All axis sizes are drawn in `[2, 3]`. The dyn-axes proptest variant
    marks each input's axis 0 as a runtime dim; tract's einsum
    optimizer's reshape pass then bails out with "Removing non-trivial
    axis #0 of dim: d_axis0_size1" when axis 0 is symbolic-but-1 (it
    can't statically prove the dim is 1, but the post-fold expectation
    is that it should be). Keeping every axis >= 2 sidesteps that
    edge case without hiding any meaningful coverage.

    We additionally forbid contracting axis 0 on BOTH inputs at once:
    the dyn-axes setup pins each input's axis 0 to the same symbolic
    `d_axis0_sizeN` name when the sizes match, so contracting both
    makes the einsum's reduction dim `K` a symbol. Tract's einsum
    codegen then folds K into the input reshape with the symbolic
    identifier instead of the resolved static value, and the reshape
    verifier rejects it as "Incompatible reshape for shape [Sym(...),
    Val(2), Val(2)] and Reshape(1, [d_axis0_sizeN, 2], [2*d_axis0_sizeN])".
    Contracting axis 0 on only one side leaves K static (the other
    side's matching size, set by the loop below), which works.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        ra = draw(st.integers(min_value=1, max_value=3))
        rb = draw(st.integers(min_value=1, max_value=3))
        n_contract = draw(st.integers(min_value=0, max_value=min(ra, rb)))
        # Pick the contracted axes on each side and a shared size for each.
        sa = [draw(st.integers(min_value=2, max_value=3)) for _ in range(ra)]
        sb = [draw(st.integers(min_value=2, max_value=3)) for _ in range(rb)]
        dims_a = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=ra - 1),
                    min_size=n_contract,
                    max_size=n_contract,
                    unique=True,
                )
            )
        )
        dims_b = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=rb - 1),
                    min_size=n_contract,
                    max_size=n_contract,
                    unique=True,
                )
            )
        )
        # Forbid contracting axis 0 on BOTH sides at once: under the
        # dyn-axes proptest variant the two axis-0 dims share a single
        # symbolic name `d_axis0_sizeN` (same size -> same dim), and
        # contracting both makes the reduction dim K symbolic, which
        # trips tract's einsum codegen.
        assume(not (0 in dims_a and 0 in dims_b))
        # Match the shared size on each contracted-axis pair.
        for da, db in zip(dims_a, dims_b, strict=True):
            sb[db] = sa[da]
        a = draw(
            tensor_st(
                tuple(sa),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        b = draw(
            tensor_st(
                tuple(sb),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        op_fn = (
            lambda da, db: lambda x, y: torch.tensordot(x, y, dims=(da, db))
        )(dims_a, dims_b)
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _dist_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.dist(a, b, p)` scalar p-norm, rank-1..3 broadcastable inputs."""

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
        p = draw(st.sampled_from([1.0, 2.0, 3.0]))
        a = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        b = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        op_fn = (lambda pp: lambda x, y: torch.dist(x, y, p=pp))(p)
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _cdist_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.cdist(a, b, p)` for rank-2 / rank-3 batched inputs."""

    @st.composite
    def _draw(draw) -> OpSample:
        batched = draw(st.booleans())
        d = draw(st.integers(min_value=1, max_value=4))
        m = draw(st.integers(min_value=1, max_value=4))
        n = draw(st.integers(min_value=1, max_value=4))
        if batched:
            bsz = draw(st.integers(min_value=1, max_value=2))
            sa = (bsz, m, d)
            sb = (bsz, n, d)
        else:
            sa = (m, d)
            sb = (n, d)
        p = draw(st.sampled_from([1.0, 2.0, 3.0]))
        a = draw(
            tensor_st(
                sa, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        b = draw(
            tensor_st(
                sb, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        op_fn = (lambda pp: lambda x, y: torch.cdist(x, y, p=pp))(p)
        return OpSample(inputs=(a, b), module=BinaryPrimitive(op_fn))

    return _draw()


def _distance_specs() -> T.List[OpSpec]:
    return [
        # All four lower to NNEF stdlib + (tensordot only) tract_core_einsum,
        # all rank-preserving / shape-only: the dyn-axes path works
        # straight away.
        OpSpec(
            name="pairwise_distance",
            aten_ops=("pairwise_distance",),
            sample_st=_pairwise_distance_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="cross",
            aten_ops=("cross",),
            sample_st=_cross_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="tensordot",
            aten_ops=("tensordot",),
            sample_st=_tensordot_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="cdist",
            aten_ops=("cdist",),
            sample_st=_cdist_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="dist",
            aten_ops=("dist",),
            sample_st=_dist_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_compatible=True,
        ),
    ]


def _embedding_bag_static_offsets_sample_st(
    mode: str,
) -> st.SearchStrategy[OpSample]:
    """`F.embedding_bag(idx, weight, offsets, mode=...)`: static offsets.

    Offsets are baked into the traced module via a `torch.tensor`
    constant, so they appear as a constant in the t2n IR (the
    typical case in real models is either this or the 2-D-input
    flattening, which produces the same effect upstream).
    """

    @st.composite
    def _draw(draw) -> OpSample:
        num_emb = draw(st.integers(min_value=2, max_value=6))
        emb_dim = draw(st.integers(min_value=2, max_value=4))
        n_bags = draw(st.integers(min_value=1, max_value=3))
        # Each bag size in [1, 3]; total `k` is sum of sizes.
        sizes = [
            draw(st.integers(min_value=1, max_value=3)) for _ in range(n_bags)
        ]
        k = sum(sizes)
        idx = draw(
            tensor_st(
                (k,),
                torch.int64,
                finite=True,
                domain=Interval(0, num_emb - 1),
            )
        )
        weights = draw(
            tensor_st(
                (num_emb, emb_dim),
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        offsets_list = [0]
        for s in sizes[:-1]:
            offsets_list.append(offsets_list[-1] + s)
        offsets_t = torch.tensor(offsets_list, dtype=torch.int64)

        class _EB(nn.Module):
            def __init__(self, off, m):
                super().__init__()
                self.register_buffer("off", off)
                self.m = m

            def forward(self, w, ix):
                return F.embedding_bag(ix, w, self.off, mode=self.m)

        return OpSample(inputs=(weights, idx), module=_EB(offsets_t, mode))

    return _draw()


def _affine_grid_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.affine_grid(theta, (N, C, H, W), align_corners)`: 2-D only."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=2))
        c = draw(st.integers(min_value=1, max_value=3))
        h = draw(st.integers(min_value=2, max_value=5))
        w = draw(st.integers(min_value=2, max_value=5))
        align_corners = draw(st.booleans())
        theta = draw(
            tensor_st(
                (n, 2, 3),
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )

        class _AG(nn.Module):
            def __init__(self, nn_, cc, hh, ww, ac):
                super().__init__()
                self.size = (nn_, cc, hh, ww)
                self.ac = ac

            def forward(self, th):
                return F.affine_grid(th, self.size, align_corners=self.ac)

        return OpSample(inputs=(theta,), module=_AG(n, c, h, w, align_corners))

    return _draw()


def _conv_tbc_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.conv_tbc(x, w, b, pad)` with baked weight/bias."""

    @st.composite
    def _draw(draw) -> OpSample:
        kernel = draw(st.integers(min_value=1, max_value=3))
        c_in = draw(st.integers(min_value=1, max_value=3))
        c_out = draw(st.integers(min_value=1, max_value=3))
        b = draw(st.integers(min_value=1, max_value=2))
        t = draw(st.integers(min_value=kernel + 1, max_value=8))
        pad = draw(st.integers(min_value=0, max_value=kernel // 2))
        x = draw(
            tensor_st(
                (t, b, c_in),
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )

        class _CTBC(nn.Module):
            def __init__(self, kk, ci, co, pp):
                super().__init__()
                self.w = nn.Parameter(torch.randn(kk, ci, co))
                self.bias = nn.Parameter(torch.randn(co))
                self.pad = pp

            def forward(self, xx):
                return torch.conv_tbc(xx, self.w, self.bias, pad=self.pad)

        return OpSample(
            inputs=(x,), module=_CTBC(kernel, c_in, c_out, pad).eval()
        )

    return _draw()


def _linalg_matrix_norm_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.linalg.matrix_norm(x, 'fro', dim, keepdim)`."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=5),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        keepdim = draw(st.booleans())
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-5.0, 5.0),
            )
        )
        op_fn = (
            lambda kd: (
                lambda t: torch.linalg.matrix_norm(t, ord="fro", keepdim=kd)
            )
        )(keepdim)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _no_tract_change_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="embedding_bag-sum",
            aten_ops=("embedding_bag",),
            sample_st=_embedding_bag_static_offsets_sample_st("sum"),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="embedding_bag-mean",
            aten_ops=("embedding_bag",),
            sample_st=_embedding_bag_static_offsets_sample_st("mean"),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="embedding_bag-max",
            aten_ops=("embedding_bag",),
            sample_st=_embedding_bag_static_offsets_sample_st("max"),
            tolerance=TractCheckTolerance.EXACT,
        ),
        OpSpec(
            name="affine_grid",
            aten_ops=("affine_grid_generator",),
            sample_st=_affine_grid_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            # `resolve_attr_axis_size` threads theta's dynamic batch
            # dim through the final reshape; H/W still need to be
            # statically known (we bake the base grid as a constant).
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="conv_tbc",
            aten_ops=("conv_tbc",),
            sample_st=_conv_tbc_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
        ),
        OpSpec(
            name="linalg_matrix_norm_fro",
            aten_ops=("linalg_matrix_norm",),
            sample_st=_linalg_matrix_norm_sample_st(),
            tolerance=TractCheckTolerance.CLOSE,
            dynamic_axes_compatible=True,
        ),
    ]


# --- Recently-shipped ops (distance + fused matmul cluster) ---


def _pdist_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.pdist(x, p)`: rank-2 input with static N (2..5).

    The t2n emitter at `torch_to_nnef/op/aten/math.py:pdist` requires
    a rank-2 input with a statically known N (the upper-triangular
    gather indices are baked into the graph), and supports `p in {1, 2}`.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=2, max_value=5))
        d = draw(st.integers(min_value=1, max_value=4))
        p = draw(st.sampled_from([1.0, 2.0]))
        x = draw(
            tensor_st(
                (n, d), torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        op_fn = (lambda pp: lambda t: torch.pdist(t, p=pp))(p)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _renorm_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.renorm(self, p, dim, maxnorm)`: rank-2-or-3 input.

    Mix rows that exceed `maxnorm` (get scaled) with rows that don't
    (stay untouched) so both branches of the fragment get exercised.

    `p` restricted to {1, 2}: the `norm_pn` fragment branch (p != 1
    and p != 2) currently fails tract type-resolution
    (`No super type for F32 and TDim` on the `pow(input, ord)` call
    inside `norm_pn`): the `ord` attribute reaches tract as a TDim
    integer instead of a float scalar.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=3))
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
        p = draw(st.sampled_from([1.0, 2.0]))
        # maxnorm spans values that some draws will exceed; keep finite
        # input range so 'exceed' is the common case.
        maxnorm = draw(
            st.floats(
                min_value=0.5,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        op_fn = (
            lambda pp, dd, mm: (
                lambda t: torch.renorm(t, p=pp, dim=dd, maxnorm=mm)
            )
        )(p, dim, maxnorm)
        return OpSample(inputs=(x,), module=UnaryPrimitive(op_fn))

    return _draw()


def _addbmm_sample_st() -> st.SearchStrategy[OpSample]:
    """Shapes `(n, p) + (b, n, m) @ (b, m, p)` for `torch.addbmm`."""

    @st.composite
    def _draw(draw) -> OpSample:
        b = draw(st.integers(min_value=1, max_value=3))
        n = draw(st.integers(min_value=1, max_value=4))
        m = draw(st.integers(min_value=1, max_value=4))
        p = draw(st.integers(min_value=1, max_value=4))
        # Bound magnitudes so the b-fold inner sum stays within f32
        # precision compared to PyTorch's accumulation.
        dom = Interval(-2.0, 2.0)
        self_t = draw(tensor_st((n, p), torch.float32, finite=True, domain=dom))
        batch1 = draw(
            tensor_st((b, n, m), torch.float32, finite=True, domain=dom)
        )
        batch2 = draw(
            tensor_st((b, m, p), torch.float32, finite=True, domain=dom)
        )
        return OpSample(
            inputs=(self_t, batch1, batch2),
            module=TernaryPrimitive(torch.addbmm),
        )

    return _draw()


def _addmv_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.addmv(self, mat, vec)`: `(m,) + (m, n) @ (n,) -> (m,)`."""

    @st.composite
    def _draw(draw) -> OpSample:
        m = draw(st.integers(min_value=1, max_value=5))
        n = draw(st.integers(min_value=1, max_value=5))
        dom = Interval(-2.0, 2.0)
        self_t = draw(tensor_st((m,), torch.float32, finite=True, domain=dom))
        mat = draw(tensor_st((m, n), torch.float32, finite=True, domain=dom))
        vec = draw(tensor_st((n,), torch.float32, finite=True, domain=dom))
        return OpSample(
            inputs=(self_t, mat, vec),
            module=TernaryPrimitive(torch.addmv),
        )

    return _draw()


def _addr_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.addr(self, vec1, vec2)`: `(m, n) + outer(vec1, vec2)`."""

    @st.composite
    def _draw(draw) -> OpSample:
        m = draw(st.integers(min_value=1, max_value=5))
        n = draw(st.integers(min_value=1, max_value=5))
        dom = Interval(-2.0, 2.0)
        self_t = draw(tensor_st((m, n), torch.float32, finite=True, domain=dom))
        vec1 = draw(tensor_st((m,), torch.float32, finite=True, domain=dom))
        vec2 = draw(tensor_st((n,), torch.float32, finite=True, domain=dom))
        return OpSample(
            inputs=(self_t, vec1, vec2),
            module=TernaryPrimitive(torch.addr),
        )

    return _draw()


def _inner_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.inner(a, b)`: rank-2 inputs, matching trailing dim."""

    @st.composite
    def _draw(draw) -> OpSample:
        m = draw(st.integers(min_value=1, max_value=4))
        n = draw(st.integers(min_value=1, max_value=4))
        k = draw(st.integers(min_value=1, max_value=5))
        dom = Interval(-3.0, 3.0)
        a = draw(tensor_st((m, k), torch.float32, finite=True, domain=dom))
        b = draw(tensor_st((n, k), torch.float32, finite=True, domain=dom))
        return OpSample(inputs=(a, b), module=BinaryPrimitive(torch.inner))

    return _draw()


def _vdot_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.vdot(a, b)`: 1-D real inputs of equal length."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=6))
        dom = Interval(-3.0, 3.0)
        a = draw(tensor_st((n,), torch.float32, finite=True, domain=dom))
        b = draw(tensor_st((n,), torch.float32, finite=True, domain=dom))
        return OpSample(inputs=(a, b), module=BinaryPrimitive(torch.vdot))

    return _draw()


def _kron_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.kron(a, b)`: rank-2 inputs."""

    @st.composite
    def _draw(draw) -> OpSample:
        m = draw(st.integers(min_value=1, max_value=3))
        n = draw(st.integers(min_value=1, max_value=3))
        p = draw(st.integers(min_value=1, max_value=3))
        q = draw(st.integers(min_value=1, max_value=3))
        dom = Interval(-3.0, 3.0)
        a = draw(tensor_st((m, n), torch.float32, finite=True, domain=dom))
        b = draw(tensor_st((p, q), torch.float32, finite=True, domain=dom))
        return OpSample(inputs=(a, b), module=BinaryPrimitive(torch.kron))

    return _draw()


def _diag_1d_to_2d_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.diag(x)` with 1-D input: build a square diag matrix."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=5))
        x = draw(
            tensor_st(
                (n,),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.diag))

    return _draw()


def _diag_2d_to_1d_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.diag(x)` with square 2-D input: extract diagonal."""

    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=5))
        x = draw(
            tensor_st(
                (n, n),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.diag))

    return _draw()


def _diagflat_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.diagflat(x)`: flatten + diag."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        sizes = draw(
            st.lists(
                st.integers(min_value=1, max_value=3),
                min_size=rank,
                max_size=rank,
            )
        )
        x = draw(
            tensor_st(
                tuple(sizes),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.diagflat))

    return _draw()


def _diag_embed_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.diag_embed(x)` with default `(dim1=-2, dim2=-1)`."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=3))
        sizes = draw(
            st.lists(
                st.integers(min_value=1, max_value=4),
                min_size=rank,
                max_size=rank,
            )
        )
        x = draw(
            tensor_st(
                tuple(sizes),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(torch.diag_embed))

    return _draw()


def _tier_a2_linalg_specs() -> T.List[OpSpec]:
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            name="inner",
            aten_ops=("inner",),
            sample_st=_inner_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="vdot",
            aten_ops=("vdot",),
            sample_st=_vdot_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_compatible=True,
        ),
        # `kron`: emits a static `[m*p, n*q]` reshape, so axis 0 of the
        # input has to be statically known.
        OpSpec(
            name="kron",
            aten_ops=("kron",),
            sample_st=_kron_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_skip_reason=(
                "kron emits a `[m*p, n*q]` reshape; static shapes only."
            ),
        ),
        OpSpec(
            name="diag_1d_to_2d",
            aten_ops=("diag",),
            sample_st=_diag_1d_to_2d_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_compatible=True,
        ),
        OpSpec(
            name="diag_2d_to_1d",
            aten_ops=("diag",),
            sample_st=_diag_2d_to_1d_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_compatible=True,
        ),
        # `diagflat`: total flat size is baked into the reshape; the
        # eye(N, N) constant also needs static N.
        OpSpec(
            name="diagflat",
            aten_ops=("diagflat",),
            sample_st=_diagflat_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_skip_reason=(
                "diagflat bakes a static flat size into the reshape."
            ),
        ),
        OpSpec(
            name="diag_embed",
            aten_ops=("diag_embed",),
            sample_st=_diag_embed_sample_st(),
            tolerance=CLOSE,
            dynamic_axes_compatible=True,
        ),
    ]


def _recent_distance_matmul_specs() -> T.List[OpSpec]:
    """`pdist` / `renorm` + the `addbmm` / `addmv` / `addr` cluster."""
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            name="pdist",
            aten_ops=("pdist",),
            sample_st=_pdist_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="renorm",
            aten_ops=("renorm",),
            sample_st=_renorm_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="addbmm",
            aten_ops=("addbmm",),
            sample_st=_addbmm_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="addmv",
            aten_ops=("addmv",),
            sample_st=_addmv_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            name="addr",
            aten_ops=("addr",),
            sample_st=_addr_sample_st(),
            tolerance=CLOSE,
        ),
    ]


# Constructors (input-less in PyTorch, wrapped with a shape-coupled input)
# + advanced index + SDPA

SPECS = (
    *_specialty_specs(),
    *_prelu_glu_einsum_specs(),
    *_max_pool_dropout_specs(),
    *_distance_specs(),
    *_no_tract_change_specs(),
    *_recent_distance_matmul_specs(),
    *_tier_a2_linalg_specs(),
)
