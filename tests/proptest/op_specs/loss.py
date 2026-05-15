"""Spec builders for the loss-family aten ops.

Each loss is exercised across the three reduction modes torch supports
(`none` / `mean` / `sum`); samples are kept small and dtype-free
(float32) since the emitter's per-op behaviour is independent of input
shape beyond the rank assumptions documented in the corresponding
`torch_to_nnef/op/aten/loss.py` handler.
"""

import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import BinaryPrimitive
from ..inputs import Interval, tensor_st
from ._common import OpSample, OpSpec

_REDUCTIONS = ("none", "mean", "sum")


def _pointwise_loss_sample_st(
    callable_factory: T.Callable[[str], T.Callable[..., torch.Tensor]],
) -> st.SearchStrategy[OpSample]:
    """Shared (input, target, reduction) draw for the elementwise losses.

    `callable_factory(reduction)` returns the actual `F.<loss>(...)` to
    invoke, capturing any scalar params (`delta`, `beta`) closure-style.
    """

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
        reduction = draw(st.sampled_from(_REDUCTIONS))
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
        return OpSample(
            inputs=(a, b),
            module=BinaryPrimitive(callable_factory(reduction)),
        )

    return _draw()


def _mse_sample_st() -> st.SearchStrategy[OpSample]:
    return _pointwise_loss_sample_st(
        lambda r: lambda x, y, _r=r: F.mse_loss(x, y, reduction=_r)
    )


def _l1_sample_st() -> st.SearchStrategy[OpSample]:
    return _pointwise_loss_sample_st(
        lambda r: lambda x, y, _r=r: F.l1_loss(x, y, reduction=_r)
    )


def _huber_sample_st() -> st.SearchStrategy[OpSample]:
    # Default `delta=1.0`; mixed -3..3 range exercises both branches.
    return _pointwise_loss_sample_st(
        lambda r: lambda x, y, _r=r: F.huber_loss(x, y, reduction=_r)
    )


def _smooth_l1_sample_st() -> st.SearchStrategy[OpSample]:
    # Default `beta=1.0`; same range covers both branches.
    return _pointwise_loss_sample_st(
        lambda r: lambda x, y, _r=r: F.smooth_l1_loss(x, y, reduction=_r)
    )


def _margin_ranking_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.margin_ranking_loss(input1, input2, target, margin, reduction)`.

    target draws from {-1, +1}; margin from [0, 1].
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=2))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        reduction = draw(st.sampled_from(_REDUCTIONS))
        margin = draw(st.sampled_from([0.0, 0.25, 0.5]))
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
        # Target as ±1 — generate as ints {0, 1} mapped to {-1, +1}.
        sign = draw(
            tensor_st(shape, torch.int64, finite=True, domain=Interval(0, 1))
        )
        t = (2 * sign - 1).to(torch.float32)

        class _MR(nn.Module):
            def forward(self, x1, x2, tgt):
                return F.margin_ranking_loss(
                    x1, x2, tgt, margin=margin, reduction=reduction
                )

        return OpSample(inputs=(a, b, t), module=_MR())

    return _draw()


def _soft_margin_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.soft_margin_loss(input, target, reduction)`.

    target draws from {-1, +1}.
    """

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
        reduction = draw(st.sampled_from(_REDUCTIONS))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        sign = draw(
            tensor_st(shape, torch.int64, finite=True, domain=Interval(0, 1))
        )
        t = (2 * sign - 1).to(torch.float32)
        return OpSample(
            inputs=(x, t),
            module=BinaryPrimitive(
                lambda x, y, _r=reduction: F.soft_margin_loss(
                    x, y, reduction=_r
                )
            ),
        )

    return _draw()


def _cosine_embedding_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.cosine_embedding_loss(x1, x2, target, margin, reduction)`.

    Rank-2 inputs `(B, D)`; target shape `(B,)` with ±1 values.
    Inputs stay away from the origin so the cosine-similarity epsilon
    clamp never fires in the comparison.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        b = draw(st.integers(min_value=1, max_value=4))
        d = draw(st.integers(min_value=2, max_value=6))
        reduction = draw(st.sampled_from(_REDUCTIONS))
        margin = draw(st.sampled_from([0.0, 0.25, 0.5]))
        x1 = draw(
            tensor_st(
                (b, d), torch.float32, finite=True, domain=Interval(0.3, 3.0)
            )
        )
        x2 = draw(
            tensor_st(
                (b, d), torch.float32, finite=True, domain=Interval(0.3, 3.0)
            )
        )
        sign = draw(
            tensor_st((b,), torch.int64, finite=True, domain=Interval(0, 1))
        )
        t = (2 * sign - 1).to(torch.float32)

        class _CE(nn.Module):
            def forward(self, x1, x2, tgt):
                return F.cosine_embedding_loss(
                    x1, x2, tgt, margin=margin, reduction=reduction
                )

        return OpSample(inputs=(x1, x2, t), module=_CE())

    return _draw()


def _bce_logits_sample_st() -> st.SearchStrategy[OpSample]:
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
        reduction = draw(st.sampled_from(_REDUCTIONS))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        t = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(0.0, 1.0)
            )
        )
        return OpSample(
            inputs=(x, t),
            module=BinaryPrimitive(
                lambda x, y, _r=reduction: F.binary_cross_entropy_with_logits(
                    x, y, reduction=_r
                )
            ),
        )

    return _draw()


def _bce_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.binary_cross_entropy(probs, target, reduction)`.

    Both inputs are kept in (eps, 1-eps) so `log(x)` / `log(1 - x)`
    stay finite; the fragment doesn't clamp.
    """

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
        reduction = draw(st.sampled_from(_REDUCTIONS))
        eps = 1e-3
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(eps, 1.0 - eps),
            )
        )
        t = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(0.0, 1.0)
            )
        )
        return OpSample(
            inputs=(x, t),
            module=BinaryPrimitive(
                lambda x, y, _r=reduction: F.binary_cross_entropy(
                    x, y, reduction=_r
                )
            ),
        )

    return _draw()


def _kl_div_sample_st(log_target: bool) -> st.SearchStrategy[OpSample]:
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
        # kl_div uses 'sum' / 'mean' / 'none' (batchmean is lowered upstream).
        reduction = draw(st.sampled_from(_REDUCTIONS))
        # `input` is log-probabilities; sample a random tensor and take
        # log_softmax along the last axis so it stays in the valid log-
        # probability range without crowding tract's checker.
        raw_x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        log_input = F.log_softmax(raw_x, dim=-1)
        raw_t = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        if log_target:
            target = F.log_softmax(raw_t, dim=-1)
        else:
            target = F.softmax(raw_t, dim=-1)
        return OpSample(
            inputs=(log_input, target),
            module=BinaryPrimitive(
                lambda x, y, _r=reduction, _lt=log_target: F.kl_div(
                    x, y, reduction=_r, log_target=_lt
                )
            ),
        )

    return _draw()


def _nll_loss_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=2, max_value=6))
        spatial_rank = draw(st.integers(min_value=0, max_value=2))
        spatial = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=spatial_rank,
                    max_size=spatial_rank,
                )
            )
        )
        reduction = draw(st.sampled_from(_REDUCTIONS))
        x = draw(
            tensor_st(
                (n, c) + spatial,
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        # Log-probabilities along the class axis.
        log_input = F.log_softmax(x, dim=1)
        target = draw(
            tensor_st(
                (n,) + spatial,
                torch.int64,
                finite=True,
                domain=Interval(0, c - 1),
            )
        )

        class _Nll(nn.Module):
            def forward(self, log_probs, tgt):
                return F.nll_loss(log_probs, tgt, reduction=reduction)

        return OpSample(inputs=(log_input, target), module=_Nll())

    return _draw()


def _cross_entropy_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        n = draw(st.integers(min_value=1, max_value=4))
        c = draw(st.integers(min_value=2, max_value=6))
        spatial_rank = draw(st.integers(min_value=0, max_value=2))
        spatial = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=4),
                    min_size=spatial_rank,
                    max_size=spatial_rank,
                )
            )
        )
        reduction = draw(st.sampled_from(_REDUCTIONS))
        x = draw(
            tensor_st(
                (n, c) + spatial,
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        target = draw(
            tensor_st(
                (n,) + spatial,
                torch.int64,
                finite=True,
                domain=Interval(0, c - 1),
            )
        )

        class _CE(nn.Module):
            def forward(self, inp, tgt):
                return F.cross_entropy(inp, tgt, reduction=reduction)

        return OpSample(inputs=(x, target), module=_CE())

    return _draw()


def _loss_specs() -> T.List[OpSpec]:
    APPROX = TractCheckTolerance.APPROXIMATE
    return [
        OpSpec(
            name="mse_loss",
            sample_st=_mse_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="l1_loss",
            sample_st=_l1_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="huber_loss",
            sample_st=_huber_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="smooth_l1_loss",
            sample_st=_smooth_l1_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="binary_cross_entropy",
            sample_st=_bce_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="binary_cross_entropy_with_logits",
            sample_st=_bce_logits_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="kl_div",
            sample_st=_kl_div_sample_st(log_target=False),
            tolerance=APPROX,
        ),
        OpSpec(
            name="kl_div_log_target",
            sample_st=_kl_div_sample_st(log_target=True),
            tolerance=APPROX,
        ),
        OpSpec(
            name="nll_loss",
            sample_st=_nll_loss_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="cross_entropy_loss",
            sample_st=_cross_entropy_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="margin_ranking_loss",
            sample_st=_margin_ranking_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="soft_margin_loss",
            sample_st=_soft_margin_sample_st(),
            tolerance=APPROX,
        ),
        OpSpec(
            name="cosine_embedding_loss",
            sample_st=_cosine_embedding_sample_st(),
            tolerance=APPROX,
        ),
    ]


SPECS = tuple(_loss_specs())
