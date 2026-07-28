"""Gap specs for neural-network layers, losses and spectral transforms.

Smaller and less uniform than the other modules. The losses are the
interesting ones for us: they are all expressible, and they are the kind
of operator a training-adjacent export would hit first, so their reasons
say what the missing lowering looks like rather than just "no emitter".

The `fft_*` entries are the n-dimensional and half-complex members of a
family we already translate in part, so the gap is narrower than it
looks: the transform exists, the axis handling does not.
"""

import typing as T

import torch
import torch.nn.functional as F
from hypothesis import strategies as st

from ...inputs import Interval, tensor_st
from .._common import NnefGapStage, OpSample, OpSpec
from ._helpers import GapModule, gap_spec

_FFT_AXES = (
    "the 1-D transforms are translated; these take an axis list, and no "
    "emitter maps that onto repeated applications"
)


@st.composite
def _image_st(draw, fn, name: str, rank: int, side: int = 4):
    """A `(1, 1, side, ...)` feature map with `rank` spatial axes."""
    shape = (1, 1) + (side,) * rank
    x = draw(tensor_st(shape, torch.float32, domain=Interval(-5.0, 5.0)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _spectral_st(draw, fn, name: str):
    """A small square signal, since these are the n-D transforms."""
    side = draw(st.sampled_from([2, 4]))
    x = draw(tensor_st((side, side), torch.float32, domain=Interval(-5.0, 5.0)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _class_loss_st(draw, fn, name: str, target_rank: int):
    """Logits `(n, c)` with either per-sample or per-class targets."""
    samples = draw(st.integers(min_value=1, max_value=4))
    classes = draw(st.integers(min_value=2, max_value=5))
    logits = draw(
        tensor_st((samples, classes), torch.float32, domain=Interval(-5.0, 5.0))
    )
    shape = (samples,) if target_rank == 1 else (samples, classes)
    target = draw(
        tensor_st(shape, torch.int64, domain=Interval(0, classes - 1))
    )
    return OpSample(inputs=(logits, target), module=GapModule(fn, name))


@st.composite
def _poisson_nll_st(draw):
    samples = draw(st.integers(min_value=1, max_value=4))
    feats = draw(st.integers(min_value=1, max_value=4))
    pred = draw(
        tensor_st((samples, feats), torch.float32, domain=Interval(-3.0, 3.0))
    )
    target = draw(
        tensor_st((samples, feats), torch.float32, domain=Interval(0.0, 5.0))
    )
    return OpSample(
        inputs=(pred, target),
        module=GapModule(F.poisson_nll_loss, "poisson_nll_loss"),
    )


@st.composite
def _ctc_st(draw):
    """Log-probs `(T, N, C)` with fixed input/target lengths."""
    time = draw(st.integers(min_value=4, max_value=8))
    classes = draw(st.integers(min_value=3, max_value=5))
    target_len = draw(st.integers(min_value=1, max_value=3))
    logits = draw(
        tensor_st((time, 2, classes), torch.float32, domain=Interval(-3.0, 3.0))
    )
    targets = draw(
        tensor_st((2, target_len), torch.int64, domain=Interval(1, classes - 1))
    )

    def _fn(lp, tg):
        return F.ctc_loss(
            lp.log_softmax(-1),
            tg,
            torch.full((2,), time, dtype=torch.long),
            torch.full((2,), target_len, dtype=torch.long),
        )

    return OpSample(inputs=(logits, targets), module=GapModule(_fn, "ctc_loss"))


def _unpool(rank: int):
    """`max_unpool` fed by the matching `max_pool`'s indices.

    `output_size` is passed explicitly: without it torch's shape check
    compares a traced tensor in a Python `if` and refuses to trace.
    """
    pool = F.max_pool2d if rank == 2 else F.max_pool3d
    unpool = F.max_unpool2d if rank == 2 else F.max_unpool3d
    size = [4] * rank

    def _fn(x):
        return unpool(*pool(x, 2, return_indices=True), 2, output_size=size)

    return _fn


SPECS: T.Tuple[OpSpec, ...] = (
    # -- pooling --
    gap_spec(
        "max_unpool2d",
        _image_st(_unpool(2), "max_unpool2d", rank=2),
        "the inverse of a pooling: a scatter into a larger buffer at "
        "runtime-known indices, which no emitter builds",
    ),
    gap_spec(
        "max_unpool3d",
        _image_st(_unpool(3), "max_unpool3d", rank=3),
        "the 3-D form of the same scatter as `max_unpool2d`",
    ),
    gap_spec(
        "fractional_max_pool2d",
        _image_st(
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
        _image_st(
            lambda x: F.fractional_max_pool3d(x, 2, output_size=(2, 2, 2)),
            "fractional_max_pool3d",
            rank=3,
            side=5,
        ),
        "the 3-D form of `fractional_max_pool2d`, same RNG problem",
        stage=NnefGapStage.EXPORT_ERROR,
        nondeterministic=True,
    ),
    # -- losses --
    gap_spec(
        "multi_margin_loss",
        _class_loss_st(F.multi_margin_loss, "multi_margin_loss", 1),
        "expressible as a gather of the target score plus a clamped "
        "difference and a mean, but no emitter composes it",
    ),
    gap_spec(
        "multilabel_margin_loss",
        _class_loss_st(F.multilabel_margin_loss, "multilabel_margin_loss", 2),
        "same composition as `multi_margin_loss`, over a target list "
        "terminated by -1",
    ),
    gap_spec(
        "poisson_nll_loss",
        _poisson_nll_st(),
        "expressible as `exp(x) - t*x` plus an optional Stirling term, "
        "but no emitter composes it",
    ),
    gap_spec(
        "ctc_loss",
        _ctc_st(),
        "a dynamic-programming recurrence over time, so a lowering "
        "means a scan rather than a fused op",
    ),
    # -- spectral --
    gap_spec(
        "fft_rfftn",
        _spectral_st(lambda x: torch.fft.rfftn(x).real, "fft_rfftn"),
        _FFT_AXES,
    ),
    gap_spec(
        "fft_irfftn",
        _spectral_st(
            lambda x: torch.fft.irfftn(x.to(torch.complex64)), "fft_irfftn"
        ),
        _FFT_AXES,
    ),
    gap_spec(
        "fft_ihfft2",
        _spectral_st(lambda x: torch.fft.ihfft2(x).real, "fft_ihfft2"),
        _FFT_AXES,
    ),
    gap_spec(
        "fft_ihfftn",
        _spectral_st(lambda x: torch.fft.ihfftn(x).real, "fft_ihfftn"),
        _FFT_AXES,
    ),
)
