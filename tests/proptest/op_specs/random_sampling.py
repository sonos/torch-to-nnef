"""Specs for the random-number operators.

All untranslated today, so every spec carries `nnef_gap`; implementing
one means deleting that field, not moving the spec.

One reason covers nearly all of them: NNEF has no random primitive and
tract has no RNG state, so there is nothing to lower these onto. That
makes the family unusual: unlike the linalg gaps, closing these is a
format question rather than a kernel question.

Every spec here sets `nondeterministic=True`. Export and runtime stay
measured; only the numerics axis is skipped, because comparing two
independent draws would report the definition of the operator rather
than anything about the exporter.

The `fn(x.shape)` factories additionally fail *before* the emitter
lookup: their inputs are all constant, so the constant-folding pass runs
them first and dies. Folding an RNG op would bake a single draw into the
graph, so that path has to keep rejecting them whatever else changes.
"""

import typing as T

import torch
from hypothesis import strategies as st

from ..inputs import Interval, tensor_st
from ..shapes import shape_st
from ._common import NnefGapStage, OpSample, OpSpec
from ._gap_common import DEFAULT_DOMAIN, GapModule, bounded, gap_spec

_NO_RNG = "NNEF has no random primitive and tract has no RNG state"
_FOLDED = f"{_NO_RNG}; constant-folded before the emitter lookup"


@st.composite
def _from_shape_st(draw, fn, name: str):
    """`fn(x.shape)`: sized from a real graph input."""
    shape = draw(shape_st(min_rank=1, max_rank=4))
    x = draw(tensor_st(shape, torch.float32, domain=DEFAULT_DOMAIN))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _like_st(draw, fn, name: str, domain=None):
    """`fn(x)`: shape and dtype taken from the input tensor."""
    shape = draw(shape_st(min_rank=0, max_rank=4))
    x = draw(tensor_st(shape, torch.float32, domain=bounded(domain)))
    return OpSample(inputs=(x,), module=GapModule(fn, name))


@st.composite
def _pair_st(draw, fn, name: str, first, second):
    """`fn(x, y)` with a domain per argument (mean/std, count/prob)."""
    shape = draw(shape_st(min_rank=1, max_rank=3))
    x = draw(tensor_st(shape, torch.float32, domain=first))
    y = draw(tensor_st(shape, torch.float32, domain=second))
    return OpSample(inputs=(x, y), module=GapModule(fn, name))


def _inplace(op: str, *args):
    """`x.<op>_(*args)` on a clone, so the input tensor is untouched."""

    def _fn(x):
        return getattr(x.clone(), f"{op}_")(*args)

    return _fn


SPECS: T.Tuple[OpSpec, ...] = (
    # -- sized from a shape: folded before the lookup --
    gap_spec(
        "rand",
        _from_shape_st(lambda x: torch.rand(x.shape), "rand"),
        _FOLDED,
        stage=NnefGapStage.EXPORT_ERROR,
        nondeterministic=True,
    ),
    gap_spec(
        "randn",
        _from_shape_st(lambda x: torch.randn(x.shape), "randn"),
        _FOLDED,
        stage=NnefGapStage.EXPORT_ERROR,
        nondeterministic=True,
    ),
    gap_spec(
        "randint",
        _from_shape_st(lambda x: torch.randint(0, 10, x.shape), "randint"),
        _FOLDED,
        stage=NnefGapStage.EXPORT_ERROR,
        nondeterministic=True,
    ),
    gap_spec(
        "randperm",
        _from_shape_st(
            lambda x: torch.randperm(5) + (x.sum() * 0).long(), "randperm"
        ),
        f"{_FOLDED}; the length is a constant, so the whole call folds",
        stage=NnefGapStage.EXPORT_ERROR,
        nondeterministic=True,
    ),
    # -- shaped from a tensor --
    gap_spec(
        "rand_like",
        _like_st(torch.rand_like, "rand_like"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "randn_like",
        _like_st(torch.randn_like, "randn_like"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "randint_like",
        _like_st(lambda x: torch.randint_like(x, 0, 10), "randint_like"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "bernoulli",
        _like_st(torch.bernoulli, "bernoulli", domain=Interval(0.0, 1.0)),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "poisson",
        _like_st(torch.poisson, "poisson", domain=Interval(0.1, 8.0)),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "gamma",
        _like_st(torch._standard_gamma, "gamma", domain=Interval(0.5, 8.0)),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "normal",
        _pair_st(
            torch.normal, "normal", Interval(-5.0, 5.0), Interval(0.1, 5.0)
        ),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "binomial",
        _pair_st(
            torch.binomial, "binomial", Interval(1.0, 10.0), Interval(0.0, 1.0)
        ),
        _NO_RNG,
        nondeterministic=True,
    ),
    # -- in-place samplers: the page merges these onto the bare row --
    gap_spec(
        "cauchy",
        _like_st(_inplace("cauchy"), "cauchy"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "exponential",
        _like_st(_inplace("exponential"), "exponential"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "geometric",
        _like_st(_inplace("geometric", 0.5), "geometric"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "log_normal",
        _like_st(_inplace("log_normal"), "log_normal"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "random",
        _like_st(_inplace("random", 0, 10), "random"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "uniform",
        _like_st(_inplace("uniform"), "uniform"),
        _NO_RNG,
        nondeterministic=True,
    ),
    gap_spec(
        "multinomial",
        # Strictly positive weights: an all-zero row is a torch error,
        # not an export question.
        _like_st(
            lambda x: torch.multinomial(x.reshape(1, -1), 1),
            "multinomial",
            domain=Interval(0.1, 5.0),
        ),
        f"{_NO_RNG}; evaluated during translation, where torch rejects "
        "the zero-filled placeholder weights",
        stage=NnefGapStage.EXPORT_ERROR,
        nondeterministic=True,
    ),
    # -- randomized activations --
    gap_spec(
        "rrelu_with_noise",
        _like_st(
            lambda x: torch._C._nn.rrelu_with_noise(
                x, torch.zeros_like(x), 0.1, 0.3, True
            ),
            "rrelu_with_noise",
        ),
        f"{_NO_RNG}; the training-mode form of `rrelu`, which samples "
        "its slope",
        nondeterministic=True,
    ),
    gap_spec(
        "rrelu",
        _like_st(torch.nn.functional.rrelu, "rrelu"),
        "no emitter, though in eval mode it is a plain leaky_relu at "
        "the midpoint slope and could reuse that one",
    ),
)
