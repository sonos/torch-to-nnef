"""Specs for operators torch_to_nnef does not translate.

Every other module here covers something we ship. This one covers the
opposite: operators with no emitter at all, so that the ONNX sweep can
measure them.

Why that matters. Specs previously existed only where t2n succeeds, so
the measured ONNX population was a strict subset of our own supported
set: an operator ONNX handles and we do not could never be graded, only
inherited as an unverified claim from the retired torch listing. The
comparison was structurally blind exactly where we are weakest.

Each spec declares an `NnefGap`, which the tract driver *asserts* rather
than trusts (see `tests/proptest/nnef_gap.py`), so a gap that gets closed
turns into a failing test rather than a stale page.

The first batch is the twenty operators the support page's `Gap vs ONNX`
filter listed: rows we do not translate, that the retired listing claimed
ONNX did, and that nothing had ever measured. They fall into three
families, and the family is usually the reason NNEF cannot express them:

- **data-dependent output shape** (`nonzero`, `masked_select`,
  `unique_dim`): the output rank is known but its extent depends on the
  values, which a static NNEF graph cannot declare.
- **RNG** (`rand`, `randn`, `bernoulli`, ...): NNEF has no random
  primitive, and tract has no RNG state to seed.
- **no NNEF primitive** (`linalg_det`, `as_strided`, ...): expressible in
  principle, but only by decomposing into something we have not written.
"""

import typing as T

import torch
import torch.nn.functional as F
from hypothesis import strategies as st

from ..inputs import Interval, tensor_st
from ..shapes import shape_st
from ._common import NnefGap, NnefGapStage, OpSample, OpSpec


def _gap(reason: str) -> NnefGap:
    """The plain case: no emitter is registered for the operator."""
    return NnefGap(stage=NnefGapStage.NO_EMITTER, reason=reason)


def _early_gap(reason: str) -> NnefGap:
    """A gap the pipeline hits *before* the emitter lookup.

    An op whose inputs are all constant is constant-folded by running it
    (`torch_graph/ir_op.py`), so the registry is never consulted. For the
    RNG factories below that path raises, which means the operator would
    still not export even if someone registered an emitter tomorrow.
    Worth recording as a distinct stage: it says where the work would
    actually have to start.
    """
    return NnefGap(stage=NnefGapStage.EXPORT_ERROR, reason=reason)


class _Fn(torch.nn.Module):
    """Call a plain function on the drawn inputs."""

    def __init__(self, fn: T.Callable[..., torch.Tensor], name: str):
        super().__init__()
        self.fn = fn
        self.name = name

    def extra_repr(self) -> str:
        return f"fn={self.name}"

    def forward(self, *args):
        return self.fn(*args)


# -- data-dependent output shape ---------------------------------------


@st.composite
def _nonzero_st(draw, as_tuple: bool) -> OpSample:
    """A tensor with a mix of zeros and non-zeros, so the count varies."""
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    # Integers over a tiny range: a float draw is almost never exactly
    # zero, which would make every example produce a full-length output
    # and hide the data-dependence this op is about.
    x = draw(tensor_st(shape, torch.int64, domain=Interval(0, 2)))
    if as_tuple:
        fn = lambda t: torch.nonzero(t, as_tuple=True)[0]  # noqa: E731
        return OpSample(inputs=(x,), module=_Fn(fn, "nonzero_numpy"))
    return OpSample(inputs=(x,), module=_Fn(torch.nonzero, "nonzero"))


@st.composite
def _masked_select_st(draw) -> OpSample:
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    x = draw(tensor_st(shape, torch.float32))
    mask = draw(tensor_st(shape, torch.bool))
    return OpSample(
        inputs=(x, mask),
        module=_Fn(torch.masked_select, "masked_select"),
    )


@st.composite
def _unique_dim_st(draw) -> OpSample:
    rows = draw(st.integers(min_value=1, max_value=6))
    cols = draw(st.integers(min_value=1, max_value=4))
    # Small integer range so duplicate rows actually occur and the
    # output really is shorter than the input.
    x = draw(tensor_st((rows, cols), torch.int64, domain=Interval(0, 2)))
    fn = lambda t: torch.unique(t, dim=0)  # noqa: E731
    return OpSample(inputs=(x,), module=_Fn(fn, "unique_dim"))


# -- RNG ---------------------------------------------------------------


@st.composite
def _shape_of_input_rng_st(draw, fn, name: str) -> OpSample:
    """`fn(x.shape)`: an RNG factory sized from a real graph input."""
    shape = draw(shape_st(min_rank=1, max_rank=4))
    x = draw(tensor_st(shape, torch.float32))
    return OpSample(inputs=(x,), module=_Fn(fn, name))


@st.composite
def _like_rng_st(draw, fn, name: str) -> OpSample:
    """`fn(x)`: an RNG factory taking its shape and dtype from x."""
    shape = draw(shape_st(min_rank=0, max_rank=4))
    x = draw(tensor_st(shape, torch.float32))
    return OpSample(inputs=(x,), module=_Fn(fn, name))


@st.composite
def _bernoulli_st(draw) -> OpSample:
    shape = draw(shape_st(min_rank=0, max_rank=3))
    probs = draw(tensor_st(shape, torch.float32, domain=Interval(0.0, 1.0)))
    return OpSample(inputs=(probs,), module=_Fn(torch.bernoulli, "bernoulli"))


@st.composite
def _multinomial_st(draw) -> OpSample:
    categories = draw(st.integers(min_value=2, max_value=8))
    rows = draw(st.integers(min_value=1, max_value=4))
    num_samples = draw(st.integers(min_value=1, max_value=categories))
    # Strictly positive weights: an all-zero row is a torch error, not an
    # export question.
    weights = draw(
        tensor_st((rows, categories), torch.float32, domain=Interval(0.1, 5.0))
    )
    fn = lambda w: torch.multinomial(w, num_samples)  # noqa: E731
    return OpSample(inputs=(weights,), module=_Fn(fn, "multinomial"))


@st.composite
def _normal_st(draw) -> OpSample:
    shape = draw(shape_st(min_rank=1, max_rank=3))
    mean = draw(tensor_st(shape, torch.float32, domain=Interval(-5.0, 5.0)))
    std = draw(tensor_st(shape, torch.float32, domain=Interval(0.1, 5.0)))
    return OpSample(inputs=(mean, std), module=_Fn(torch.normal, "normal"))


@st.composite
def _empty_st(draw) -> OpSample:
    shape = draw(shape_st(min_rank=1, max_rank=4))
    x = draw(tensor_st(shape, torch.float32))
    fn = lambda t: torch.empty(t.shape, dtype=t.dtype)  # noqa: E731
    return OpSample(inputs=(x,), module=_Fn(fn, "empty"))


# -- no NNEF primitive -------------------------------------------------


@st.composite
def _as_strided_st(draw) -> OpSample:
    rows = draw(st.integers(min_value=1, max_value=4))
    cols = draw(st.integers(min_value=1, max_value=4))
    # A contiguous (rows, cols) view of a flat buffer: the simplest
    # legal restriding, and the one an emitter would have to handle
    # first.
    x = draw(tensor_st((rows * cols,), torch.float32))
    fn = lambda t: torch.as_strided(  # noqa: E731
        t, (rows, cols), (cols, 1)
    )
    return OpSample(inputs=(x,), module=_Fn(fn, "as_strided"))


@st.composite
def _masked_scatter_st(draw) -> OpSample:
    shape = draw(shape_st(min_rank=1, max_rank=3, max_dim=6))
    numel = 1
    for dim in shape:
        numel *= dim
    x = draw(tensor_st(shape, torch.float32))
    mask = draw(tensor_st(shape, torch.bool))
    # Source must hold at least `mask.sum()` elements; sizing it to the
    # full numel keeps every draw legal without inspecting the mask.
    source = draw(tensor_st((numel,), torch.float32))
    return OpSample(
        inputs=(x, mask, source),
        module=_Fn(lambda t, m, s: t.masked_scatter(m, s), "masked_scatter"),
    )


@st.composite
def _square_matrix_st(draw, fn, name: str, positive_definite: bool):
    """A batched square matrix, optionally made positive definite."""
    size = draw(st.integers(min_value=1, max_value=4))
    batch = draw(st.integers(min_value=0, max_value=2))
    shape = ((batch,) if batch else ()) + (size, size)
    x = draw(tensor_st(shape, torch.float32, domain=Interval(-3.0, 3.0)))
    if positive_definite:
        # `logdet` of a matrix with negative determinant is NaN, which
        # says nothing about export support, so keep the determinant
        # positive: A @ A^T + n*I is symmetric positive definite.
        x = x @ x.transpose(-1, -2) + torch.eye(size) * size
    return OpSample(inputs=(x,), module=_Fn(fn, name))


@st.composite
def _rrelu_st(draw) -> OpSample:
    shape = draw(shape_st(min_rank=0, max_rank=4))
    x = draw(tensor_st(shape, torch.float32))
    # `training=False` (the default) makes rrelu deterministic: it uses
    # the midpoint of the slope range instead of sampling it. So this is
    # the one op in the RNG family that is safe to compare numerically.
    return OpSample(inputs=(x,), module=_Fn(F.rrelu, "rrelu"))


_DATA_DEPENDENT = (
    "output extent depends on the input values, which a static NNEF "
    "graph cannot declare"
)
_NO_RNG = "NNEF has no random primitive and tract has no RNG state"
#: The `fn(x.shape)` factories take only constant inputs, so t2n tries to
#: constant-fold them before any emitter lookup. Folding an RNG op would
#: bake one draw into the graph, so this path has to reject them however
#: the argument marshalling is fixed.
_FOLDED_RNG = f"{_NO_RNG}; constant-folded before the emitter lookup"


SPECS: T.Tuple[OpSpec, ...] = (
    # -- data-dependent output shape --
    OpSpec(
        name="gap-nonzero",
        sample_st=_nonzero_st(as_tuple=False),
        aten_ops=("nonzero",),
        nnef_gap=_gap(_DATA_DEPENDENT),
    ),
    OpSpec(
        name="gap-nonzero_numpy",
        sample_st=_nonzero_st(as_tuple=True),
        aten_ops=("nonzero_numpy",),
        nnef_gap=_gap(
            f"`nonzero(as_tuple=True)`: {_DATA_DEPENDENT}, once per axis"
        ),
    ),
    OpSpec(
        name="gap-masked_select",
        sample_st=_masked_select_st(),
        aten_ops=("masked_select",),
        nnef_gap=_gap(_DATA_DEPENDENT),
    ),
    OpSpec(
        name="gap-unique_dim",
        sample_st=_unique_dim_st(),
        aten_ops=("unique_dim",),
        nnef_gap=_gap(_DATA_DEPENDENT),
    ),
    # -- RNG --
    OpSpec(
        name="gap-rand",
        sample_st=_shape_of_input_rng_st(lambda x: torch.rand(x.shape), "rand"),
        aten_ops=("rand",),
        nnef_gap=_early_gap(_FOLDED_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-randn",
        sample_st=_shape_of_input_rng_st(
            lambda x: torch.randn(x.shape), "randn"
        ),
        aten_ops=("randn",),
        nnef_gap=_early_gap(_FOLDED_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-randint",
        sample_st=_shape_of_input_rng_st(
            lambda x: torch.randint(0, 10, x.shape), "randint"
        ),
        aten_ops=("randint",),
        nnef_gap=_early_gap(_FOLDED_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-rand_like",
        sample_st=_like_rng_st(torch.rand_like, "rand_like"),
        aten_ops=("rand_like",),
        nnef_gap=_gap(_NO_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-randn_like",
        sample_st=_like_rng_st(torch.randn_like, "randn_like"),
        aten_ops=("randn_like",),
        nnef_gap=_gap(_NO_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-randint_like",
        sample_st=_like_rng_st(
            lambda x: torch.randint_like(x, 0, 10), "randint_like"
        ),
        aten_ops=("randint_like",),
        nnef_gap=_gap(_NO_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-bernoulli",
        sample_st=_bernoulli_st(),
        aten_ops=("bernoulli",),
        nnef_gap=_gap(_NO_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-multinomial",
        sample_st=_multinomial_st(),
        aten_ops=("multinomial",),
        nnef_gap=_early_gap(
            f"{_NO_RNG}; evaluated during translation, where torch "
            "rejects the zero-filled placeholder weights"
        ),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-normal",
        sample_st=_normal_st(),
        aten_ops=("normal",),
        nnef_gap=_gap(_NO_RNG),
        nondeterministic=True,
    ),
    OpSpec(
        name="gap-empty",
        sample_st=_empty_st(),
        aten_ops=("empty",),
        nnef_gap=_gap(
            "allocates without initializing, so there is nothing for a "
            "declarative graph to describe"
        ),
        nondeterministic=True,
    ),
    # -- no NNEF primitive --
    OpSpec(
        name="gap-as_strided",
        sample_st=_as_strided_st(),
        aten_ops=("as_strided",),
        nnef_gap=_gap(
            "arbitrary size/stride views have no NNEF equivalent; only "
            "the contiguous cases reduce to a reshape"
        ),
    ),
    OpSpec(
        name="gap-masked_scatter",
        sample_st=_masked_scatter_st(),
        aten_ops=("masked_scatter",),
        nnef_gap=_gap(
            "consumes the source in mask order, so the read index is "
            "itself a cumulative sum of the mask"
        ),
    ),
    OpSpec(
        name="gap-linalg_det",
        sample_st=_square_matrix_st(
            torch.linalg.det, "linalg_det", positive_definite=False
        ),
        aten_ops=("linalg_det",),
        nnef_gap=_gap("no NNEF decomposition (would need LU)"),
    ),
    OpSpec(
        name="gap-logdet",
        sample_st=_square_matrix_st(
            torch.logdet, "logdet", positive_definite=True
        ),
        aten_ops=("logdet",),
        nnef_gap=_gap("same LU gap as `linalg_det`"),
    ),
    OpSpec(
        name="gap-rrelu",
        sample_st=_rrelu_st(),
        aten_ops=("rrelu",),
        nnef_gap=_gap(
            "no emitter, though in eval mode it is a plain leaky_relu at "
            "the midpoint slope and could reuse that one"
        ),
    ),
)
