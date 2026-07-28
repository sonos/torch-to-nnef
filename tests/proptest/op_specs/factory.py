"""Spec builders for the factory op group."""

import typing as T

import torch
import torch.nn.functional as F
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
from ._gap_common import (
    REASON_LAYOUT,
    gap_spec,
    shape_only_st,
)


class _ZerosFromShapeOf(torch.nn.Module):
    """`torch.zeros(*x.shape, dtype=x.dtype)`: derives shape from input."""

    def forward(self, x):
        return torch.zeros(x.shape, dtype=x.dtype)


class _OnesFromShapeOf(torch.nn.Module):
    """`torch.ones(*x.shape, dtype=x.dtype)`."""

    def forward(self, x):
        return torch.ones(x.shape, dtype=x.dtype)


class _FullFromShapeOf(torch.nn.Module):
    """`torch.full(x.shape, fill_value, dtype=x.dtype)`: swept fills."""

    def __init__(self, fill_value: float):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        return torch.full(x.shape, self.fill_value, dtype=x.dtype)


class _ArangeFromInput(torch.nn.Module):
    """`torch.arange(start, end, step)`: start/end/step baked at init.

    The input is ignored at runtime, but kept so the export pipeline has
    a real graph input. We attach a no-op dependency via `+ x.sum() * 0`
    so the graph extractor sees the tensor.
    """

    def __init__(self, start: int, end: int, step: int):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x):
        return torch.arange(self.start, self.end, self.step) + (x.sum() * 0)


class _ScalarTensorOfDtypeOf(torch.nn.Module):
    """`torch.scalar_tensor(value, dtype=x.dtype)`: 0-d constant."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.scalar_tensor(self.value, dtype=x.dtype) + (x.sum() * 0)


class _NewZerosFromInput(torch.nn.Module):
    """`Tensor.new_zeros(shape)`: derives shape and dtype from input."""

    def forward(self, x):
        return x.new_zeros(x.shape)


def _zeros_from_shape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(inputs=(x,), module=_ZerosFromShapeOf())

    return _draw()


def _ones_from_shape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(inputs=(x,), module=_OnesFromShapeOf())

    return _draw()


def _full_from_shape_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        fill_value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            module=_FullFromShapeOf(fill_value),
        )

    return _draw()


def _arange_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        start = draw(st.integers(min_value=0, max_value=5))
        length = draw(st.integers(min_value=1, max_value=20))
        step = draw(st.integers(min_value=1, max_value=3))
        end = start + length * step
        x = draw(
            tensor_st(
                (2, 3), torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(
            inputs=(x,),
            module=_ArangeFromInput(start, end, step),
        )

    return _draw()


def _scalar_tensor_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        x = draw(
            tensor_st(
                (2, 3), torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(
            inputs=(x,),
            module=_ScalarTensorOfDtypeOf(value),
        )

    return _draw()


def _new_zeros_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-1.0, 1.0)
            )
        )
        return OpSample(inputs=(x,), module=_NewZerosFromInput())

    return _draw()


def _index_advanced_sample_st() -> st.SearchStrategy[OpSample]:
    """`x[long_tensor]`: advanced indexing along axis 0.

    Output shape: index_tensor.shape + x.shape[1:].
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
        n_idx = draw(st.integers(min_value=1, max_value=4))
        idx = draw(
            tensor_st(
                (n_idx,),
                torch.int64,
                finite=True,
                domain=Interval(0, shape[0] - 1),
            )
        )
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )

        def op_fn(t, i):
            return t[i]

        return OpSample(
            inputs=(x, idx),
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _sdpa_sample_st() -> st.SearchStrategy[OpSample]:
    """`F.scaled_dot_product_attention(Q, K, V)`: shape (B, H, S, D)."""

    @st.composite
    def _draw(draw) -> OpSample:
        b = draw(st.integers(min_value=1, max_value=2))
        h = draw(st.integers(min_value=1, max_value=2))
        s = draw(st.integers(min_value=2, max_value=4))
        d = draw(st.integers(min_value=2, max_value=4))
        # Same shape for Q, K, V (typical use case).
        domain = Interval(-1.0, 1.0)
        q = draw(
            tensor_st((b, h, s, d), torch.float32, finite=True, domain=domain)
        )
        k = draw(
            tensor_st((b, h, s, d), torch.float32, finite=True, domain=domain)
        )
        v = draw(
            tensor_st((b, h, s, d), torch.float32, finite=True, domain=domain)
        )
        return OpSample(
            inputs=(q, k, v),
            module=TernaryPrimitive(F.scaled_dot_product_attention),
        )

    return _draw()


def _constructors_index_sdpa_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    VERY = TractCheckTolerance.VERY
    return [
        OpSpec(
            name="zeros",
            aten_ops=("zeros",),
            sample_st=_zeros_from_shape_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="ones",
            aten_ops=("ones",),
            sample_st=_ones_from_shape_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="full",
            aten_ops=("full",),
            sample_st=_full_from_shape_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="arange",
            aten_ops=("arange",),
            sample_st=_arange_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="scalar_tensor",
            aten_ops=("scalar_tensor",),
            sample_st=_scalar_tensor_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="new_zeros",
            aten_ops=("new_zeros",),
            sample_st=_new_zeros_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="index",
            aten_ops=("index",),
            sample_st=_index_advanced_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="sdpa",
            aten_ops=("scaled_dot_product_attention",),
            sample_st=_sdpa_sample_st(),
            tolerance=VERY,
        ),
    ]


# FFT (real-input forward and inverse)


def _fft_op_real_output(
    op: T.Callable[..., torch.Tensor], **kwargs
) -> T.Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a complex-output FFT op so the model output is real.

    PyTorch's `fft.*` ops return complex64; t2n simulates complex as a
    real tensor with an extra trailing axis of size 2 (`[real, imag]`).
    The proptest comparator only handles real tensors, so we apply
    `view_as_real` on the output: the layout then matches t2n's
    internal representation and the two sides are directly comparable.
    """

    def wrapped(x: torch.Tensor) -> torch.Tensor:
        # `.resolve_conj()` is a no-op when the conjugate bit isn't
        # set; `fft.ifft` does set it (lazy conjugation), so
        # `view_as_real` would otherwise raise.
        return torch.view_as_real(op(x, **kwargs).resolve_conj())

    return wrapped


def _fft_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """`torch.fft.fft(input, n=None, dim=-1, norm=None)`.

    The t2n FFT emitter (`torch_to_nnef/op/aten/fft.py:_fft`) requires
    `n` and `norm` to be None on the version path we test, and works
    on real (float32) input by padding to complex internally.
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
        # Bound input modestly to keep FFT magnitudes in range.
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(_fft_op_real_output(op, dim=dim)),
        )

    return _draw()


def _irfft_sample_st() -> st.SearchStrategy[OpSample]:
    """Real -> rfft -> irfft round-trip.

    The model is just `torch.fft.irfft(torch.fft.rfft(x, dim=d), dim=d)`,
    which yields a real tensor on both sides -- no view_as_real wrap
    needed for comparison.
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
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )

        def _roundtrip(x_in: torch.Tensor, d: int = dim) -> torch.Tensor:
            return torch.fft.irfft(torch.fft.rfft(x_in, dim=d), dim=d)

        return OpSample(inputs=(x,), module=UnaryPrimitive(_roundtrip))

    return _draw()


def _fftn_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """`torch.fft.fftn(input, s=None, dim=None, norm=None)` strategy.

    Draws a rank-2-or-3 real tensor and picks a contiguous prefix of
    axes to transform. The t2n emitter requires `s` and `norm` None.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=2, max_value=3))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=2, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        # Use last k axes (the most common pattern).
        k = draw(st.integers(min_value=1, max_value=rank))
        dim = tuple(range(rank - k, rank))
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-2.0, 2.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(_fft_op_real_output(op, dim=dim)),
        )

    return _draw()


def _fft_specs() -> T.List[OpSpec]:
    # Each strategy wraps the FFT op with `view_as_real` so the model
    # output is a real `(..., 2)` tensor -- the proptest comparator can
    # then diff PyTorch vs tract directly without needing a complex-
    # aware bridge. The wrap also sidesteps the conjugate-bit / numpy()
    # error that the bare ifft output hit in t2n's NPZ writer.
    return [
        OpSpec(
            name="fft_fft",
            aten_ops=("fft_fft",),
            sample_st=_fft_sample_st(torch.fft.fft),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            name="fft_ifft",
            aten_ops=("fft_ifft",),
            sample_st=_fft_sample_st(torch.fft.ifft),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            name="fft_rfft",
            aten_ops=("fft_rfft",),
            sample_st=_fft_sample_st(torch.fft.rfft),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            name="fft_fftn",
            aten_ops=("fft_fftn",),
            sample_st=_fftn_sample_st(torch.fft.fftn),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            name="fft_ifftn",
            aten_ops=("fft_ifftn",),
            sample_st=_fftn_sample_st(torch.fft.ifftn),
            tolerance=TractCheckTolerance.SUPER,
        ),
        OpSpec(
            # `fft_irfft` takes a Hermitian-symmetric one-sided spectrum
            # and returns real. We feed it a real-input rfft to build
            # such a spectrum; the comparator then sees real-on-both-
            # sides without needing the view_as_real wrapper.
            name="fft_irfft",
            aten_ops=("fft_irfft",),
            sample_st=_irfft_sample_st(),
            tolerance=TractCheckTolerance.SUPER,
        ),
    ]


# Identity-like glue ops + dtype casts + simple mutators-as-functional


def _identity_unary_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """Generic unary identity (clone, contiguous, detach).

    These are no-ops on the tensor data at runtime in eval mode.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(op))

    return _draw()


class _CastToDtype(torch.nn.Module):
    """`Tensor.to(dtype)`: runtime dtype cast."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


class _TypeAsFromOther(torch.nn.Module):
    """`Tensor.type_as(other)`: cast to other's dtype."""

    def forward(self, a, b):
        return a.type_as(b)


class _FillFunctional(torch.nn.Module):
    """Functional `torch.full_like(x, value)` standing in for fill_."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def _to_dtype_sample_st() -> st.SearchStrategy[OpSample]:
    """`Tensor.to(dtype)`: sweep cast targets among supported floats."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        target_dtype = draw(
            st.sampled_from([torch.float32, torch.float16, torch.float64])
        )
        return OpSample(inputs=(x,), module=_CastToDtype(target_dtype))

    return _draw()


def _type_as_sample_st() -> st.SearchStrategy[OpSample]:
    """`a.type_as(b)`: a takes b's dtype."""

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        a = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        # b is a tiny tensor whose dtype we want to inherit.
        target_dtype = draw(st.sampled_from([torch.float32, torch.float16]))
        b = draw(
            tensor_st(
                (1,),
                target_dtype,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        return OpSample(inputs=(a, b), module=_TypeAsFromOther())

    return _draw()


def _fill_sample_st() -> st.SearchStrategy[OpSample]:
    """Functional fill via `full_like`.

    PyTorch traces inplace `fill_` as `full_like` when no in-place
    graph is needed; this spec exercises that path.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=4, min_dim=2))
        x = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-10.0, 10.0)
            )
        )
        value = draw(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(inputs=(x,), module=_FillFunctional(value))

    return _draw()


def _glue_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="clone",
            aten_ops=("clone",),
            sample_st=_identity_unary_sample_st(torch.clone),
            tolerance=EXACT,
        ),
        OpSpec(
            name="contiguous",
            aten_ops=("contiguous",),
            sample_st=_identity_unary_sample_st(lambda t: t.contiguous()),
            tolerance=EXACT,
        ),
        OpSpec(
            name="detach",
            aten_ops=("detach",),
            sample_st=_identity_unary_sample_st(lambda t: t.detach()),
            tolerance=EXACT,
        ),
        OpSpec(
            name="to_dtype",
            aten_ops=("to",),
            sample_st=_to_dtype_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="type_as",
            aten_ops=("type_as",),
            sample_st=_type_as_sample_st(),
            tolerance=EXACT,
        ),
        OpSpec(
            name="fill",
            aten_ops=("fill",),
            sample_st=_fill_sample_st(),
            tolerance=EXACT,
        ),
    ]


# --- Complex constructors / conjugates ---
#
# These ops either consume or produce complex tensors. PyTorch's
# `complex64` doesn't survive the NPZ comparator unmodified, so:
#   - inputs that need to be complex are built via `view_as_complex`
#     on a real `(..., 2)` tensor;
#   - outputs that are complex are wrapped with `view_as_real(...)`
#     plus `.resolve_conj()` (lazy-conjugation fix-up).
# This matches the `_fft_op_real_output` pattern above.


def _complex_constructor_sample_st() -> st.SearchStrategy[OpSample]:
    """`torch.complex(real, imag)` returns complex; wrap with view_as_real.

    The t2n emitter at `torch_to_nnef/op/aten/complex.py:complex`
    stacks `real` and `imag` on a new trailing axis, producing the
    `(..., 2)` layout we read back with `view_as_real`.
    """

    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=1, max_rank=3, min_dim=2))
        real = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )
        imag = draw(
            tensor_st(
                shape, torch.float32, finite=True, domain=Interval(-3.0, 3.0)
            )
        )

        def _fn(r: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
            return torch.view_as_real(torch.complex(r, i).resolve_conj())

        return OpSample(inputs=(real, imag), module=BinaryPrimitive(_fn))

    return _draw()


_COMPLEX_UNARY_DEFAULT_DOMAIN = Interval(-3.0, 3.0)


def _complex_unary_sample_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
    *,
    domain: T.Optional[Interval] = None,
) -> st.SearchStrategy[OpSample]:
    """Complex-input unary op (`conj` / `conj_physical` / `sgn`).

    The model takes a real `(..., 2)` input which it folds into
    complex via `view_as_complex`, applies the op, and folds back
    via `view_as_real` so the comparator sees real-on-both-sides.
    """
    domain = domain or _COMPLEX_UNARY_DEFAULT_DOMAIN

    @st.composite
    def _draw(draw) -> OpSample:
        # `view_as_complex` requires last dim == 2 and a contiguous
        # last axis: build a free leading shape then append `(2,)`.
        leading = draw(shape_st(min_rank=1, max_rank=3, min_dim=2, max_dim=5))
        shape = tuple(leading) + (2,)
        x = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))

        def _fn(t: torch.Tensor) -> torch.Tensor:
            return torch.view_as_real(
                op(torch.view_as_complex(t)).resolve_conj()
            )

        return OpSample(inputs=(x,), module=UnaryPrimitive(_fn))

    return _draw()


def _complex_specs() -> T.List[OpSpec]:
    """`complex` / `conj` / `conj_physical` / `sgn` on complex tensors."""
    CLOSE = TractCheckTolerance.CLOSE
    return [
        OpSpec(
            name="complex",
            aten_ops=("complex",),
            sample_st=_complex_constructor_sample_st(),
            tolerance=CLOSE,
        ),
        OpSpec(
            # `conj` on complex flips the sign of the imag slice via the
            # `conjugate` NNEF fragment. (`resolve_conj-complex` is not
            # needed: `resolve_conj` is a no-op on a non-lazy tensor.)
            name="conj-complex",
            aten_ops=("conj",),
            sample_st=_complex_unary_sample_st(torch.conj),
            tolerance=CLOSE,
        ),
        OpSpec(
            # Mirror of `conj-complex`; `conj_physical` shares the
            # `_emit_conjugate` code path but is a separate aten op.
            name="conj_physical-complex",
            aten_ops=("conj_physical",),
            sample_st=_complex_unary_sample_st(torch.conj_physical),
            tolerance=CLOSE,
        ),
        OpSpec(
            # `sgn` on complex maps `z -> z / |z|` via `sgn_complex`
            # (with the 0 -> 0 carve-out); real-input `sgn` is already
            # covered by the existing `sign` spec.
            # Subnormal-near inputs underflow `x*x + y*y` to 0 in tract's
            # f32 (e.g. real=imag=3.35e-38 -> x*x=1.12e-75 below normal),
            # which then silently propagates 0 through the divide rather
            # than the documented z/|z|. Spec stays xfail until the
            # `sgn_complex` fragment guards against the underflow.
            name="sgn-complex-xfail",
            aten_ops=("sgn",),
            sample_st=_complex_unary_sample_st(
                torch.sgn, domain=Interval(-3.0, 3.0)
            ),
            tolerance=CLOSE,
            xfail_reason=(
                "sgn_complex underflows `x*x + y*y` to 0 for f32 inputs "
                "near the subnormal boundary (~1e-19): tract returns 0, "
                "torch returns z/|z|. Falsifying example: "
                "real=imag=3.35e-38 -> tract 0, torch 0.707..."
            ),
        ),
    ]


def _uninitialized_specs() -> T.Tuple[OpSpec, ...]:
    """Allocations with no defined contents, plus the deprecated range.

    Not translated yet: each spec carries `nnef_gap`, so the tract
    driver asserts the failure and the ONNX sweep still measures
    it. Implementing one means deleting that one field.
    """
    return (
        gap_spec(
            "empty",
            shape_only_st(
                lambda t: torch.empty(t.shape, dtype=t.dtype), "empty"
            ),
            "allocates without initializing, so there is nothing for a "
            "declarative graph to describe",
            nondeterministic=True,
        ),
        gap_spec(
            "empty_strided",
            shape_only_st(
                lambda t: torch.empty_strided((2, 2), (2, 1)) + t.sum() * 0,
                "empty_strided",
            ),
            f"{REASON_LAYOUT}; uninitialized as well",
            nondeterministic=True,
        ),
        gap_spec(
            "new_empty_strided",
            shape_only_st(
                lambda t: t.new_empty_strided((2, 2), (2, 1)),
                "new_empty_strided",
            ),
            f"{REASON_LAYOUT}; uninitialized as well",
            nondeterministic=True,
        ),
        gap_spec(
            "empty_permuted",
            shape_only_st(
                lambda t: torch.empty_permuted((2, 2), (1, 0)) + t.sum() * 0,
                "empty_permuted",
            ),
            f"{REASON_LAYOUT}; uninitialized as well",
            nondeterministic=True,
        ),
        gap_spec(
            "range",
            shape_only_st(lambda t: torch.range(0, 4) + t.sum() * 0, "range"),
            "the inclusive-end variant of `arange`, which we do translate; "
            "no emitter maps the deprecated spelling onto it",
        ),
    )


SPECS = (
    *_constructors_index_sdpa_specs(),
    *_fft_specs(),
    *_glue_specs(),
    *_complex_specs(),
    *_uninitialized_specs(),
)
