"""Spec builders for the factory op group."""

import typing as T
from functools import partial

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


class _ZerosFromShapeOf(torch.nn.Module):
    """``torch.zeros(*x.shape, dtype=x.dtype)`` -- derives shape from input."""

    def forward(self, x):
        return torch.zeros(x.shape, dtype=x.dtype)


class _OnesFromShapeOf(torch.nn.Module):
    """``torch.ones(*x.shape, dtype=x.dtype)``."""

    def forward(self, x):
        return torch.ones(x.shape, dtype=x.dtype)


class _FullFromShapeOf(torch.nn.Module):
    """``torch.full(x.shape, fill_value, dtype=x.dtype)`` -- swept fills."""

    def __init__(self, fill_value: float):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        return torch.full(x.shape, self.fill_value, dtype=x.dtype)


class _ArangeFromInput(torch.nn.Module):
    """``torch.arange(start, end, step)`` -- start/end/step baked at init.

    The input is ignored at runtime, but kept so the export pipeline has
    a real graph input. We attach a no-op dependency via ``+ x.sum() * 0``
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
    """``torch.scalar_tensor(value, dtype=x.dtype)`` -- 0-d constant."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.scalar_tensor(self.value, dtype=x.dtype) + (x.sum() * 0)


class _NewZerosFromInput(torch.nn.Module):
    """``Tensor.new_zeros(shape)`` -- derives shape and dtype from input."""

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
        return OpSample(inputs=(x,), kwargs={}, module=_ZerosFromShapeOf())

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
        return OpSample(inputs=(x,), kwargs={}, module=_OnesFromShapeOf())

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
            kwargs={},
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
            kwargs={},
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
            kwargs={},
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
        return OpSample(inputs=(x,), kwargs={}, module=_NewZerosFromInput())

    return _draw()


def _index_advanced_sample_st() -> st.SearchStrategy[OpSample]:
    """``x[long_tensor]`` -- advanced indexing along axis 0.

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
            kwargs={},
            module=BinaryPrimitive(op_fn),
        )

    return _draw()


def _sdpa_sample_st() -> st.SearchStrategy[OpSample]:
    """``F.scaled_dot_product_attention(Q, K, V)`` -- shape (B, H, S, D)."""

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
            kwargs={},
            module=TernaryPrimitive(F.scaled_dot_product_attention),
        )

    return _draw()


def _constructors_index_sdpa_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    VERY = TractCheckTolerance.VERY
    return [
        OpSpec(
            name="zeros",
            sample_st=_zeros_from_shape_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="ones",
            sample_st=_ones_from_shape_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="full",
            sample_st=_full_from_shape_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="arange",
            sample_st=_arange_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.int64,),
        ),
        OpSpec(
            name="scalar_tensor",
            sample_st=_scalar_tensor_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="new_zeros",
            sample_st=_new_zeros_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="index",
            sample_st=_index_advanced_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="sdpa",
            sample_st=_sdpa_sample_st(),
            tolerance=VERY,
            dtypes_hint=(torch.float32,),
        ),
    ]


# FFT (real-input forward and inverse)


def _fft_sample_st(
    op: T.Callable[..., torch.Tensor],
) -> st.SearchStrategy[OpSample]:
    """``torch.fft.fft(input, n=None, dim=-1, norm=None)``.

    The t2n FFT emitter (``torch_to_nnef/op/aten/fft.py:_fft``) requires
    ``n`` and ``norm`` to be None on the version path we test, and works
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
            kwargs={},
            module=UnaryPrimitive(partial(op, dim=dim)),
        )

    return _draw()


def _fft_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            # PyTorch's complex output (shape ``(...,)`` complex64) and
            # tract's unfolded output (shape ``(..., 2)`` real, with the
            # last axis being ``[real, imag]``) don't compare apples-to-
            # apples in the current comparator. FFT proptest support
            # needs a complex-aware comparator that either folds tract's
            # output back to complex or unfolds PyTorch's output to
            # match tract.
            name="fft_fft-xfail",
            sample_st=_fft_sample_st(torch.fft.fft),
            tolerance=TractCheckTolerance.SUPER,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "FFT returns complex; comparator doesn't bridge "
                "PyTorch's complex64 output vs tract's (real, imag) "
                "unfolded layout."
            ),
        ),
        OpSpec(
            # Additionally, t2n's NPZ writer at
            # ``model_wrapper.py`` raises ``RuntimeError: Can't call
            # numpy() on Tensor that has conjugate bit set`` for IFFT
            # output -- needs a ``.resolve_conj()`` before serialization.
            name="fft_ifft-xfail",
            sample_st=_fft_sample_st(torch.fft.ifft),
            tolerance=TractCheckTolerance.SUPER,
            dtypes_hint=(torch.float32,),
            xfail_reason=(
                "Same complex-output comparator gap as fft_fft, plus "
                "t2n model_wrapper.py missing .resolve_conj() before "
                ".numpy() for ifft output (conjugate bit set)."
            ),
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
        return OpSample(inputs=(x,), kwargs={}, module=UnaryPrimitive(op))

    return _draw()


class _CastToDtype(torch.nn.Module):
    """``Tensor.to(dtype)`` -- runtime dtype cast."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


class _TypeAsFromOther(torch.nn.Module):
    """``Tensor.type_as(other)`` -- cast to other's dtype."""

    def forward(self, a, b):
        return a.type_as(b)


class _FillFunctional(torch.nn.Module):
    """Functional ``torch.full_like(x, value)`` standing in for fill_."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def _to_dtype_sample_st() -> st.SearchStrategy[OpSample]:
    """``Tensor.to(dtype)`` -- sweep cast targets among supported floats."""

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
        return OpSample(
            inputs=(x,), kwargs={}, module=_CastToDtype(target_dtype)
        )

    return _draw()


def _type_as_sample_st() -> st.SearchStrategy[OpSample]:
    """``a.type_as(b)`` -- a takes b's dtype."""

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
        return OpSample(inputs=(a, b), kwargs={}, module=_TypeAsFromOther())

    return _draw()


def _fill_sample_st() -> st.SearchStrategy[OpSample]:
    """Functional fill via ``full_like``.

    PyTorch traces inplace ``fill_`` as ``full_like`` when no in-place
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
        return OpSample(inputs=(x,), kwargs={}, module=_FillFunctional(value))

    return _draw()


def _glue_specs() -> T.List[OpSpec]:
    EXACT = TractCheckTolerance.EXACT
    return [
        OpSpec(
            name="clone",
            sample_st=_identity_unary_sample_st(torch.clone),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="contiguous",
            sample_st=_identity_unary_sample_st(lambda t: t.contiguous()),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="detach",
            sample_st=_identity_unary_sample_st(lambda t: t.detach()),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
        OpSpec(
            name="to_dtype",
            sample_st=_to_dtype_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32, torch.float16, torch.float64),
        ),
        OpSpec(
            name="type_as",
            sample_st=_type_as_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32, torch.float16),
        ),
        OpSpec(
            name="fill",
            sample_st=_fill_sample_st(),
            tolerance=EXACT,
            dtypes_hint=(torch.float32,),
        ),
    ]


# Depth: conv with dilation/groups + pool with dilation
