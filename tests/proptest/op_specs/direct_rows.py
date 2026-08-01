"""Direct specs for support-page rows not covered by broader samples.

These are intentionally narrow. Most existing specs exercise public
PyTorch APIs, which sometimes trace to a lower-level or decomposed ATen
name. This module fills the row-level gaps by calling the exact ATen
overload the support page lists, so the ONNX artifact can attach a
measurement to that row instead of falling back to the retired listing.
"""

import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import BinaryPrimitive, TernaryPrimitive, UnaryPrimitive
from ..inputs import Interval, tensor_st
from ..shapes import shape_st
from ._common import NnefGapStage, OpSample, OpSpec
from ._gap_common import gap_spec

_DEFAULT_FLOAT_DOMAIN = Interval(-3.0, 3.0)


class _FnModule(nn.Module):
    """Tiny named wrapper around an exact ATen call."""

    def __init__(self, fn: T.Callable[..., T.Any], name: str):
        super().__init__()
        self.fn = fn
        self.name = name

    def extra_repr(self) -> str:
        return f"op={self.name}"

    def forward(self, *args):
        return self.fn(*args)


class _OutputOnlyModule(nn.Module):
    """Return only the sequence output from recurrent modules."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return self.module(*args)[0]


def _anchor(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Attach a constant-like factory output to a real graph input."""
    return output + x.sum() * 0


def _unary_float_st(
    fn: T.Callable[[torch.Tensor], torch.Tensor],
    name: str,
    *,
    min_rank: int = 1,
    max_rank: int = 3,
    domain: Interval = _DEFAULT_FLOAT_DOMAIN,
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=min_rank, max_rank=max_rank, max_dim=5))
        x = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        return OpSample(inputs=(x,), module=UnaryPrimitive(fn))

    return _draw()


def _binary_same_float_st(
    fn: T.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    name: str,
    *,
    min_rank: int = 1,
    max_rank: int = 3,
    domain: Interval = _DEFAULT_FLOAT_DOMAIN,
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=min_rank, max_rank=max_rank, max_dim=5))
        x = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        y = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        return OpSample(inputs=(x, y), module=BinaryPrimitive(fn))

    return _draw()


def _ternary_same_float_st(
    fn: T.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    name: str,
    *,
    min_rank: int = 1,
    max_rank: int = 3,
    domain: Interval = _DEFAULT_FLOAT_DOMAIN,
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        shape = draw(shape_st(min_rank=min_rank, max_rank=max_rank, max_dim=5))
        x = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        y = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        z = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        return OpSample(inputs=(x, y, z), module=TernaryPrimitive(fn))

    return _draw()


def _fixed_unary_float_st(
    fn: T.Callable[[torch.Tensor], torch.Tensor],
    *,
    shape: T.Tuple[int, ...] = (2, 3),
    domain: Interval = _DEFAULT_FLOAT_DOMAIN,
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        return OpSample(inputs=(x,), module=UnaryPrimitive(fn))

    return _draw()


def _fixed_binary_same_float_st(
    fn: T.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    shape: T.Tuple[int, ...] = (2, 3),
    domain: Interval = _DEFAULT_FLOAT_DOMAIN,
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        y = draw(tensor_st(shape, torch.float32, finite=True, domain=domain))
        return OpSample(inputs=(x, y), module=BinaryPrimitive(fn))

    return _draw()


def _dummy_input_st(
    fn: T.Callable[[torch.Tensor], torch.Tensor], name: str
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(
            tensor_st(
                (2, 3),
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        return OpSample(inputs=(x,), module=UnaryPrimitive(fn))

    return _draw()


def _angle_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        real = draw(
            tensor_st(
                (2, 3),
                torch.float32,
                finite=True,
                domain=Interval(0.5, 3.0),
            )
        )
        imag = draw(
            tensor_st(
                (2, 3),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        x = torch.stack((real, imag), dim=-1)
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.angle.default(torch.view_as_complex(t))
            ),
        )

    return _draw()


def _activation_and_elementwise_specs() -> T.Tuple[OpSpec, ...]:
    exact = TractCheckTolerance.EXACT
    approx = TractCheckTolerance.APPROXIMATE
    very = TractCheckTolerance.VERY
    return (
        OpSpec(
            name="addcmul",
            aten_ops=("addcmul",),
            sample_st=_ternary_same_float_st(
                torch.ops.aten.addcmul.default, "addcmul"
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="angle",
            aten_ops=("angle",),
            sample_st=_angle_sample_st(),
            tolerance=approx,
        ),
        OpSpec(
            name="celu",
            aten_ops=("celu",),
            sample_st=_unary_float_st(torch.ops.aten.celu.default, "celu"),
            tolerance=very,
        ),
        OpSpec(
            name="clamp_max",
            aten_ops=("clamp_max",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.clamp_max.default(x, 0.25),
                "clamp_max",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="clamp_min",
            aten_ops=("clamp_min",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.clamp_min.default(x, -0.25),
                "clamp_min",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="hardshrink",
            aten_ops=("hardshrink",),
            sample_st=_unary_float_st(
                torch.ops.aten.hardshrink.default, "hardshrink"
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="isfinite",
            aten_ops=("isfinite",),
            sample_st=_unary_float_st(
                torch.ops.aten.isfinite.default, "isfinite"
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="lerp",
            aten_ops=("lerp",),
            sample_st=_ternary_same_float_st(
                torch.ops.aten.lerp.Tensor,
                "lerp",
                domain=Interval(-2.0, 2.0),
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="log_sigmoid",
            aten_ops=("log_sigmoid",),
            sample_st=_unary_float_st(
                torch.ops.aten.log_sigmoid.default, "log_sigmoid"
            ),
            tolerance=very,
        ),
        OpSpec(
            name="logit",
            aten_ops=("logit",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.logit.default(x, 1e-6),
                "logit",
                domain=Interval(0.01, 0.99),
            ),
            tolerance=very,
        ),
        OpSpec(
            name="rsub",
            aten_ops=("rsub",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.rsub.Scalar(x, 2.0, 1), "rsub"
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="softshrink",
            aten_ops=("softshrink",),
            sample_st=_unary_float_st(
                torch.ops.aten.softshrink.default, "softshrink"
            ),
            tolerance=exact,
        ),
    )


def _factory_specs() -> T.Tuple[OpSpec, ...]:
    exact = TractCheckTolerance.EXACT
    approx = TractCheckTolerance.APPROXIMATE
    return (
        OpSpec(
            name="bartlett_window",
            aten_ops=("bartlett_window",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.bartlett_window.default(8, dtype=x.dtype),
                    x,
                ),
                "bartlett_window",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="blackman_window",
            aten_ops=("blackman_window",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.blackman_window.default(8, dtype=x.dtype),
                    x,
                ),
                "blackman_window",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="empty_like",
            aten_ops=("empty_like",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.fill_.Scalar(
                    torch.ops.aten.empty_like.default(x),
                    0.0,
                ),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="fft_fftfreq",
            aten_ops=("fft_fftfreq",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.fft_fftfreq.default(8, dtype=x.dtype), x
                ),
                "fft_fftfreq",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="fft_rfftfreq",
            aten_ops=("fft_rfftfreq",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.fft_rfftfreq.default(8, dtype=x.dtype), x
                ),
                "fft_rfftfreq",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="hamming_window",
            aten_ops=("hamming_window",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.hamming_window.default(8, dtype=x.dtype), x
                ),
                "hamming_window",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="hann_window",
            aten_ops=("hann_window",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.hann_window.default(8, dtype=x.dtype), x
                ),
                "hann_window",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="kaiser_window",
            aten_ops=("kaiser_window",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.kaiser_window.default(8, dtype=x.dtype), x
                ),
                "kaiser_window",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="linspace",
            aten_ops=("linspace",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.linspace.default(0.0, 1.0, 8, dtype=x.dtype),
                    x,
                ),
                "linspace",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="logspace",
            aten_ops=("logspace",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.logspace.default(
                        0.0, 1.0, 8, 10.0, dtype=x.dtype
                    ),
                    x,
                ),
                "logspace",
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
        ),
        OpSpec(
            name="new_empty",
            aten_ops=("new_empty",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.fill_.Scalar(
                    torch.ops.aten.new_empty.default(x, [2, 3]),
                    0.0,
                ),
                "new_empty",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="new_full",
            aten_ops=("new_full",),
            sample_st=_unary_float_st(
                lambda x: _anchor(
                    torch.ops.aten.new_full.default(x, [2, 3], 1.25), x
                ),
                "new_full",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="new_ones",
            aten_ops=("new_ones",),
            sample_st=_unary_float_st(
                lambda x: _anchor(
                    torch.ops.aten.new_ones.default(x, [2, 3]), x
                ),
                "new_ones",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="quantize_per_tensor",
            aten_ops=("quantize_per_tensor", "dequantize"),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.dequantize.self(
                    torch.ops.aten.quantize_per_tensor.default(
                        x, 0.05, 10, torch.quint8
                    )
                ),
                "quantize_per_tensor",
                domain=Interval(-2.0, 2.0),
            ),
            tolerance=TractCheckTolerance.APPROXIMATE,
        ),
        OpSpec(
            name="tril_indices",
            aten_ops=("tril_indices",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.tril_indices.default(
                        4, 5, 0, dtype=torch.int64
                    ).to(torch.float32),
                    x,
                ),
                "tril_indices",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="triu_indices",
            aten_ops=("triu_indices",),
            sample_st=_dummy_input_st(
                lambda x: _anchor(
                    torch.ops.aten.triu_indices.default(
                        4, 5, 0, dtype=torch.int64
                    ).to(torch.float32),
                    x,
                ),
                "triu_indices",
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="vander",
            aten_ops=("vander",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.vander.default(x, 4, False),
                shape=(3,),
                domain=Interval(0.25, 2.0),
            ),
            tolerance=approx,
        ),
    )


def _shape_and_selection_specs() -> T.Tuple[OpSpec, ...]:
    exact = TractCheckTolerance.EXACT
    return (
        OpSpec(
            name="block_diag",
            aten_ops=("block_diag",),
            sample_st=_binary_same_float_st(
                lambda x, y: torch.ops.aten.block_diag.default([x, y]),
                "block_diag",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="cartesian_prod",
            aten_ops=("cartesian_prod",),
            sample_st=_binary_same_float_st(
                lambda x, y: torch.ops.aten.cartesian_prod.default(
                    [x.reshape(-1), y.reshape(-1)]
                ),
                "cartesian_prod",
                min_rank=1,
                max_rank=1,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="cummax",
            aten_ops=("cummax",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.cummax.default(x, 0)[0],
                shape=(3, 2),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="cummin",
            aten_ops=("cummin",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.cummin.default(x, 0)[0],
                shape=(3, 2),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="hstack",
            aten_ops=("hstack",),
            sample_st=_fixed_binary_same_float_st(
                lambda x, y: torch.ops.aten.hstack.default([x, y]),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="linalg_diagonal",
            aten_ops=("linalg_diagonal",),
            sample_st=_unary_float_st(
                torch.ops.aten.linalg_diagonal.default,
                "linalg_diagonal",
                min_rank=2,
                max_rank=3,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="matrix_H",
            aten_ops=("matrix_H",),
            sample_st=_unary_float_st(
                torch.ops.aten.matrix_H.default,
                "matrix_H",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="movedim",
            aten_ops=("movedim",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.movedim.int(x, 0, -1),
                "movedim",
                min_rank=2,
                max_rank=4,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="native_channel_shuffle",
            aten_ops=("native_channel_shuffle",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.native_channel_shuffle.default(x, 2),
                shape=(1, 4, 3, 3),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="one_hot",
            aten_ops=("one_hot",),
            sample_st=_one_hot_sample_st(),
            tolerance=exact,
        ),
        OpSpec(
            name="take_along_dim",
            aten_ops=("take_along_dim",),
            sample_st=_take_along_dim_sample_st(),
            tolerance=exact,
        ),
        OpSpec(
            name="unflatten",
            aten_ops=("unflatten",),
            sample_st=_unflatten_sample_st(),
            tolerance=exact,
        ),
        OpSpec(
            name="unsafe_chunk",
            aten_ops=("unsafe_chunk",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.unsafe_chunk.default(x, 2, 0)[0],
                shape=(4, 3),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="unsafe_split",
            aten_ops=("unsafe_split",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.unsafe_split.Tensor(x, 1, 0)[0],
                "unsafe_split",
                min_rank=1,
                max_rank=3,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="unsafe_split_with_sizes",
            aten_ops=("unsafe_split_with_sizes",),
            sample_st=_unsafe_split_with_sizes_sample_st(),
            tolerance=exact,
        ),
        OpSpec(
            name="view_as",
            aten_ops=("view_as",),
            sample_st=_binary_same_float_st(
                torch.ops.aten.view_as.default,
                "view_as",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="vstack",
            aten_ops=("vstack",),
            sample_st=_fixed_binary_same_float_st(
                lambda x, y: torch.ops.aten.vstack.default([x, y]),
            ),
            tolerance=exact,
        ),
        OpSpec(
            name="zero",
            aten_ops=("zero",),
            sample_st=_fixed_unary_float_st(lambda x: x.clone().zero_()),
            tolerance=exact,
        ),
    )


def _one_hot_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((2, 3), torch.int64, domain=Interval(0, 4)))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.one_hot.default(t, 5).to(torch.float32)
            ),
        )

    return _draw()


def _take_along_dim_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((2, 4), torch.float32, finite=True))
        idx = draw(tensor_st((2, 2), torch.int64, domain=Interval(0, 3)))
        return OpSample(
            inputs=(x, idx),
            module=BinaryPrimitive(
                lambda t, i: torch.ops.aten.take_along_dim.default(t, i, 1)
            ),
        )

    return _draw()


def _unsafe_split_with_sizes_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((4, 3), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.unsafe_split_with_sizes.default(
                    t, [1, 3], 0
                )[0]
            ),
        )

    return _draw()


def _unflatten_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((2, 3), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.unflatten.int(t.reshape(-1), 0, [2, 3])
            ),
        )

    return _draw()


def _complex_row_specs() -> T.Tuple[OpSpec, ...]:
    close = TractCheckTolerance.CLOSE
    return (
        OpSpec(
            name="imag",
            aten_ops=("imag",),
            sample_st=_complex_unary_row_st(
                lambda z: torch.ops.aten.imag.default(z), "imag"
            ),
            tolerance=close,
        ),
        OpSpec(
            name="polar",
            aten_ops=("polar",),
            sample_st=_polar_sample_st(),
            tolerance=close,
        ),
        OpSpec(
            name="real",
            aten_ops=("real",),
            sample_st=_complex_unary_row_st(
                lambda z: torch.ops.aten.real.default(z), "real"
            ),
            tolerance=close,
        ),
        OpSpec(
            name="view_as_complex",
            aten_ops=("view_as_complex",),
            sample_st=_view_as_complex_sample_st(),
            tolerance=close,
        ),
        OpSpec(
            name="view_as_real",
            aten_ops=("view_as_real",),
            sample_st=_complex_unary_row_st(
                lambda z: torch.ops.aten.view_as_real.default(z),
                "view_as_real",
            ),
            tolerance=close,
        ),
    )


def _complex_unary_row_st(
    op: T.Callable[[torch.Tensor], torch.Tensor],
    name: str,
) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((2, 3, 2), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(lambda t: op(torch.view_as_complex(t))),
        )

    return _draw()


def _view_as_complex_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((2, 3, 2), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.view_as_real(
                    torch.ops.aten.view_as_complex.default(t)
                )
            ),
        )

    return _draw()


def _polar_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        radius = draw(
            tensor_st(
                (2, 3),
                torch.float32,
                finite=True,
                domain=Interval(0.1, 3.0),
            )
        )
        angle = draw(tensor_st((2, 3), torch.float32, finite=True))
        return OpSample(
            inputs=(radius, angle),
            module=BinaryPrimitive(
                lambda r, a: torch.view_as_real(
                    torch.ops.aten.polar.default(r, a)
                )
            ),
        )

    return _draw()


def _conv_pool_specs() -> T.Tuple[OpSpec, ...]:
    exact = TractCheckTolerance.EXACT
    approx = TractCheckTolerance.APPROXIMATE
    return (
        OpSpec(
            name="adaptive_avg_pool3d",
            aten_ops=("adaptive_avg_pool3d",),
            sample_st=_adaptive_avg_pool3d_sample_st(),
            tolerance=exact,
        ),
        OpSpec(
            name="adaptive_max_pool1d",
            aten_ops=("adaptive_max_pool1d",),
            sample_st=_adaptive_max_pool1d_sample_st(),
            tolerance=exact,
        ),
        OpSpec(
            name="adaptive_max_pool3d",
            aten_ops=("adaptive_max_pool3d",),
            sample_st=_adaptive_max_pool3d_sample_st(),
            tolerance=exact,
        ),
        gap_spec(
            "conv_transpose1d",
            _conv_transpose_sample_st(1),
            "the direct ATen transposed-convolution row reaches a registered "
            "emitter, but the current exporter crashes before producing NNEF",
            stage=NnefGapStage.RAW_ERROR,
            emitter_registered=True,
        ),
        gap_spec(
            "conv_transpose2d",
            _conv_transpose_sample_st(2),
            "same direct ATen crash as `conv_transpose1d`",
            stage=NnefGapStage.RAW_ERROR,
            emitter_registered=True,
        ),
        gap_spec(
            "conv_transpose3d",
            _conv_transpose_sample_st(3),
            "same direct ATen crash as `conv_transpose1d`",
            stage=NnefGapStage.RAW_ERROR,
            emitter_registered=True,
        ),
        OpSpec(
            name="grid_sampler_3d",
            aten_ops=("grid_sampler_3d",),
            sample_st=_grid_sampler_3d_sample_st(),
            tolerance=approx,
        ),
        gap_spec(
            "replication_pad1d",
            _replication_pad1d_sample_st(),
            "the registered emitter writes NNEF, but tract declines to run "
            "the emitted 1-D replication-pad graph",
            stage=NnefGapStage.TRACT_ERROR,
            emitter_registered=True,
        ),
    )


def _adaptive_avg_pool3d_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((1, 2, 4, 4, 4), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.adaptive_avg_pool3d.default(
                    t, [2, 2, 2]
                )
            ),
        )

    return _draw()


def _adaptive_max_pool1d_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((1, 2, 6), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.adaptive_max_pool1d.default(t, [3])[0]
            ),
        )

    return _draw()


def _adaptive_max_pool3d_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((1, 2, 4, 4, 4), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.adaptive_max_pool3d.default(
                    t, [2, 2, 2]
                )[0]
            ),
        )

    return _draw()


def _conv_transpose_sample_st(rank: int) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        spatial = (5,) * rank
        kernel = (3,) * rank
        x = draw(tensor_st((1, 2, *spatial), torch.float32, finite=True))
        weight = draw(tensor_st((2, 3, *kernel), torch.float32, finite=True))
        fn = {
            1: torch.ops.aten.conv_transpose1d.default,
            2: torch.ops.aten.conv_transpose2d.input,
            3: torch.ops.aten.conv_transpose3d.input,
        }[rank]
        return OpSample(
            inputs=(x, weight),
            module=BinaryPrimitive(
                lambda t, w, _fn=fn, _r=rank: _fn(
                    t,
                    w,
                    None,
                    [1] * _r,
                    [0] * _r,
                    [0] * _r,
                    1,
                    [1] * _r,
                )
            ),
        )

    return _draw()


def _grid_sampler_3d_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((1, 2, 3, 3, 3), torch.float32, finite=True))
        grid = draw(
            tensor_st(
                (1, 2, 2, 2, 3),
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        return OpSample(
            inputs=(x, grid),
            module=BinaryPrimitive(
                lambda t, g: torch.ops.aten.grid_sampler_3d.default(
                    t, g, 0, 0, True
                )
            ),
        )

    return _draw()


def _replication_pad1d_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((1, 2, 4), torch.float32, finite=True))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.replication_pad1d.default(t, [1, 2])
            ),
        )

    return _draw()


def _linear_algebra_specs() -> T.Tuple[OpSpec, ...]:
    approx = TractCheckTolerance.APPROXIMATE
    close = TractCheckTolerance.CLOSE
    return (
        OpSpec(
            name="baddbmm",
            aten_ops=("baddbmm",),
            sample_st=_baddbmm_sample_st(),
            tolerance=approx,
        ),
        OpSpec(
            name="bilinear",
            aten_ops=("bilinear",),
            sample_st=_bilinear_sample_st(),
            tolerance=close,
        ),
        OpSpec(
            name="chain_matmul",
            aten_ops=("chain_matmul",),
            sample_st=_chain_matmul_sample_st(),
            tolerance=close,
        ),
        OpSpec(
            name="frobenius_norm",
            aten_ops=("frobenius_norm",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.frobenius_norm.dim(x, [0, 1], False),
                "frobenius_norm",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="linalg_cross",
            aten_ops=("linalg_cross",),
            sample_st=_linalg_cross_sample_st(),
            tolerance=approx,
        ),
        OpSpec(
            name="linalg_norm",
            aten_ops=("linalg_norm",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.linalg.norm(x, ord=2, dim=1),
            ),
            tolerance=approx,
        ),
        gap_spec(
            "norm",
            _fixed_unary_float_st(lambda x: torch.ops.aten.norm.Scalar(x, 2.0)),
            "the direct `aten::norm` row reaches a registered emitter, but "
            "the current exporter crashes before producing NNEF",
            stage=NnefGapStage.RAW_ERROR,
            emitter_registered=True,
        ),
        OpSpec(
            name="trapz",
            aten_ops=("trapz",),
            sample_st=_fixed_unary_float_st(
                lambda x: torch.ops.aten.trapz.dx(x, dx=1.0, dim=1),
            ),
            tolerance=close,
        ),
    )


def _baddbmm_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        self_t = draw(tensor_st((2, 3, 5), torch.float32, finite=True))
        a = draw(tensor_st((2, 3, 4), torch.float32, finite=True))
        b = draw(tensor_st((2, 4, 5), torch.float32, finite=True))
        return OpSample(
            inputs=(self_t, a, b),
            module=TernaryPrimitive(torch.ops.aten.baddbmm.default),
        )

    return _draw()


def _linalg_cross_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        domain = Interval(-2.0, 2.0)
        x = draw(tensor_st((2, 3), torch.float32, finite=True, domain=domain))
        y = draw(tensor_st((2, 3), torch.float32, finite=True, domain=domain))
        return OpSample(
            inputs=(x, y),
            module=BinaryPrimitive(
                lambda a, b: torch.ops.aten.linalg_cross.default(a, b, dim=-1)
            ),
        )

    return _draw()


def _bilinear_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        domain = Interval(-1.0, 1.0)
        x = draw(tensor_st((2, 3), torch.float32, finite=True, domain=domain))
        y = draw(tensor_st((2, 4), torch.float32, finite=True, domain=domain))
        weight = draw(
            tensor_st((5, 3, 4), torch.float32, finite=True, domain=domain)
        )
        return OpSample(
            inputs=(x, y, weight),
            module=_FnModule(
                lambda a, b, w: torch.ops.aten.bilinear.default(a, b, w, None),
                "bilinear",
            ),
        )

    return _draw()


def _chain_matmul_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        domain = Interval(-1.0, 1.0)
        a = draw(tensor_st((2, 3), torch.float32, finite=True, domain=domain))
        b = draw(tensor_st((3, 4), torch.float32, finite=True, domain=domain))
        c = draw(tensor_st((4, 2), torch.float32, finite=True, domain=domain))
        return OpSample(
            inputs=(a, b, c),
            module=_FnModule(
                lambda x, y, z: torch.ops.aten.chain_matmul.default([x, y, z]),
                "chain_matmul",
            ),
        )

    return _draw()


def _loss_specs() -> T.Tuple[OpSpec, ...]:
    approx = TractCheckTolerance.APPROXIMATE
    close = TractCheckTolerance.CLOSE
    return (
        OpSpec(
            name="hinge_embedding_loss",
            aten_ops=("hinge_embedding_loss",),
            sample_st=_hinge_embedding_loss_sample_st(),
            tolerance=approx,
        ),
        OpSpec(
            name="nll_loss_direct",
            aten_ops=("nll_loss",),
            sample_st=_nll_loss_row_sample_st(spatial_rank=0),
            tolerance=approx,
        ),
        OpSpec(
            name="nll_loss2d",
            aten_ops=("nll_loss2d",),
            sample_st=_nll_loss_row_sample_st(spatial_rank=2),
            tolerance=approx,
        ),
        OpSpec(
            name="triplet_margin_loss",
            aten_ops=("triplet_margin_loss",),
            sample_st=_triplet_margin_loss_sample_st(),
            tolerance=close,
        ),
    )


def _hinge_embedding_loss_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(tensor_st((4,), torch.float32, finite=True))
        raw = draw(tensor_st((4,), torch.int64, domain=Interval(0, 1)))
        target = (raw * 2 - 1).to(torch.float32)
        return OpSample(
            inputs=(x, target),
            module=BinaryPrimitive(
                lambda a, b: torch.ops.aten.hinge_embedding_loss.default(
                    a, b, 1.0, 1
                )
            ),
        )

    return _draw()


def _nll_loss_row_sample_st(spatial_rank: int) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        c = 4
        spatial = (2,) * spatial_rank
        x = draw(
            tensor_st(
                (2, c, *spatial),
                torch.float32,
                finite=True,
                domain=Interval(-3.0, 3.0),
            )
        )
        log_probs = F.log_softmax(x, dim=1)
        target = draw(
            tensor_st((2, *spatial), torch.int64, domain=Interval(0, c - 1))
        )
        fn = (
            torch.ops.aten.nll_loss.default
            if spatial_rank == 0
            else torch.ops.aten.nll_loss2d.default
        )
        return OpSample(
            inputs=(log_probs, target),
            module=BinaryPrimitive(
                lambda a, b, _fn=fn: _fn(a, b, None, 1, -100)
            ),
        )

    return _draw()


def _triplet_margin_loss_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        domain = Interval(-0.5, 0.5)
        anchor = draw(
            tensor_st((3, 4), torch.float32, finite=True, domain=domain)
        )
        positive = anchor + 2.0
        negative = anchor + 0.25
        return OpSample(
            inputs=(anchor, positive, negative),
            module=TernaryPrimitive(
                lambda a, p, n: torch.ops.aten.triplet_margin_loss.default(
                    a, p, n, 1.0, 2.0, 1e-6, False, 1
                )
            ),
        )

    return _draw()


def _rnn_specs() -> T.Tuple[OpSpec, ...]:
    approx = TractCheckTolerance.APPROXIMATE
    return (
        OpSpec(
            name="gru",
            aten_ops=("gru",),
            sample_st=_recurrent_module_sample_st(nn.GRU),
            tolerance=approx,
        ),
        OpSpec(
            name="lstm",
            aten_ops=("lstm",),
            sample_st=_recurrent_module_sample_st(nn.LSTM),
            tolerance=approx,
        ),
        OpSpec(
            name="rnn_relu",
            aten_ops=("rnn_relu",),
            sample_st=_recurrent_module_sample_st(
                lambda i, h, batch_first: nn.RNN(
                    i, h, nonlinearity="relu", batch_first=batch_first
                )
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="rnn_tanh",
            aten_ops=("rnn_tanh",),
            sample_st=_recurrent_module_sample_st(
                lambda i, h, batch_first: nn.RNN(
                    i, h, nonlinearity="tanh", batch_first=batch_first
                )
            ),
            tolerance=approx,
        ),
    )


def _recurrent_module_sample_st(factory) -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        input_size = 3
        hidden_size = 4
        batch_first = draw(st.booleans())
        shape = (2, 3, input_size) if batch_first else (3, 2, input_size)
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        module = factory(
            input_size, hidden_size, batch_first=batch_first
        ).eval()
        return OpSample(inputs=(x,), module=_OutputOnlyModule(module))

    return _draw()


def _spectral_specs() -> T.Tuple[OpSpec, ...]:
    return (
        gap_spec(
            "stft",
            _stft_sample_st(),
            "the direct `aten::stft` row reaches a registered emitter, but "
            "the current exporter crashes before producing NNEF",
            stage=NnefGapStage.RAW_ERROR,
            emitter_registered=True,
        ),
        OpSpec(
            name="istft",
            aten_ops=("istft",),
            sample_st=_istft_sample_st(),
            tolerance=TractCheckTolerance.SUPER,
        ),
    )


def _stft_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(
            tensor_st(
                (16,),
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.view_as_real(
                    torch.ops.aten.stft.default(
                        t, 4, 2, 4, None, False, False, True
                    )
                )
            ),
        )

    return _draw()


def _istft_sample_st() -> st.SearchStrategy[OpSample]:
    @st.composite
    def _draw(draw) -> OpSample:
        x = draw(
            tensor_st(
                (3, 4, 2),
                torch.float32,
                finite=True,
                domain=Interval(-1.0, 1.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                lambda t: torch.ops.aten.istft.default(
                    torch.view_as_complex(t),
                    4,
                    2,
                    4,
                    None,
                    False,
                    False,
                    True,
                    None,
                    False,
                )
            ),
        )

    return _draw()


def _misc_specs() -> T.Tuple[OpSpec, ...]:
    approx = TractCheckTolerance.APPROXIMATE
    return (
        OpSpec(
            name="logsumexp",
            aten_ops=("logsumexp",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.logsumexp.default(x, [1], False),
                "logsumexp",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="mvlgamma",
            aten_ops=("mvlgamma",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.mvlgamma.default(x, 2),
                "mvlgamma",
                domain=Interval(1.0, 5.0),
            ),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="nanmean",
            aten_ops=("nanmean",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.nanmean.default(x, [1], False),
                "nanmean",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=approx,
        ),
        OpSpec(
            name="nansum",
            aten_ops=("nansum",),
            sample_st=_unary_float_st(
                lambda x: torch.ops.aten.nansum.default(x, [1], False),
                "nansum",
                min_rank=2,
                max_rank=2,
            ),
            tolerance=approx,
        ),
    )


SPECS: T.Tuple[OpSpec, ...] = (
    _activation_and_elementwise_specs()
    + _factory_specs()
    + _shape_and_selection_specs()
    + _complex_row_specs()
    + _conv_pool_specs()
    + _linear_algebra_specs()
    + _loss_specs()
    + _rnn_specs()
    + _spectral_specs()
    + _misc_specs()
)
