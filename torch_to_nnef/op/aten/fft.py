import logging

import nnef
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
)
from torch_to_nnef.torch_graph import PythonConstant
from torch_to_nnef.utils import torch_version

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


def _fft(
    node,
    g,
    name_to_tensor,
    inference_target,
    inverse=False,
):
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L360
    # const Tensor& self, c10::optional<int64_t> n, int64_t dim,
    # c10::optional<c10::string_view> norm
    if (
        not isinstance(inference_target, TractNNEF)
        or inference_target.version < "0.20.7"
    ):
        raise T2NErrorNotImplemented(inference_target)
    input_node, n_node, dim_node, norm_node = node.inputs
    if n_node.data is not None or norm_node.data is not None:
        raise T2NErrorNotImplemented("n or norm unexpected")

    dim = pick_axis(input_node, dim_node.data)

    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if input_node.dtype in [torch.float32, torch.float64]:
        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "unsqueeze",
            inputs=nnef_tensor,
            attrs={"axes": [pick_axis(input_node, -1) + 1]},
            pass_quantization_params=True,
            output_tensor_name_suffix="complex_cast_unsqueze",
        )
        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="pad",
            inputs=output_nnef_tensor,
            attrs={
                "padding": [(0, 0)] * input_node.rank + [(0, 1)],
                "value": 0.0,
            },
            output_tensor_name_suffix="complex_cast_pad",
        )
        casted_complex_input_tensor = output_nnef_tensor
    elif input_node.dtype not in [torch.complex64, torch.complex128]:
        raise T2NErrorNotImplemented()
    else:
        casted_complex_input_tensor = nnef_tensor

    suffix = None
    if inverse and norm_node.data is None:
        # backward by default means 1/n
        suffix = "need_norm"
        norm_node.set_data("backward")

    output_tensor = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_fft",
        inputs=casted_complex_input_tensor,
        attrs={"axis": dim, "inverse": inverse},
        output_tensor_name_suffix=suffix,
    )
    if inverse and norm_node.data == "backward":
        if inference_target.has_dynamic_axes:
            raise T2NErrorNotImplemented("Need to use implement")

        divisor_value = input_node.shape[dim]
        divisor_tensor = get_or_add_tensor_variable_in_nnef(
            g,
            PythonConstant(
                name=f"{output_tensor.name}_divisor",
                data=float(divisor_value),
            ),
            name_to_tensor,
        )

        # input_node, n_node, dim_node, norm_node = node.inputs
        input_to_real_tensor = output_tensor

        node.outputs[0].dtype = torch.complex64
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "div",
            inputs=(
                input_to_real_tensor,
                divisor_tensor,
            ),
        )

    return ["tract_core"]


@OP_REGISTRY.register()
def stft(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:stft' to NNEF."""
    # NEED SOME FACTOR OUT WITH _FFT and fix to pass window in NNEF-Tools
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L826
    if (
        not isinstance(inference_target, TractNNEF)
        or inference_target.version < "0.20.7"
    ):
        raise T2NErrorNotImplemented(inference_target)
    if "0.21.14" <= inference_target.version <= "0.21.15":
        LOGGER.warning(
            "tract %s has a known slice-fusion bug that corrupts STFT "
            "results (https://github.com/sonos/tract/commit/8b8f4537c). "
            "Use tract 0.21.13 or >= 0.22.0 instead.",
            inference_target.version.to_str(),
        )
    if torch_version() < "2.7.0":
        (
            input_node,  # Tensor
            n_fft_node,  # int,
            hop_length_node,  # Optional[int] = None
            win_length_node,  # Optional[int] = None
            window_node,  # Optional[Tensor] = None
            normalized_node,  # bool = False
            onesided_node,  # Optional[bool] = None
            _,  # return_complex_node Optional[bool] = None
        ) = node.inputs
        # is_center = True
        # pad_kind = "reflect"
    else:
        (
            input_node,  # Tensor
            n_fft_node,  # int,
            hop_length_node,  # Optional[int] = None
            win_length_node,  # Optional[int] = None
            window_node,  # Optional[Tensor] = None
            normalized_node,  # bool = False
            onesided_node,  # Optional[bool] = None
            center_node,
            pad_node,
            *_,  # return_complex_node Optional[bool] = None
        ) = node.inputs
        assert center_node.data is True
        assert pad_node.data is None
    assert isinstance(n_fft_node.data, int)
    assert isinstance(hop_length_node.data, int)
    assert isinstance(win_length_node.data, int) or win_length_node.data is None
    assert window_node.dtype == torch.float32
    if win_length_node.data is None:
        win_length_node.set_data(n_fft_node.data, force_shape=True)
    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if input_node.dtype in [torch.float32, torch.float64]:
        if input_node.shape[-1] == 1:
            output_nnef_tensor = nnef_tensor
        else:
            output_nnef_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "unsqueeze",
                inputs=nnef_tensor,
                attrs={"axes": [pick_axis(input_node, -1) + 1]},
                pass_quantization_params=True,
                output_tensor_name_suffix="complex_cast_unsqueze",
            )

        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="pad",
            inputs=output_nnef_tensor,
            attrs={
                "padding": [(0, 0)] * input_node.rank + [(0, 1)],
                "value": 0.0,
            },
            output_tensor_name_suffix="complex_cast_pad",
        )
        casted_complex_input_tensor = output_nnef_tensor
    elif input_node.dtype not in [torch.complex64, torch.complex128]:
        raise T2NErrorNotImplemented(
            f"complex type not supported: {input_node.dtype}"
        )
    else:
        casted_complex_input_tensor = nnef_tensor
    dim = pick_axis(input_node, -1)

    if window_node.data is None:
        window_node.set_data(torch.ones(win_length_node.data), force_shape=True)

    window_tensor = get_or_add_tensor_variable_in_nnef(
        g, window_node, name_to_tensor
    )
    frame = n_fft_node.data
    stride = hop_length_node.data
    # n_fft_node not exposed ?
    output_nnef_tensor = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_stft",
        inputs=casted_complex_input_tensor,
        attrs={
            "axis": dim,
            "frame": frame,
            "stride": stride,
            "window": nnef.Identifier(window_tensor.name),
        },
        output_tensor_name_suffix="core_op",
    )
    if onesided_node.data is None or onesided_node.data:
        # with length == window size
        # slice rank: dim - 1 by $onesided_max_dim
        onesided_max_idx = (n_fft_node.data >> 1) + 1
        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=output_nnef_tensor,
            output_tensor_name_suffix="pre_cast_back",
            attrs={
                "axes": [output_nnef_tensor.rank - 1],
                "begin": [0],
                "end": [onesided_max_idx],
                "stride": [1],
            },
        )

    transposed_axes = list(range(len(output_nnef_tensor.shape)))
    # permute to follow numpy way of things (as well as tract)
    transposed_axes[dim], transposed_axes[dim + 1] = (
        transposed_axes[dim + 1],
        transposed_axes[dim],
    )
    suffix_outname = ""
    if normalized_node.data:
        suffix_outname = "_prenorm"
    output_nnef_tensor = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=output_nnef_tensor,
        attrs={"axes": transposed_axes},
        pass_quantization_params=True,
        output_tensor_name_suffix=suffix_outname,
    )

    if normalized_node.data:
        # multiplied by (frame_length)−0.5
        multiplier = get_or_add_tensor_variable_in_nnef(
            g,
            PythonConstant(
                name=f"{output_nnef_tensor.name}_frame_length", data=frame**-0.5
            ),
            name_to_tensor,
        )
        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "mul",
            inputs=(output_nnef_tensor, multiplier),
            pass_quantization_params=True,
        )

    return ["tract_core"]


@OP_REGISTRY.register()
def fft_fft(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:fft_fft' to NNEF."""
    return _fft(
        node,
        g,
        name_to_tensor,
        inverse=False,
        inference_target=inference_target,
    )


@OP_REGISTRY.register()
def fft_ifft(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:fft_ifft' to NNEF."""
    return _fft(
        node,
        g,
        name_to_tensor,
        inverse=True,
        inference_target=inference_target,
    )


def _real_to_complex_pad(
    g, node, name_to_tensor, input_node, nnef_tensor, suffix_base
):
    """Stack a zero imaginary part onto a real tensor.

    Mirrors the t2n convention used by `_fft` / `stft`: complex is
    simulated as a real tensor with an extra trailing axis of size 2
    (`[real, imag]`).
    """
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=nnef_tensor,
        attrs={"axes": [pick_axis(input_node, -1) + 1]},
        pass_quantization_params=True,
        output_tensor_name_suffix=f"{suffix_base}_unsqueeze",
    )
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "pad",
        inputs=out,
        attrs={
            "padding": [(0, 0)] * input_node.rank + [(0, 1)],
            "value": 0.0,
        },
        output_tensor_name_suffix=f"{suffix_base}_pad",
    )
    return out


def _check_fft_target(inference_target):
    if (
        not isinstance(inference_target, TractNNEF)
        or inference_target.version < "0.20.7"
    ):
        raise T2NErrorNotImplemented(inference_target)


@OP_REGISTRY.register()
def fft_rfft(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:fft_rfft' to NNEF.

    Real input -> one-sided complex spectrum on `dim`. Mirrors
    `fft_fft` (pad to complex, run `tract_core_fft`) then slices the
    FFT axis to the first `N // 2 + 1` bins.
    """
    _check_fft_target(inference_target)
    input_node, n_node, dim_node, norm_node = node.inputs
    if n_node.data is not None or norm_node.data is not None:
        raise T2NErrorNotImplemented("n or norm unexpected")
    if input_node.dtype not in [torch.float32, torch.float64]:
        raise T2NErrorNotImplemented(
            f"fft_rfft expects real input; got dtype={input_node.dtype}"
        )
    dim = pick_axis(input_node, dim_node.data)
    n_fft = input_node.shape[dim]
    onesided_max_idx = (n_fft >> 1) + 1

    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    cmplx = _real_to_complex_pad(
        g, node, name_to_tensor, input_node, nnef_tensor, "rfft_complex_cast"
    )
    full = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_fft",
        inputs=cmplx,
        attrs={"axis": dim, "inverse": False},
        output_tensor_name_suffix="rfft_full_spectrum",
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=full,
        attrs={
            "axes": [dim],
            "begin": [0],
            "end": [onesided_max_idx],
            "stride": [1],
        },
    )
    return ["tract_core"]


def _fftn_loop(
    g, node, name_to_tensor, inference_target, inverse, suffix_prefix
):
    """N-dimensional (i)FFT decomposed as a chain of `tract_core_fft`.

    The aten signature is `(input, s?, dim?, norm?)`. We only support
    `s=None` (no resizing) and `norm=None` (default backward
    semantics: forward applies no scaling, inverse divides by the
    product of FFT lengths).
    """
    input_node, s_node, dim_node, norm_node = node.inputs
    if s_node.data is not None:
        raise T2NErrorNotImplemented("s (per-axis sizing) not supported")
    if norm_node.data is not None:
        raise T2NErrorNotImplemented("norm unexpected")

    # Default `dim`: all real-tensor axes (the trailing complex axis is
    # added below and never gets transformed).
    raw_dims = dim_node.data
    if raw_dims is None:
        if input_node.dtype in [torch.complex64, torch.complex128]:
            # complex stored as [..., 2]; real-tensor rank is one less.
            real_rank = input_node.rank - 1
        else:
            real_rank = input_node.rank
        raw_dims = list(range(real_rank))
    dims = [pick_axis(input_node, d) for d in raw_dims]

    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if input_node.dtype in [torch.float32, torch.float64]:
        current = _real_to_complex_pad(
            g,
            node,
            name_to_tensor,
            input_node,
            nnef_tensor,
            f"{suffix_prefix}_complex_cast",
        )
    elif input_node.dtype in [torch.complex64, torch.complex128]:
        current = nnef_tensor
    else:
        raise T2NErrorNotImplemented(
            f"fftn expects real or complex input; got {input_node.dtype}"
        )

    # Chain one tract_core_fft per requested axis. For forward, the
    # last call writes `node.outputs[0]`. For inverse with backward
    # norm we route every FFT to an intermediate suffix and let the
    # division op produce the final output.
    last_idx = len(dims) - 1
    for i, dim in enumerate(dims):
        is_last = i == last_idx
        suffix = f"_axis{dim}" if inverse or not is_last else ""
        current = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_fft",
            inputs=current,
            attrs={"axis": dim, "inverse": inverse},
            output_tensor_name_suffix=suffix,
        )

    if inverse:
        # Backward norm: divide by the product of FFT lengths.
        if inference_target.has_dynamic_axes:
            raise T2NErrorNotImplemented(
                "fft_ifftn under dynamic axes needs runtime shape-of"
            )
        divisor_value = 1
        for d in dims:
            divisor_value *= input_node.shape[d]
        divisor_tensor = get_or_add_tensor_variable_in_nnef(
            g,
            PythonConstant(
                name=f"{node.outputs[0].export_name}_fftn_divisor",
                data=float(divisor_value),
            ),
            name_to_tensor,
        )
        node.outputs[0].dtype = torch.complex64
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "div",
            inputs=(current, divisor_tensor),
        )

    return ["tract_core"]


@OP_REGISTRY.register()
def fft_irfft(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:fft_irfft' to NNEF.

    One-sided complex spectrum -> real signal of length `n` (defaults
    to `2 * (K - 1)` where `K = input.shape[dim]`). Reconstruct the
    Hermitian-symmetric full spectrum: take the slice `[1, K-1)`,
    reverse it on `dim`, conjugate (negate imag), concat after the
    input -> `(..., n, 2)`; run an inverse FFT; divide by `n`; drop
    the imaginary part.
    """
    _check_fft_target(inference_target)
    input_node, n_node, dim_node, norm_node = node.inputs
    if norm_node.data is not None:
        raise T2NErrorNotImplemented("norm unexpected")
    if input_node.dtype not in (torch.complex64, torch.complex128):
        raise T2NErrorNotImplemented(
            f"fft_irfft expects complex input; got dtype={input_node.dtype}"
        )

    # `input_node.rank` is the PyTorch (complex) rank; NNEF emission
    # adds a trailing-2 axis at position `input_node.rank`. The FFT
    # axis sits inside the complex view, so `pick_axis` works against
    # the PyTorch rank as usual.
    dim = pick_axis(input_node, dim_node.data)
    complex_axis = input_node.rank
    k = input_node.shape[dim]
    if not isinstance(k, int):
        raise T2NErrorNotImplemented("fft_irfft on dynamic FFT axis")
    if k < 2:
        raise T2NErrorNotImplemented(
            f"fft_irfft needs `input.shape[dim] >= 2`; got {k}"
        )
    n = n_node.data if n_node.data is not None else 2 * (k - 1)

    inp = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)

    # 1. Mirror chunk: slice `[1, K-1)` on the FFT axis -> (..., K-2, 2).
    if k == 2:
        # The Hermitian mirror is empty; the input itself is the full
        # 2-bin spectrum. Skip the slice / flip / concat.
        full_spec = inp
    else:
        mirror_src = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=inp,
            attrs={
                "axes": [dim],
                "begin": [1],
                "end": [k - 1],
                "stride": [1],
            },
            output_tensor_name_suffix="_irfft_mirror_src",
        )

        # 2. Reverse on the FFT axis: `tract_core_gather` with the
        # constant `[K-3, ..., 0]` index tensor.
        idx_const = PythonConstant(
            name=f"{node.outputs[0].export_name}_irfft_flip_idx",
            data=torch.arange(k - 3, -1, -1, dtype=torch.int64),
        )
        idx_ref = get_or_add_tensor_variable_in_nnef(
            g, idx_const, name_to_tensor
        )
        flipped = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_gather",
            inputs=[mirror_src, idx_ref],
            attrs={"axis": dim},
            force_consistent_inputs_shapes=False,
            output_tensor_name_suffix="_irfft_flipped",
        )

        # 3. Conjugate: keep real, negate imag, concat back.
        real_part = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=flipped,
            attrs={
                "axes": [complex_axis],
                "begin": [0],
                "end": [1],
                "stride": [1],
            },
            output_tensor_name_suffix="_irfft_re",
        )
        imag_part = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=flipped,
            attrs={
                "axes": [complex_axis],
                "begin": [1],
                "end": [2],
                "stride": [1],
            },
            output_tensor_name_suffix="_irfft_im",
        )
        neg1_const = PythonConstant(
            name=f"{node.outputs[0].export_name}_irfft_neg1",
            data=torch.tensor(-1.0, dtype=torch.float32),
        )
        neg1_ref = get_or_add_tensor_variable_in_nnef(
            g, neg1_const, name_to_tensor
        )
        neg_imag = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "mul",
            inputs=[imag_part, neg1_ref],
            output_tensor_name_suffix="_irfft_neg_im",
        )
        conj = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "concat",
            inputs=[real_part, neg_imag],
            attrs={"axis": complex_axis},
            ensure_tuple=False,
            output_tensor_name_suffix="_irfft_conj",
        )

        # 4. Concat input with conjugate mirror on the FFT axis
        #    -> (..., n, 2).
        full_spec = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "concat",
            inputs=[inp, conj],
            attrs={"axis": dim},
            ensure_tuple=False,
            output_tensor_name_suffix="_irfft_full_spec",
        )

    # 5. Inverse FFT on the FFT axis.
    ifft = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_fft",
        inputs=full_spec,
        attrs={"axis": dim, "inverse": True},
        output_tensor_name_suffix="_irfft_ifft",
    )

    # 6. Divide by `n` (default backward norm for irfft).
    n_const = PythonConstant(
        name=f"{node.outputs[0].export_name}_irfft_divisor",
        data=torch.tensor(float(n), dtype=torch.float32),
    )
    n_ref = get_or_add_tensor_variable_in_nnef(g, n_const, name_to_tensor)
    scaled = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "div",
        inputs=[ifft, n_ref],
        output_tensor_name_suffix="_irfft_scaled",
    )

    # 7. Drop the imaginary part: slice last axis `[0:1]` then squeeze.
    real_only = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "slice",
        inputs=scaled,
        attrs={
            "axes": [complex_axis],
            "begin": [0],
            "end": [1],
            "stride": [1],
        },
        output_tensor_name_suffix="_irfft_real_slice",
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=real_only,
        attrs={"axes": [complex_axis]},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def fft_fftn(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:fft_fftn' to NNEF (forward N-dim FFT)."""
    _check_fft_target(inference_target)
    return _fftn_loop(
        g,
        node,
        name_to_tensor,
        inference_target,
        inverse=False,
        suffix_prefix="fftn",
    )


@OP_REGISTRY.register()
def fft_ifftn(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:fft_ifftn' to NNEF (inverse N-dim FFT)."""
    _check_fft_target(inference_target)
    return _fftn_loop(
        g,
        node,
        name_to_tensor,
        inference_target,
        inverse=True,
        suffix_prefix="ifftn",
    )
