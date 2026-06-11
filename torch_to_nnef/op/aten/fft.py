"""Aten FFT/STFT/ISTFT handlers mapping to NNEF ops.

Contains helpers for complex layout handling and tract-core FFT glue.
"""

import logging

import nnef
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
)
from torch_to_nnef.torch_graph import PythonConstant
from torch_to_nnef.utils import torch_version

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


def _logical_rank(input_node) -> int:
    """Logical (PyTorch-visible) rank of a t2n IR tensor.

    Complex tensors are uniformly view-tagged (see
    `TorchToNGraphExtractor.build_nnef_graph`): the IR rank is one
    more than the logical rank, with the trailing axis carrying re/im.

    Use this when you need the *count* of logical axes -- e.g., to
    build a default `dim=list(range(...))` for `fftn`'s no-`dim` path.
    For resolving a *single* (possibly negative) PyTorch dim to a
    storage axis index, use `_pick_logical_axis` here or
    `op.helper.pick_axis`; both adjust for the trailing complex axis,
    and they differ only in scope (`pick_axis` is the generic-op entry
    point and also handles `FixedTensorList` inputs).
    """
    if input_node.dtype in _COMPLEX_DTYPES:
        return input_node.rank - 1
    return input_node.rank


def _pick_logical_axis(input_node, raw_dim) -> int:
    """Resolve a (possibly negative) PyTorch dim against the logical rank."""
    if raw_dim is None:
        raw_dim = -1
    if raw_dim < 0:
        raw_dim += _logical_rank(input_node)
    return raw_dim


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

    # Resolve `dim` against the logical rank. `_pick_logical_axis`
    # handles both real and view-tagged complex inputs: for complex it
    # strips the trailing complex axis from the logical rank so a
    # negative dim (e.g. `-1`) doesn't land on the (re, imag) axis.
    dim = _pick_logical_axis(input_node, dim_node.data)

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
        # Slice the n_fft axis to keep only the one-sided spectrum.
        # tract_core_stft inserts the frame axis at `dim + 1`, so the
        # n_fft axis sits there; the trailing axis (== rank - 1) is the
        # complex (re/im) pair and must not be sliced.
        onesided_max_idx = (n_fft_node.data >> 1) + 1
        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=output_nnef_tensor,
            output_tensor_name_suffix="pre_cast_back",
            attrs={
                "axes": [dim + 1],
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

    # Default `dim`: all logical (non-complex) axes. For view-tagged
    # complex inputs the trailing-2 axis is not an FFT axis.
    raw_dims = dim_node.data
    if raw_dims is None:
        raw_dims = list(range(_logical_rank(input_node)))
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

    # After `build_nnef_graph`'s pre-pass every complex IR tensor is
    # view-tagged: storage rank = logical_rank + 1, trailing axis = 2.
    dim = _pick_logical_axis(input_node, dim_node.data)
    complex_axis = input_node.rank - 1
    inp = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    k = input_node.shape[dim]
    if not isinstance(k, int):
        raise T2NErrorNotImplemented("fft_irfft on dynamic FFT axis")
    if k < 2:
        raise T2NErrorNotImplemented(
            f"fft_irfft needs `input.shape[dim] >= 2`; got {k}"
        )
    n = n_node.data if n_node.data is not None else 2 * (k - 1)
    out_name = node.outputs[0].export_name
    _emit = _make_shape_overriding_emit(g, node, name_to_tensor)
    in_shape = tuple(input_node.shape)

    # 1. Mirror chunk: slice `[1, K-1)` on the FFT axis -> (..., K-2, 2).
    if k == 2:
        # The Hermitian mirror is empty; the input itself is the full
        # 2-bin spectrum. Skip the slice / flip / concat.
        full_spec = inp
    else:
        mirror_shape = _shape_with(in_shape, dim, k - 2)
        mirror_src = _emit(
            "slice",
            inputs=inp,
            attrs={
                "axes": [dim],
                "begin": [1],
                "end": [k - 1],
                "stride": [1],
            },
            suffix="_irfft_mirror_src",
            new_shape=mirror_shape,
        )

        # 2. Reverse on the FFT axis: `tract_core_gather` with the
        # constant `[K-3, ..., 0]` index tensor.
        idx_const = PythonConstant(
            name=f"{out_name}_irfft_flip_idx",
            data=torch.arange(k - 3, -1, -1, dtype=torch.int64),
        )
        idx_ref = get_or_add_tensor_variable_in_nnef(
            g, idx_const, name_to_tensor
        )
        flipped = _emit(
            "tract_core_gather",
            inputs=[mirror_src, idx_ref],
            attrs={"axis": dim},
            suffix="_irfft_flipped",
            new_shape=mirror_shape,
            # `mirror_src` (rank R) and `idx_ref` (rank 1) deliberately
            # have different ranks; suppress the auto rank-aligner.
            force_consistent_inputs_shapes=False,
        )

        # 3. Conjugate: keep real, negate imag, concat back.
        half_shape = _shape_with(mirror_shape, complex_axis, 1)
        real_part = _emit(
            "slice",
            inputs=flipped,
            attrs={
                "axes": [complex_axis],
                "begin": [0],
                "end": [1],
                "stride": [1],
            },
            suffix="_irfft_re",
            new_shape=half_shape,
        )
        imag_part = _emit(
            "slice",
            inputs=flipped,
            attrs={
                "axes": [complex_axis],
                "begin": [1],
                "end": [2],
                "stride": [1],
            },
            suffix="_irfft_im",
            new_shape=half_shape,
        )
        neg1_const = PythonConstant(
            name=f"{out_name}_irfft_neg1",
            data=torch.tensor(-1.0, dtype=torch.float32),
        )
        neg1_ref = get_or_add_tensor_variable_in_nnef(
            g, neg1_const, name_to_tensor
        )
        neg_imag = _emit(
            "mul",
            inputs=[imag_part, neg1_ref],
            suffix="_irfft_neg_im",
            new_shape=half_shape,
        )
        conj = _emit(
            "concat",
            inputs=[real_part, neg_imag],
            attrs={"axis": complex_axis},
            ensure_tuple=False,
            suffix="_irfft_conj",
            new_shape=mirror_shape,
        )

        # 4. Concat input with conjugate mirror on the FFT axis -> (..., n, 2).
        #    `force_consistent_inputs_shapes=False`: `inp` carries the
        #    complex dtype tag (view-tagged complex); `conj` came out of
        #    a `mul` with a real scalar and is tagged float. The auto
        #    rank-aligner's complex/real branch would then append a
        #    trailing-1 to `conj` and a leading-1 to `inp`, breaking the
        #    concat shape. Bypass it -- the actual storage ranks match.
        full_spec = _emit(
            "concat",
            inputs=[inp, conj],
            attrs={"axis": dim},
            ensure_tuple=False,
            suffix="_irfft_full_spec",
            new_shape=_shape_with(in_shape, dim, n),
            force_consistent_inputs_shapes=False,
        )

    # 5. Inverse FFT on the FFT axis.
    ifft_shape = _shape_with(in_shape, dim, n)
    ifft = _emit(
        "tract_core_fft",
        inputs=full_spec,
        attrs={"axis": dim, "inverse": True},
        suffix="_irfft_ifft",
        new_shape=ifft_shape,
    )

    # 6. Divide by `n` (default backward norm for irfft).
    n_const = PythonConstant(
        name=f"{out_name}_irfft_divisor",
        data=torch.tensor(float(n), dtype=torch.float32),
    )
    n_ref = get_or_add_tensor_variable_in_nnef(g, n_const, name_to_tensor)
    scaled = _emit(
        "div",
        inputs=[ifft, n_ref],
        suffix="_irfft_scaled",
        new_shape=ifft_shape,
    )

    # 7. Drop the imaginary part: slice last axis `[0:1]` then squeeze.
    real_only = _emit(
        "slice",
        inputs=scaled,
        attrs={
            "axes": [complex_axis],
            "begin": [0],
            "end": [1],
            "stride": [1],
        },
        suffix="_irfft_real_slice",
        new_shape=_shape_with(ifft_shape, complex_axis, 1),
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


def _freq_handler(op_name, torch_fn, g, node, name_to_tensor):
    """Shared body for `fft_fftfreq` / `fft_rfftfreq`.

    Signature: `(n, d, dtype, layout, device, pin_memory)`. The result
    is fully determined by `n` and `d` at trace time, so bake the
    vector via `torch_fn` and register it as a NNEF constant.
    """
    n_node = node.inputs[0]
    d_node = node.inputs[1] if len(node.inputs) >= 2 else None
    if not (
        isinstance(n_node, PythonConstant) and isinstance(n_node.data, int)
    ):
        raise T2NErrorNotImplemented(
            f"aten::{op_name}: dynamic n not supported"
        )
    d_val = 1.0
    if (
        d_node is not None
        and isinstance(d_node, PythonConstant)
        and isinstance(d_node.data, (int, float))
    ):
        d_val = float(d_node.data)
    onode = node.outputs[0]
    out_dtype = onode.dtype or torch.float32
    onode.set_data(
        torch_fn(n_node.data, d=d_val, dtype=out_dtype),
        force_dtype=True,
        force_shape=True,
    )
    add_tensor_variable_node_as_nnef_tensor(g, onode, name_to_tensor)


@OP_REGISTRY.register()
def fft_fftfreq(g, node, name_to_tensor, **kwargs):
    """Map `aten::fft_fftfreq(n, d, ...)` to a NNEF constant.

    `torch.fft.fftfreq(n, d)` is fully determined by `n` and `d`; the
    common trace-time path already sees both as Python constants.
    """
    _freq_handler("fft_fftfreq", torch.fft.fftfreq, g, node, name_to_tensor)


@OP_REGISTRY.register()
def fft_rfftfreq(g, node, name_to_tensor, **kwargs):
    """Map `aten::fft_rfftfreq(n, d, ...)` to a NNEF constant."""
    _freq_handler("fft_rfftfreq", torch.fft.rfftfreq, g, node, name_to_tensor)


def _make_shape_overriding_emit(g, node, name_to_tensor):
    """Build an `_emit` helper that overrides each intermediate's NNEF shape.

    Multi-step decompositions (`fft_irfft`, `istft`) create many
    intermediate NNEF tensors that all inherit `node.outputs[0].shape`
    (the *final* output shape) from `add_single_output_op`. That
    inherited shape confuses tract's rank-alignment on subsequent
    broadcasts. This factory wraps `add_single_output_op` so each
    caller supplies the actual intermediate storage shape.
    """

    def _emit(op_type, *, inputs, attrs=None, suffix, new_shape, **kwargs):
        out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            op_type,
            inputs=inputs,
            attrs=attrs or {},
            output_tensor_name_suffix=suffix,
            **kwargs,
        )
        if new_shape is not None:
            out.shape = tuple(new_shape)
        return out

    return _emit


def _shape_with(shape, axis, new_size):
    """Return `shape` with `shape[axis]` replaced by `new_size`."""
    return tuple(shape[:axis]) + (new_size,) + tuple(shape[axis + 1 :])


def _istft_parse_args(node):
    """Validate `aten::istft` arguments and return the supported subset."""
    (
        input_node,
        n_fft_node,
        hop_node,
        win_len_node,
        window_node,
        center_node,
        normalized_node,
        onesided_node,
        length_node,
        return_complex_node,
    ) = node.inputs
    n_fft = n_fft_node.data
    hop = hop_node.data if hop_node.data is not None else n_fft // 4
    win_len = win_len_node.data if win_len_node.data is not None else n_fft
    center = True if center_node.data is None else center_node.data
    if (
        (onesided_node.data is not None and not onesided_node.data)
        or (normalized_node.data is True)
        or (return_complex_node.data is True)
        or (length_node.data is not None)
    ):
        raise T2NErrorNotImplemented(
            "istft: only (onesided=True, normalized=False, "
            "return_complex=False, length=None) is supported"
        )
    if input_node.dtype not in _COMPLEX_DTYPES:
        raise T2NErrorNotImplemented(
            f"istft expects complex input; got dtype={input_node.dtype}"
        )
    if input_node.rank < 3 or input_node.rank > 4:
        raise T2NErrorNotImplemented(
            "istft input must be view-tagged complex of rank 3 "
            "(freq, T_frames, 2) or 4 (B, freq, T_frames, 2); got rank="
            f"{input_node.rank}"
        )
    freq_size = input_node.shape[input_node.rank - 3]
    onesided_expected = (n_fft >> 1) + 1
    if freq_size != onesided_expected:
        raise T2NErrorNotImplemented(
            f"istft input freq size {freq_size} != onesided expected "
            f"{onesided_expected}"
        )
    return input_node, window_node, n_fft, hop, win_len, center


def _istft_hermitian_full_spec(
    g, node, name_to_tensor, _emit, *, inp, input_node, n_fft
):
    """Reconstruct the full Hermitian-symmetric spectrum on the freq axis.

    Mirrors `fft_irfft`: slice `[1, K-1)`, gather-flip on the freq axis,
    conjugate, concat the conjugate mirror after the input. Skips the
    chain when K=2 (mirror would be empty).
    """
    freq_axis = input_node.rank - 3
    frames_axis = input_node.rank - 2
    complex_axis = input_node.rank - 1
    frames_size = input_node.shape[frames_axis]
    k = input_node.shape[freq_axis]
    leading = tuple(input_node.shape[:-3])
    out_name = node.outputs[0].export_name

    if k == 2:
        return inp

    mirror_src = _emit(
        "slice",
        inputs=inp,
        attrs={
            "axes": [freq_axis],
            "begin": [1],
            "end": [k - 1],
            "stride": [1],
        },
        suffix="_istft_mirror_src",
        new_shape=leading + (k - 2, frames_size, 2),
    )
    idx_const = PythonConstant(
        name=f"{out_name}_istft_flip_idx",
        data=torch.arange(k - 3, -1, -1, dtype=torch.int64),
    )
    idx_ref = get_or_add_tensor_variable_in_nnef(g, idx_const, name_to_tensor)
    flipped = _emit(
        "tract_core_gather",
        inputs=[mirror_src, idx_ref],
        attrs={"axis": freq_axis},
        suffix="_istft_flipped",
        new_shape=leading + (k - 2, frames_size, 2),
        force_consistent_inputs_shapes=False,
    )
    re_part = _emit(
        "slice",
        inputs=flipped,
        attrs={
            "axes": [complex_axis],
            "begin": [0],
            "end": [1],
            "stride": [1],
        },
        suffix="_istft_re",
        new_shape=leading + (k - 2, frames_size, 1),
    )
    im_part = _emit(
        "slice",
        inputs=flipped,
        attrs={
            "axes": [complex_axis],
            "begin": [1],
            "end": [2],
            "stride": [1],
        },
        suffix="_istft_im",
        new_shape=leading + (k - 2, frames_size, 1),
    )
    neg1_const = PythonConstant(
        name=f"{out_name}_istft_neg1",
        data=torch.tensor(-1.0, dtype=torch.float32),
    )
    neg1_ref = get_or_add_tensor_variable_in_nnef(g, neg1_const, name_to_tensor)
    neg_im = _emit(
        "mul",
        inputs=[im_part, neg1_ref],
        suffix="_istft_neg_im",
        new_shape=leading + (k - 2, frames_size, 1),
    )
    conj = _emit(
        "concat",
        inputs=[re_part, neg_im],
        attrs={"axis": complex_axis},
        ensure_tuple=False,
        suffix="_istft_conj",
        new_shape=leading + (k - 2, frames_size, 2),
    )
    return _emit(
        "concat",
        inputs=[inp, conj],
        attrs={"axis": freq_axis},
        ensure_tuple=False,
        suffix="_istft_full_spec",
        new_shape=leading + (n_fft, frames_size, 2),
        force_consistent_inputs_shapes=False,
    )


def _istft_pad_window(win_data, n_fft):
    """Symmetric zero-pad `win_data` to length `n_fft` if shorter."""
    if win_data.shape[0] == n_fft:
        return win_data.float()
    pad_left = (n_fft - win_data.shape[0]) // 2
    pad_right = n_fft - win_data.shape[0] - pad_left
    return torch.nn.functional.pad(
        win_data.float(), (pad_left, pad_right), value=0.0
    )


def _istft_emit_ola_norm(
    g,
    node,
    name_to_tensor,
    _emit,
    *,
    ola,
    inference_target,
    win_data,
    n_fft,
    hop,
    frames_size,
    audio_len_raw,
    batch_size,
):
    """Divide the OLA tensor by the window^2 OLA (static or dynamic axes).

    Static: precompute the full-length divisor offline.

    Dynamic axes: use a scalar central-region COLA constant. This requires
    the `(window, hop)` pair to be COLA-satisfying (the window^2 OLA
    settles to a constant value across the central region). The export
    probes a long-enough signal, checks the central plateau is flat, and
    raises if not -- otherwise the emitted graph would silently produce
    wrong amplitudes. Boundary samples diverge by construction; they're
    dropped by the `center=True` crop and pulse-mode warm-up hides any
    early divergence.
    """
    out_name = node.outputs[0].export_name
    win_sq = win_data * win_data

    if inference_target.has_dynamic_axes:
        probe_len = max(n_fft * 8, audio_len_raw)
        probe_ola = torch.zeros(probe_len, dtype=torch.float32)
        n_probe_frames = (probe_len - n_fft) // hop + 1
        for i in range(n_probe_frames):
            probe_ola[i * hop : i * hop + n_fft] += win_sq
        # Verify the central region is constant (COLA condition). Probe a
        # quarter-length window centred on the middle so edge ramps don't
        # poison the check.
        c_lo = probe_len // 2 - probe_len // 8
        c_hi = probe_len // 2 + probe_len // 8
        central = probe_ola[c_lo:c_hi]
        cola_const = float(central[central.numel() // 2].item())
        if not torch.allclose(
            central,
            torch.full_like(central, cola_const),
            atol=1e-5,
            rtol=1e-4,
        ):
            raise T2NErrorNotImplemented(
                "istft under dynamic axes requires a window^2 OLA that "
                "is constant across the signal's central region (the "
                "divisor is baked at trace time as a single scalar). "
                f"Got per-sample range [{central.min().item():.3e}, "
                f"{central.max().item():.3e}] around mid={cola_const:.3e}. "
                "Use sqrt(hann_window), sqrt(hamming_window), or a "
                "vorbis window at hop = n_fft / 2 -- plain Hann/Hamming "
                "at that hop satisfy COLA only for `w`, not `w^2`."
            )
        norm_const = PythonConstant(
            name=f"{out_name}_istft_cola_const",
            data=torch.tensor(max(cola_const, 1e-11), dtype=torch.float32),
        )
    else:
        window_sq_ola = torch.zeros(audio_len_raw, dtype=torch.float32)
        for i in range(frames_size):
            window_sq_ola[i * hop : i * hop + n_fft] += win_sq
        # Clamp to avoid /0 at the boundaries.
        window_sq_ola = torch.clamp(window_sq_ola, min=1e-11)
        norm_const = PythonConstant(
            name=f"{out_name}_istft_win_sq_ola",
            data=window_sq_ola.reshape(1, 1, audio_len_raw),
        )

    norm_ref = get_or_add_tensor_variable_in_nnef(g, norm_const, name_to_tensor)
    # Divisor is rank-3 `(1, 1, audio_len_raw)`; under B>1 the leading-1
    # broadcasts against `(B, 1, audio_len_raw)` so we only need to label
    # the recorded shape with the right batch.
    return _emit(
        "div",
        inputs=[ola, norm_ref],
        suffix="_istft_normed",
        new_shape=(batch_size, 1, audio_len_raw),
    )


def _istft_emit_finalize(
    g,
    node,
    name_to_tensor,
    _emit,
    *,
    normed,
    batched_added,
    center,
    n_fft,
    audio_len_raw,
    batch_size,
):
    """Apply the optional center crop and squeeze the synthetic axes.

    The final emit (the one that writes `node.outputs[0]`) is the
    suffix-less call; every earlier intermediate carries a suffix.
    """
    do_crop = center
    do_final_squeeze = batched_added

    if not (do_crop or do_final_squeeze):
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=normed,
            attrs={"axes": [1]},
        )
        return

    channel_squeezed = _emit(
        "squeeze",
        inputs=normed,
        attrs={"axes": [1]},
        suffix="_istft_audio_raw",
        new_shape=(batch_size, audio_len_raw),
    )

    if do_crop:
        crop = n_fft // 2
        final_len = audio_len_raw - 2 * crop
        slice_attrs = {
            "axes": [1],
            "begin": [crop],
            "end": [crop + final_len],
            "stride": [1],
        }
        if do_final_squeeze:
            cropped = _emit(
                "slice",
                inputs=channel_squeezed,
                attrs=slice_attrs,
                suffix="_istft_cropped",
                new_shape=(batch_size, final_len),
            )
        else:
            add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=channel_squeezed,
                attrs=slice_attrs,
            )
            return
    else:
        cropped = channel_squeezed

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=cropped,
        attrs={"axes": [0]},
    )


@OP_REGISTRY.register()
def istft(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:istft' to NNEF.

    Implements `torch.istft` for the common case (onesided=True,
    normalized=False, return_complex=False, length=None). Decomposition:

      1. Build the Hermitian-symmetric full spectrum on the freq axis.
      2. Inverse FFT on the freq axis, divide by n_fft.
      3. Take the real part (slice the complex axis to [0:1] then squeeze).
      4. Multiply by `window` (broadcast on the freq/time-of-frame axis).
      5. Overlap-add via `deconv` with an identity kernel of size
         (n_fft, 1, n_fft) and stride=`hop_length`.
      6. Divide by the precomputed OLA of `window**2` to undo the
         per-sample window-weight accumulation.
      7. If `center=True`, slice off `n_fft // 2` samples from each end.
    """
    _check_fft_target(inference_target)
    input_node, window_node, n_fft, hop, win_len, center = _istft_parse_args(
        node
    )
    freq_axis = input_node.rank - 3
    complex_axis = input_node.rank - 1
    frames_size = input_node.shape[input_node.rank - 2]
    leading = tuple(input_node.shape[:-3])
    inp = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    out_name = node.outputs[0].export_name
    _emit = _make_shape_overriding_emit(g, node, name_to_tensor)

    # 1. Hermitian-symmetric full spectrum on the freq axis.
    full_spec = _istft_hermitian_full_spec(
        g,
        node,
        name_to_tensor,
        _emit,
        inp=inp,
        input_node=input_node,
        n_fft=n_fft,
    )

    # 2. Inverse FFT + 3. divide by n_fft (backward norm).
    ifft = _emit(
        "tract_core_fft",
        inputs=full_spec,
        attrs={"axis": freq_axis, "inverse": True},
        suffix="_istft_ifft",
        new_shape=leading + (n_fft, frames_size, 2),
    )
    n_ref = get_or_add_tensor_variable_in_nnef(
        g,
        PythonConstant(
            name=f"{out_name}_istft_n_div",
            data=torch.tensor(float(n_fft), dtype=torch.float32),
        ),
        name_to_tensor,
    )
    scaled = _emit(
        "div",
        inputs=[ifft, n_ref],
        suffix="_istft_scaled",
        new_shape=leading + (n_fft, frames_size, 2),
    )

    # 4. Real part (slice complex axis [0:1] then squeeze).
    re_slice = _emit(
        "slice",
        inputs=scaled,
        attrs={"axes": [complex_axis], "begin": [0], "end": [1], "stride": [1]},
        suffix="_istft_real_slice",
        new_shape=leading + (n_fft, frames_size, 1),
    )
    re_squeeze = _emit(
        "squeeze",
        inputs=re_slice,
        attrs={"axes": [complex_axis]},
        suffix="_istft_frames",
        new_shape=leading + (n_fft, frames_size),
    )

    # 5. Window multiply. Shape the window so its rank matches
    #    `re_squeeze`'s rank, with the singleton on the time axis -- that
    #    way NNEF's `mul` doesn't pad either side and the auto-aligner
    #    doesn't inject a leading 1 we'd have to track. For rank-3 istft
    #    input the window is (n_fft, 1); for rank-4 it's (1, n_fft, 1).
    if window_node.data is None:
        window_node.set_data(
            torch.ones(win_len, dtype=torch.float32), force_shape=True
        )
    win_data = _istft_pad_window(window_node.data, n_fft)
    re_squeeze_rank = len(leading) + 2
    win_shape = (1,) * len(leading) + (n_fft, 1)
    win_ref = get_or_add_tensor_variable_in_nnef(
        g,
        PythonConstant(
            name=f"{out_name}_istft_window",
            data=win_data.reshape(*win_shape).float(),
        ),
        name_to_tensor,
    )
    windowed = _emit(
        "mul",
        inputs=[re_squeeze, win_ref],
        suffix="_istft_windowed",
        new_shape=leading + (n_fft, frames_size),
    )

    # 6. OLA via deconv with an identity kernel. The deconv expects
    #    (B, C_in=n_fft, T_frames); add a synthetic batch axis when the
    #    istft input has no leading dim (IR rank 3 = bare (freq, T, 2)).
    batched_added = re_squeeze_rank == 2
    # Batch axis that survives the deconv: 1 when we just synthesised it,
    # otherwise the original leading dim (rank-4 input).
    ola_batch_size = 1 if batched_added else int(leading[0])
    if batched_added:
        ola_in = _emit(
            "unsqueeze",
            inputs=windowed,
            attrs={"axes": [0]},
            suffix="_istft_ola_in",
            new_shape=(1, n_fft, frames_size),
        )
    else:
        ola_in = windowed
    kernel_ref = get_or_add_tensor_variable_in_nnef(
        g,
        PythonConstant(
            name=f"{out_name}_istft_ola_kernel",
            data=torch.eye(n_fft, dtype=torch.float32).unsqueeze(0),
        ),
        name_to_tensor,
    )
    bias_ref = get_or_add_tensor_variable_in_nnef(
        g,
        PythonConstant(
            name=f"{out_name}_istft_ola_bias",
            data=torch.zeros(1, dtype=torch.float32),
        ),
        name_to_tensor,
    )
    audio_len_raw = (frames_size - 1) * hop + n_fft
    ola = _emit(
        "deconv",
        inputs=[ola_in, kernel_ref, bias_ref],
        attrs={
            "dilation": [1],
            "padding": [(0, 0)],
            "stride": [hop],
            "groups": 1,
            "border": "constant",
        },
        suffix="_istft_ola",
        new_shape=(ola_batch_size, 1, audio_len_raw),
        force_consistent_inputs_shapes=False,
    )

    # 7. Window^2 OLA normalisation + 8. center crop / squeeze synth axes.
    normed = _istft_emit_ola_norm(
        g,
        node,
        name_to_tensor,
        _emit,
        ola=ola,
        inference_target=inference_target,
        win_data=win_data,
        n_fft=n_fft,
        hop=hop,
        frames_size=frames_size,
        audio_len_raw=audio_len_raw,
        batch_size=ola_batch_size,
    )
    _istft_emit_finalize(
        g,
        node,
        name_to_tensor,
        _emit,
        normed=normed,
        batched_added=batched_added,
        center=center,
        n_fft=n_fft,
        audio_len_raw=audio_len_raw,
        batch_size=ola_batch_size,
    )
    return ["tract_core"]
