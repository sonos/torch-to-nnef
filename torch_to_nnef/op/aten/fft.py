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


def _is_view_tagged_complex(input_node) -> bool:
    """True iff `input_node` is the trailing-2 / dtype-tagged complex form.

    t2n's chosen convention for complex tensors is *view-tagged*: a
    PyTorch tensor of logical complex rank N is represented in the IR
    as rank N+1 with the trailing axis of size 2 carrying the
    `(real, imag)` pair, and `dtype` marked `complex64` / `complex128`.
    Real tensors keep their PyTorch rank and a real dtype.

    `view_as_complex`, `complex`, `polar`, and every FFT output (after
    this module's normalisation) all produce view-tagged complex; the
    consumer handlers (`fft_irfft`, `angle`, `conj`, `real`/`imag`,
    `sgn`, `view_as_real`) all read `rank - 1` for the complex axis.
    Keeping the convention uniform across t2n is what lets chains like
    `view_as_complex(x).fft.irfft(...)` work with no special-casing.
    """
    # `rank >= 2` is load-bearing: a view-tagged complex tensor carries
    # the trailing-2 axis *on top of* at least one logical axis, so its
    # IR rank is at least 2. A rank-1 complex tensor with shape `[2]`
    # (which a rank-1 `rfft` of a length-2 signal produces) is logically
    # rank-1 with 2 complex bins, *not* view-tagged.
    return (
        input_node.dtype in (torch.complex64, torch.complex128)
        and input_node.rank >= 2
        and isinstance(input_node.shape, list)
        and input_node.shape[-1] == 2
    )


def _pick_logical_axis(input_node, raw_dim) -> int:
    """Resolve a (possibly negative) PyTorch dim against the *logical* rank.

    For view-tagged complex inputs the logical rank is `IR_rank - 1`
    (the trailing-2 axis is the complex pair, not a logical axis); for
    real inputs the logical rank equals the IR rank.
    """
    logical_rank = (
        input_node.rank - 1
        if _is_view_tagged_complex(input_node)
        else input_node.rank
    )
    if raw_dim is None:
        raw_dim = -1
    if raw_dim < 0:
        raw_dim += logical_rank
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
    # pylint: disable=too-many-branches
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
    # Promote the IR output to the view-tagged complex convention used
    # across t2n (IR rank = logical_rank + 1, trailing-2 axis carrying
    # the (real, imag) pair). PyTorch's tracer reports `aten::stft` as
    # rank N complex; the NNEF emission chain produces rank N+1 real
    # storage. This update must happen before the final
    # `add_single_output_op` because the NNEF tensor registered under
    # `node.outputs[0].export_name` takes its shape from
    # `node.outputs[0].shape` at registration time.
    if node.outputs[0].shape[-1] != 2:
        node.outputs[0].shape = list(node.outputs[0].shape) + [2]

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
    # Promote the IR output to the t2n view-tagged complex convention:
    # rank = logical_rank + 1, last axis = 2 (the (real, imag) pair).
    # Without this, an rfft output whose logical bin-count happens to be
    # 2 is indistinguishable from a view-tagged complex (both have
    # `shape[-1] == 2`) and `_is_view_tagged_complex` would misclassify
    # downstream (e.g. `fft.irfft(fft.rfft(...))` on a length-2 axis).
    node.outputs[0].shape = list(node.outputs[0].shape) + [2]
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
        if _is_view_tagged_complex(input_node):
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

    # Two IR layouts arrive here for a complex tensor (see
    # `_is_view_tagged_complex` for the full discussion). The view-tagged
    # form (`view_as_complex` / `complex` / `polar`) keeps the trailing-2
    # axis in the IR shape; the logical form (`fft.fft` / `fft.rfft` /
    # `fft.ifft` / `fft.fftn` / `fft.ifftn` / `stft`) keeps the PyTorch
    # logical rank. The handler dispatches against the heuristic; the
    # only fragile case (a logical complex tensor whose FFT axis happens
    # to be 2) is implausible in real models: FFT axes are practically
    # never 2.
    if _is_view_tagged_complex(input_node):
        # View-tagged: trailing-2 already in IR shape.
        # logical rank = IR rank - 1, complex axis sits at IR rank - 1.
        dim = _pick_logical_axis(input_node, dim_node.data)
        complex_axis = input_node.rank - 1
    else:
        # Logical: NNEF emission upstream adds the trailing-2 axis at
        # position `input_node.rank` in the NNEF tensor. Pick the FFT
        # dim directly from the input rank, and set the complex axis
        # to the new last axis that NNEF appends.
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
        #    -> (..., n, 2). Force `force_consistent_inputs_shapes=False`:
        #    every intermediate NNEF tensor we created above carries
        #    `node.outputs[0]`'s shape (the irfft's final-output rank,
        #    which is the *real* signal rank without the trailing-2),
        #    so the auto-aligner's rank comparison against `inp` (the
        #    full view-tagged complex tensor) sees an artificial
        #    mismatch and prepends an extra leading-1. The actual NNEF
        #    op semantics are correct; we just need to keep the helper
        #    from second-guessing them.
        full_spec = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "concat",
            inputs=[inp, conj],
            attrs={"axis": dim},
            ensure_tuple=False,
            force_consistent_inputs_shapes=False,
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


@OP_REGISTRY.register()
def istft(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:istft' to NNEF.

    Implements `torch.istft` for the common case (onesided=True,
    normalized=False, return_complex=False, length=None, static shapes).
    Decomposition:

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
    # pylint: disable=too-many-branches,too-many-statements
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
    normalized = False if normalized_node.data is None else normalized_node.data
    onesided = True if onesided_node.data is None else onesided_node.data
    length = length_node.data
    return_complex = (
        False if return_complex_node.data is None else return_complex_node.data
    )
    if not onesided:
        raise T2NErrorNotImplemented("istft onesided=False not supported")
    if normalized:
        raise T2NErrorNotImplemented("istft normalized=True not supported")
    if return_complex:
        raise T2NErrorNotImplemented("istft return_complex=True not supported")
    if length is not None:
        raise T2NErrorNotImplemented("istft length= argument not supported")
    if not _is_view_tagged_complex(input_node):
        raise T2NErrorNotImplemented(
            "istft expects view-tagged complex input (rank N+1, trailing 2)"
        )
    if input_node.rank < 3:
        raise T2NErrorNotImplemented(
            "istft input must have at least (freq, T_frames, 2); got rank="
            f"{input_node.rank}"
        )

    # Storage layout for view-tagged complex (..., freq, T_frames, 2):
    freq_axis = input_node.rank - 3
    frames_axis = input_node.rank - 2
    complex_axis = input_node.rank - 1
    freq_size = input_node.shape[freq_axis]
    frames_size = input_node.shape[frames_axis]
    onesided_expected = (n_fft >> 1) + 1
    if freq_size != onesided_expected:
        raise T2NErrorNotImplemented(
            f"istft input freq size {freq_size} != onesided expected "
            f"{onesided_expected}"
        )

    inp = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    out_name = node.outputs[0].export_name

    def _emit(op_type, *, inputs, attrs=None, suffix, new_shape, **kwargs):
        """Wrap `add_single_output_op` and patch the recorded NNEF.

        shape on the intermediate. Intermediates emitted via
        `add_single_output_op` otherwise inherit `node.outputs[0].shape`
        (the istft's final audio shape), which confuses tract's
        rank-alignment when later ops broadcast.
        """
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

    # Leading batch dims, before (freq, T_frames, 2):
    leading = tuple(input_node.shape[:-3])

    # 1. Hermitian-symmetric full spectrum on the freq axis. Same pattern
    #    as `fft_irfft`: slice [1, K-1), gather-flip, conjugate, concat
    #    after the input. K = freq_size; the mirror has size K-2.
    k = freq_size
    if k == 2:
        full_spec = inp
    else:
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
        idx_ref = get_or_add_tensor_variable_in_nnef(
            g, idx_const, name_to_tensor
        )
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
        neg1_ref = get_or_add_tensor_variable_in_nnef(
            g, neg1_const, name_to_tensor
        )
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
        full_spec = _emit(
            "concat",
            inputs=[inp, conj],
            attrs={"axis": freq_axis},
            ensure_tuple=False,
            suffix="_istft_full_spec",
            new_shape=leading + (n_fft, frames_size, 2),
            force_consistent_inputs_shapes=False,
        )

    # 2. Inverse FFT on the freq axis.
    ifft = _emit(
        "tract_core_fft",
        inputs=full_spec,
        attrs={"axis": freq_axis, "inverse": True},
        suffix="_istft_ifft",
        new_shape=leading + (n_fft, frames_size, 2),
    )

    # 3. Divide by n_fft (backward norm).
    n_const = PythonConstant(
        name=f"{out_name}_istft_n_div",
        data=torch.tensor(float(n_fft), dtype=torch.float32),
    )
    n_ref = get_or_add_tensor_variable_in_nnef(g, n_const, name_to_tensor)
    scaled = _emit(
        "div",
        inputs=[ifft, n_ref],
        suffix="_istft_scaled",
        new_shape=leading + (n_fft, frames_size, 2),
    )

    # 4. Take the real part: slice complex axis [0:1], then squeeze.
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

    # 5. Window multiply. The window broadcasts on the (now collapsed) freq
    #    axis: shape (n_fft,) broadcasts to (..., n_fft, T_frames) by inserting
    #    an axis at the end. Pre-shape it to (n_fft, 1) so the broadcast is
    #    unambiguous against (..., n_fft, T_frames).
    if window_node.data is None:
        window_node.set_data(
            torch.ones(win_len, dtype=torch.float32), force_shape=True
        )
    # The user-provided window may have win_length < n_fft; pad to n_fft.
    win_data = window_node.data
    if win_data.shape[0] != n_fft:
        pad_left = (n_fft - win_data.shape[0]) // 2
        pad_right = n_fft - win_data.shape[0] - pad_left
        win_data = torch.nn.functional.pad(
            win_data.float(), (pad_left, pad_right), value=0.0
        )
    # Reshape window to (1, n_fft, 1) so NNEF's left-aligned broadcast
    # against the frames tensor (B, n_fft, T_frames) lines the window
    # up on the n_fft axis. (Without the explicit leading 1, tract
    # prepends 1s to (n_fft, 1) -> (320, 1, 1) and broadcasts the
    # frames out across the wrong dim.)
    win_const = PythonConstant(
        name=f"{out_name}_istft_window",
        data=win_data.reshape(1, n_fft, 1).float(),
    )
    win_ref = get_or_add_tensor_variable_in_nnef(g, win_const, name_to_tensor)
    windowed = _emit(
        "mul",
        inputs=[re_squeeze, win_ref],
        suffix="_istft_windowed",
        new_shape=leading + (n_fft, frames_size),
    )

    # 6. Overlap-add via deconv with an identity kernel (1, n_fft, n_fft).
    #    Input to deconv must be (B, C_in=n_fft, T_frames); if the
    #    original istft input has no batch axis (logical rank 2: just
    #    (freq, T_frames)), we add one. IR rank with the trailing-2
    #    complex axis: logical-rank 2 -> IR rank 3.
    if input_node.rank == 3:
        ola_in = _emit(
            "unsqueeze",
            inputs=windowed,
            attrs={"axes": [0]},
            suffix="_istft_ola_in",
            new_shape=(1, n_fft, frames_size),
        )
        batched_added = True
    else:
        ola_in = windowed
        batched_added = False
    # NNEF deconv kernel layout is (out_channels, in_channels, kernel_size);
    # identity OLA: each input channel i pastes a unit pulse at output
    # offset i. So kernel[0, i, j] = delta(i, j) -> torch.eye(n_fft) with
    # a leading singleton out_channels axis.
    eye_kernel = torch.eye(n_fft, dtype=torch.float32).unsqueeze(
        0
    )  # (1, n_fft, n_fft)
    kernel_const = PythonConstant(
        name=f"{out_name}_istft_ola_kernel",
        data=eye_kernel,
    )
    kernel_ref = get_or_add_tensor_variable_in_nnef(
        g, kernel_const, name_to_tensor
    )
    bias_const = PythonConstant(
        name=f"{out_name}_istft_ola_bias",
        data=torch.zeros(1, dtype=torch.float32),
    )
    bias_ref = get_or_add_tensor_variable_in_nnef(g, bias_const, name_to_tensor)
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
        new_shape=(1, 1, audio_len_raw),
        force_consistent_inputs_shapes=False,
    )

    # 7. Divide by the OLA of window**2 to undo the per-sample window-weight
    #    accumulation. For static shapes, precompute the full-length divisor
    #    offline. For dynamic axes, fall back to a scalar COLA constant
    #    computed from the window: in the central region of a long signal
    #    with `hop <= n_fft`, the window^2 OLA settles to a constant value
    #    that depends only on the window itself (not on the signal length).
    #    Boundary samples are dropped by the center=True crop anyway, and
    #    pulse-mode warm-up further hides any early divergence.
    win_np = win_data.numpy()
    if inference_target.has_dynamic_axes:
        # Compute the central COLA constant over a long-enough signal.
        probe_len = max(n_fft * 8, audio_len_raw)
        probe_ola = torch.zeros(probe_len, dtype=torch.float32)
        win_sq = torch.tensor(win_np * win_np, dtype=torch.float32)
        n_probe_frames = (probe_len - n_fft) // hop + 1
        for i in range(n_probe_frames):
            probe_ola[i * hop : i * hop + n_fft] += win_sq
        cola_const = float(probe_ola[probe_len // 2].item())
        norm_const = PythonConstant(
            name=f"{out_name}_istft_cola_const",
            data=torch.tensor(max(cola_const, 1e-11), dtype=torch.float32),
        )
        norm_ref = get_or_add_tensor_variable_in_nnef(
            g, norm_const, name_to_tensor
        )
        normed = _emit(
            "div",
            inputs=[ola, norm_ref],
            suffix="_istft_normed",
            new_shape=(1, 1, audio_len_raw)
            if isinstance(audio_len_raw, int)
            else None,
        )
    else:
        window_sq_ola = torch.zeros(audio_len_raw, dtype=torch.float32)
        win_sq = torch.tensor(win_np * win_np, dtype=torch.float32)
        for i in range(frames_size):
            window_sq_ola[i * hop : i * hop + n_fft] += win_sq
        # Clamp to avoid /0 at the boundaries.
        window_sq_ola = torch.clamp(window_sq_ola, min=1e-11)
        norm_const = PythonConstant(
            name=f"{out_name}_istft_win_sq_ola",
            data=window_sq_ola.reshape(1, 1, audio_len_raw),
        )
        norm_ref = get_or_add_tensor_variable_in_nnef(
            g, norm_const, name_to_tensor
        )
        normed = _emit(
            "div",
            inputs=[ola, norm_ref],
            suffix="_istft_normed",
            new_shape=(1, 1, audio_len_raw),
        )

    # 8. Squeeze the channel axis (it was 1) and apply the optional
    #    center crop. The final emit (the one that should land on
    #    `node.outputs[0]`) is suffix-less; earlier intermediates carry
    #    suffixes.
    do_crop = center
    do_final_squeeze = batched_added  # squeeze the synthetic batch back

    if do_crop or do_final_squeeze:
        channel_squeezed = _emit(
            "squeeze",
            inputs=normed,
            attrs={"axes": [1]},
            suffix="_istft_audio_raw",
            new_shape=(1, audio_len_raw),
        )
    else:
        # No further ops downstream; this squeeze writes to the IR output.
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=normed,
            attrs={"axes": [1]},
        )
        return ["tract_core"]

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
                new_shape=(1, final_len),
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
            return ["tract_core"]
    else:
        cropped = channel_squeezed

    # do_final_squeeze branch: peel off the synthetic batch axis.
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=cropped,
        attrs={"axes": [0]},
    )
    return ["tract_core"]
