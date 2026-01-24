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
