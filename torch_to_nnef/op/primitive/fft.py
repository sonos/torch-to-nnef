import nnef
import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    OpRegistry,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
)
from torch_to_nnef.torch_graph import PythonConstant

OP_REGISTRY = OpRegistry()


def _fft(
    node,
    g,
    name_to_tensor,
    nnef_spec_strict,
    has_dynamic_axes,
    inverse=False,
    tract_feature_flags=None,
):
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L360
    # const Tensor& self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError("NNEF strict can not export FFT")
    input_node, n_node, dim_node, norm_node = node.inputs
    if n_node.data is not None or norm_node.data is not None:
        raise TorchToNNEFNotImplementedError("n or norm unexpected")

    dim = pick_rank(input_node, dim_node.data)

    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if input_node.dtype in [torch.float32, torch.float64]:
        """# sadly casting is not implemented in tract so we use another way
        casted_complex_input_tensor, _ = cast_to_if_not_dtype_and_variable(
            g,
            name_to_tensor,
            node,
            nnef_tensor=nnef_tensor,
            cast_to=np.complex64,
            suffix="precast_complex",
        )
        """
        output_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "unsqueeze",
            inputs=nnef_tensor,
            attrs={"axes": [pick_rank(input_node, -1) + 1]},
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
        if tract_feature_flags is not None and "complex" in tract_feature_flags:
            casted_complex_input_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_inner_dim_to_complex",
                inputs=output_nnef_tensor,
                pass_quantization_params=True,
                output_tensor_name_suffix="complex_cast",
            )
        else:
            casted_complex_input_tensor = output_nnef_tensor
    elif input_node.dtype not in [torch.complex64, torch.complex128]:
        raise TorchToNNEFNotImplementedError()
    else:
        casted_complex_input_tensor = nnef_tensor

    suffix = None
    if inverse and norm_node.data is None:
        # backward by default means 1/n
        suffix = "need_norm"
        norm_node.data = "backward"

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
        if has_dynamic_axes:
            raise TorchToNNEFNotImplementedError("Need to use implement")

        divisor_value = input_node.shape[dim]
        divisor_tensor = get_or_add_tensor_variable_in_nnef(
            g,
            PythonConstant(
                name=output_tensor.name + "_divisor",
                data=divisor_value,
            ),
            name_to_tensor,
        )

        # input_node, n_node, dim_node, norm_node = node.inputs
        if tract_feature_flags is not None and "complex" in tract_feature_flags:
            input_to_real_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_complex_to_inner_dim",
                inputs=output_tensor,
                output_tensor_name_suffix="cast_pre_norm_div",
            )
        else:
            input_to_real_tensor = output_tensor

        if tract_feature_flags is not None and "complex" in tract_feature_flags:
            real_normed_tensor_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "div",
                inputs=(
                    input_to_real_tensor,
                    divisor_tensor,
                ),
                output_tensor_name_suffix="norm_div",
            )
            # retransform to complex
            add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_inner_dim_to_complex",
                inputs=real_normed_tensor_tensor,
            )
        else:
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
    nnef_spec_strict,
    has_dynamic_axes,
    tract_feature_flags,
    **kwargs,
):
    # NEED SOME FACTOR OUT WITH _FFT and fix to pass window in NNEF-Tools
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L826
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError(
            "STFT not supported by vanilla NNEF"
        )
    (
        input_node,  # Tensor
        n_fft_node,  # int,
        hop_length_node,  # Optional[int] = None
        win_length_node,  # Optional[int] = None
        window_node,  # Optional[Tensor] = None
        normalized_node,  # bool = False
        onesided_node,  # Optional[bool] = None
        return_complex_node,  # Optional[bool] = None
    ) = node.inputs
    assert isinstance(n_fft_node.data, int)
    assert isinstance(hop_length_node.data, int)
    assert isinstance(win_length_node.data, int)
    assert window_node.dtype == torch.float32 and window_node.shape == [
        n_fft_node.data
    ]
    assert normalized_node.data is False

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
                attrs={"axes": [pick_rank(input_node, -1) + 1]},
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
        if tract_feature_flags is not None and "complex" in tract_feature_flags:
            casted_complex_input_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_inner_dim_to_complex",
                inputs=output_nnef_tensor,
                pass_quantization_params=True,
                output_tensor_name_suffix="complex_cast",
            )
        else:
            casted_complex_input_tensor = output_nnef_tensor
    elif input_node.dtype not in [torch.complex64, torch.complex128]:
        raise TorchToNNEFNotImplementedError(
            f"complex type not supported: {input_node.dtype}"
        )
    else:
        casted_complex_input_tensor = nnef_tensor
    dim = pick_rank(input_node, -1)
    window_tensor = get_or_add_tensor_variable_in_nnef(
        g, window_node, name_to_tensor
    )
    frame = win_length_node.data
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
    if onesided_node.data:
        # with length == window size
        # slice rank: dim - 1 by $onesided_max_dim
        onesided_max_idx = (window_node.shape[0] >> 1) + 1
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

    if return_complex_node.data is False:
        if tract_feature_flags is not None and "complex" in tract_feature_flags:
            output_nnef_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_complex_to_inner_dim",
                inputs=output_nnef_tensor,
                output_tensor_name_suffix="cast_back_real",
            )

    transposed_axes = list(range(len(output_nnef_tensor.shape)))
    # permute to follow numpy way of things (as well as tract)
    transposed_axes[dim], transposed_axes[dim + 1] = (
        transposed_axes[dim + 1],
        transposed_axes[dim],
    )
    output_nnef_tensor = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=output_nnef_tensor,
        attrs={"axes": transposed_axes},
        pass_quantization_params=True,
    )

    return ["tract_core"]


@OP_REGISTRY.register()
def fft_fft(
    g,
    node,
    name_to_tensor,
    nnef_spec_strict,
    has_dynamic_axes,
    tract_feature_flags,
    **kwargs,
):
    return _fft(
        node,
        g,
        name_to_tensor,
        nnef_spec_strict,
        has_dynamic_axes,
        inverse=False,
        tract_feature_flags=tract_feature_flags,
    )


@OP_REGISTRY.register()
def fft_ifft(
    g,
    node,
    name_to_tensor,
    nnef_spec_strict,
    has_dynamic_axes,
    tract_feature_flags,
    **kwargs,
):
    return _fft(
        node,
        g,
        name_to_tensor,
        nnef_spec_strict,
        has_dynamic_axes,
        inverse=True,
        tract_feature_flags=tract_feature_flags,
    )
