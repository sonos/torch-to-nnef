"""Advanced QTensor (<= 8bits) with complex quant scheme non torch native"""

import abc
import tempfile
import typing as T
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_to_nnef import bitpack
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError


class QScheme(abc.ABC):
    @abc.abstractmethod
    def quantize_as_torch(self, fp_tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def dequantize(self, u8_tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def clone_with_scale_factor(self, scale_factor):
        raise NotImplementedError()


class QZPScaleScalar(QScheme):
    def __init__(self, zero_point: int, scale: float):
        assert isinstance(zero_point, int)
        assert isinstance(scale, float)
        assert scale != 0
        self.zero_point = zero_point
        self.scale = scale

    def quantize_as_torch(self, fp_tensor):
        return torch.quantize_per_tensor(
            fp_tensor,
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8,
        )

    def clone_with_scale_factor(self, scale_factor):
        return self.__class__(
            zero_point=(self.zero_point / scale_factor).to(
                dtype=self.zero_point.dtype
            ),
            scale=(self.scale / scale_factor).to(dtype=self.scale.dtype),
        )

    def to_zpscale_per_channel(self, tensor: torch.Tensor, dim: int = -1):
        return QZPScalePerChannel(
            zero_point=(torch.zeros(tensor.shape[dim]) + self.zero_point).to(
                torch.int32
            ),
            scale=torch.zeros(tensor.shape[dim]) + self.scale,
            dim=dim,
        )

    def dequantize(self, u8_tensor):
        return (u8_tensor.to(torch.float32) - self.zero_point) * self.scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zero_point={self.zero_point}, scale={self.scale})"


class QZPScalePerChannel(QScheme):
    def __init__(
        self, zero_point: torch.Tensor, scale: torch.Tensor, dim: int = -1
    ):
        if (
            zero_point.dtype == torch.int64
            and zero_point.abs().max() < torch.iinfo(torch.int64).max
        ):
            zero_point = zero_point.to(torch.int32)
        assert zero_point.dtype == torch.int32, zero_point.dtype
        if scale.dtype == torch.float64:
            scale = scale.to(torch.float32)
        assert scale.dtype == torch.float32
        # assert len(zero_point.shape) == 1 # TODO replace by check only 1 dim > 1
        # assert len(scale.shape) == 1
        assert zero_point.shape == scale.shape
        assert (scale != 0).all(), scale
        self.zero_point = zero_point
        self.scale = scale
        self.dim = dim

    def quantize_as_torch(self, fp_tensor):
        return torch.quantize_per_channel(
            fp_tensor,
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8,
        )

    def clone_with_scale_factor(self, scale_factor):
        return self.__class__(
            zero_point=(self.zero_point / scale_factor).to(
                dtype=self.zero_point.dtype
            ),
            scale=(self.scale / scale_factor).to(dtype=self.scale.dtype),
            dim=self.dim,
        )

    def dequantize(self, u8_tensor):
        return (u8_tensor.to(torch.float32) - self.zero_point) * self.scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zero_point={self.zero_point}, scale={self.scale})"


class QZPScalePerGroup(QScheme):
    def __init__(
        self, group_size: int, zero_point: torch.Tensor, scale: torch.Tensor
    ):
        assert zero_point.dtype == torch.int32
        assert scale.dtype == torch.float32
        assert zero_point.shape == scale.shape
        assert (scale != 0).all(), scale
        self.group_size: int = group_size
        self.zero_point = zero_point
        self.scale = scale

    @classmethod
    def min_max_quantize_float_tensor(
        cls, fp_tensor, group_size: int, n_bits: int
    ) -> T.Tuple[QScheme, torch.Tensor]:
        fp_tensor_per_group = fp_tensor.flatten().reshape(group_size, -1)
        min_per_group = fp_tensor_per_group.min(dim=0).values
        max_per_group = fp_tensor_per_group.max(dim=0).values

        scale = (max_per_group - min_per_group) / ((2**n_bits) - 1)
        zero_point = (-(min_per_group / scale).round()).to(torch.int32)
        qshape = [1] + [scale.shape[0]]
        qscheme = cls(
            group_size=group_size,
            zero_point=zero_point.reshape(qshape),
            scale=scale.reshape(qshape),
        )
        return (
            qscheme,
            qscheme._quantize_as_u8(fp_tensor_per_group)
            .reshape(fp_tensor.shape)
            .to(torch.uint8),
        )

    def _quantize_as_u8(self, fp_tensor):
        assert (
            len(fp_tensor.shape) == 2 and fp_tensor.shape[0] == self.group_size
        )
        u8_tensor_per_group = (
            (fp_tensor / self.scale) + self.zero_point
        ).round()
        return u8_tensor_per_group

    def dequantize(self, u8_tensor):
        u8_tensor_per_group = u8_tensor.flatten().reshape(self.group_size, -1)
        fp_tensor_per_group = (
            u8_tensor_per_group.float() - self.zero_point
        ).round() * self.scale
        return fp_tensor_per_group.reshape(u8_tensor.shape)

    def quantize_as_torch(self, fp_tensor):
        raise TorchToNNEFNotImplementedError(
            "native torch does not suport per chunk"
        )

    def clone_with_scale_factor(self, scale_factor):
        return self.__class__(
            zero_point=(self.zero_point / scale_factor).to(
                dtype=self.zero_point.dtype
            ),
            scale=(self.scale / scale_factor).to(dtype=self.scale.dtype),
            group_size=self.group_size,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"scale={self.scale})"
        )


class PackingLayout(str, Enum):
    TILED = "tiled"
    CONTIGUOUS = "contiguous"


class PackingStrategy:
    def __init__(
        self,
        bit_width: int,
        layout: T.Optional[PackingLayout] = PackingLayout.TILED,
    ):
        assert bit_width in [1, 2, 3, 4, 8]
        assert isinstance(layout, PackingLayout)
        if layout == PackingLayout.CONTIGUOUS:
            raise NotImplementedError("packing contiguous not yet implemented")
        self.bit_width = bit_width
        self.layout = layout

    def pack(self, uint8_tensor: torch.Tensor):
        if self.layout == PackingLayout.CONTIGUOUS:
            raise NotImplementedError("packing contiguous not yet implemented")
        return {
            1: bitpack.TensorB1,
            2: bitpack.TensorB2,
            3: bitpack.TensorB3,
            4: bitpack.TensorB4,
            8: bitpack.TensorB8,
        }[self.bit_width].pack(uint8_tensor)


class TargetDType:
    """targetted dtype

    This structure is needed to allow facade between
    """

    def __init__(self, torch_dtype, qscheme: T.Optional[QScheme] = None):
        self.torch_dtype = torch_dtype
        if qscheme is not None:
            # because in tract
            # this will only be part of
            assert isinstance(
                qscheme, QZPScaleScalar
            ), "other qscheme can not leaks into rest of graph"
        self.qscheme = qscheme

    def to_torch_tensor(self, fp_tensor):
        if self.qscheme is None:
            return fp_tensor.to(self.torch_dtype)
        return self.qscheme.quantize_as_torch(fp_tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(torch_dtype={self.torch_dtype}, qscheme={self.qscheme})"


class QTensor(nn.Module):
    """Common interface for all Quantized storage"""

    def to_torch_float_tensor(self) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self):
        return (
            self.to_torch_float_tensor() * 1.0
        )  # dummy  mul by 1 necessary to avoid torch agressive trace simplification


class QTensorBasic(QTensor):
    """Dissociated QParams and quantized values with basic 1 element per uint8.

    Whathever the bit-width size, 1 element per uint8 is maintained in memory
    While very wastefull this allows to handle ALL tensor shape and QScheme without
    any problem contrary to all other QTensor implementation.

    This QTensorBasic IS NOT meant to be exported in production to tract
    because export will happen statically & there is no bit-packing thus no memory benefit will be hold.
    However it is possible to export it to tract to evaluate accuracy of such model
    with tract math lib manipulations.

    Main Goal/Benefit: Ability to explore quant scheme not optimized but promissing and validate those in tract.

    As an example:
        -> GGUF only implement per groups quantization specific variants (see: QTensorGGUF)
        -> QTensorSepParamsWithPack: is limited to some specific divisibility for float tensor provided 1st dim
    """


try:
    import gguf

    class QTensorGGUF(QTensor):
        """GGUF tensor storage

        Aka tensor format used in Llama.cpp & GGML
        (2024-04-02)

        we define:
            bpw = bit per weight

        ./ggml/src/ggml-common.h
        ./ggml/src/ggml-quants.c

        This storage is heavily tweaked/optimal for LLM so,
        no guaranty it will perform best on other NN arch / ML tasks

        To our knowedge there is 3 kinds of formats:
        - old legacy formats (still in use in many models):
            (tensor shape need to be divisible by 32)
            Q{X}_0 -> symetric quant with per group quantization of 32 elements
                Q8_0 -> [f16 scale, 32x(8bits element)] -> 34 bytes per group -> 8,5 bpw
                Q4_0 -> [f16 scale, 32x(4bits element)] -> 18 bytes per group -> 4.5 bpw
                Q5_0 -> [f16 scale, 32x(5th bit of each element), 32x(4 first bits of each element)]
                            -> 22 bytes per group
                            -> 5.5 bpw

            Q{X}_1 -> asymetric quant with per group quantization of 32 elements
                Q8_1 -> [f16 scale, f16 min, 32x(8bits element)] -> 36 bytes per group -> 9 bpw
                Q4_1 -> [f16 scale, f16 min, 32x(4bits element)] -> 20 bytes per group -> 5 bpw
                Q5_1 -> [f16 scale, f16 min, 32x(5th bit of each element), 32x(4 first bit of each elements)]
                            -> 24 bytes per group
                            -> 6 bpw

        - new formats: With double quantization formats where qparams are quantized themselves,
                macro quantization parameters are used to dequantize qparams

            Format group elements by 256 (so tensor shape need to be divisible by 256)

            Q{X}_K:
                Q2_K -> [16x(4bits min, 4bits scale), 256x(2bits element), f16 macro scale, f16 macro min]
                            -> 84 bytes per group
                            -> 2,625 bpw

                Q3_K -> [
                        256x(3rd bit per element),
                        256x(2 first bits bit per element),
                        16x(6bits quantized scales),
                        f16 macro scale
                        ]
                            -> 110 bytes per group
                            -> 3,4375 bpw
                Q4_K -> [f16 macro scale, f16 macro min, 8x(6bits min, 6bits scale), 128x(4bits element)]
                            -> 80 bytes per group
                            -> 5 bpw
                Q5_K -> [
                        f16 macro scale,
                        f16 macro min,
                        8x(6bits min, 6bits scale),
                        256x(5th bit per element),
                        256x(4 first bits per element)
                        ]
                            -> 176 bytes per group
                            -> 5,5 bpw
                Q6_K -> ..
                Q8_K -> ..

        - new formats: with non-linearity or very low bit-width
            IQ{X}_{SIZE}:
                where SIZE can be:
                    XXS, XS, S, M
                and X can be 1, 2, 3

                format not studied but:
                    iq1_s --> ... --> 1.56 bpw
                    iq1_m -> ... --> 1.75 bpw
                    iq2_xxs --> [f16 scale, 256x(2bits element)] -> 2.0625 bpw
                    iq2_xs --> [f16 macro scale, 256x(2bits element), ~n x qscale~] -> 2.3125 bpw
                    iq2_xs --> ... -> 2.5625 bpw
                    iq3_xxs --> ... -> 3.0625 bpw

                i-quants familly is also providing in some case non linear quantization, ending with "_nl" notation
                see: https://github.com/ggerganov/llama.cpp/discussions/5063#discussioncomment-8383732
                for performance

        However to date gguf 0.6.0 only reference familly Q_{X}_0, Q_{X}_1 and Q_{X}_K
            other are on main but not yet released

        warning!: elements order is not maintained packing is applied in tile
            in 4 bit by example on 32 element stored index would be in store:
                [0, 16, 1, 17, ..., 15, 31]

        Implementation details:
            As of 2024-04-02 we only rely on gguf python library,

            ggml modified: https://github.com/JulienBalianSonos/ggml.git
            is only meant for tract unittest generation purpose

            This limit us to only quantization but not dequantization implementation
            for production export, we do this because:
            GGML python library (on mainstream), is not well supported (more a POC)
            and it needs you to provide specific .so library to link against library via
            env variable.

        """

        def __init__(
            self,
            float_torch_tensor: torch.Tensor,
            gguf_data_type: int,  # : "GGUFDataType"
        ):
            super().__init__()
            if isinstance(float_torch_tensor, nn.Parameter):
                float_torch_tensor = float_torch_tensor.data
            self._float_torch_tensor = float_torch_tensor
            self.gguf_data_type = gguf_data_type

        def _write_tensor_in_gguf_file(
            self,
            dirpath: Path,
            variable_name: str,
            np_float_tensor: np.ndarray,
            dtype: int,
        ):
            filepath = dirpath / f"{variable_name}.gguf"
            # Example usage with a file
            gguf_writer = gguf.GGUFWriter(filepath, "tract_custom")
            # gguf_writer.add_block_count(1)
            gguf_writer.add_tensor(
                variable_name, np_float_tensor, raw_dtype=dtype
            )

            gguf_writer.write_header_to_file()
            gguf_writer.write_kv_data_to_file()
            gguf_writer.write_tensors_to_file()
            gguf_writer.close()
            return filepath

        def _get_tensor_data_from_gguf_file(
            self, gguf_file_path: str, variable_name: str
        ):
            reader = gguf.GGUFReader(gguf_file_path)
            for tensor in reader.tensors:
                if tensor.name == variable_name:
                    return tensor.data
            raise ValueError(
                f"not found tensor '{variable_name}' in gguf file: {gguf_file_path}"
            )

        @property
        def ggml_data_np_tensor(self) -> np.ndarray:
            with tempfile.TemporaryDirectory() as dir_path:
                filepath = self._write_tensor_in_gguf_file(
                    Path(dir_path),
                    "a",
                    self._float_torch_tensor.numpy(),
                    self.gguf_data_type,
                )
                qdata = self._get_tensor_data_from_gguf_file(filepath, "a")
            return qdata

        def to_torch_float_tensor(self):
            return self._float_torch_tensor

        def __repr__(self) -> str:
            try:
                return (
                    f"{self.__class__.__name__}(shape={tuple(self.float_torch_tensor.shape)},"
                    f" gguf_target_dtype={self.gguf_data_type})"
                )
            except AttributeError:
                return f"{self.__class__.__name__}(?)"

except ImportError as exp:
    # feature gate: gguf_dtype
    print(exp)


class QTensorSepParamsWithPack(QTensor):
    """Quantized Tensor

    This format is usefull to store directly in RAM as torch tensor as there
    is very few manipulations to restore float tensor whatever PyTorch backend
    this make it faster than the others format against all hardware achitecture
    while saving RAM significantly without dedicated C/CPP code.

    Pro:
        1. lot of variants supported (all group size / per chan ...)

    Limitation:
        1. Your initial float tensor first dimension need to be divisible by
        number of element packable in a byte
        2. No implementation in tract

    """

    def __init__(
        self,
        packed_tensor: bitpack.BitPackedTensor,
        qscheme: T.Optional[QScheme],
        target_dtype: TargetDType,
    ):
        super().__init__()
        assert isinstance(qscheme, QScheme)
        assert isinstance(packed_tensor, bitpack.BitPackedTensor)
        self.packed_torch_tensor = packed_tensor
        self.qscheme = qscheme
        self.target_dtype = target_dtype

    def to_torch_tensor(self) -> torch.Tensor:
        u8_tensor = self.packed_torch_tensor.unpack()
        if self.qscheme:
            fp_tensor = self.qscheme.dequantize(u8_tensor)
        else:
            fp_tensor = u8_tensor
        return self.target_dtype.to_torch_tensor(fp_tensor)

    def detach(self):
        return self

    def requires_grad_(self, *args, **kwargs):
        return self

    def clone(self):
        raise NotImplementedError()

    def to(self, *args, **kwargs):
        raise NotImplementedError()

    def cast_into_n_bits(self, n_bits: int):
        current_n_bits_per_elm = self.packed_torch_tensor.n_bits()
        if n_bits == current_n_bits_per_elm:
            return self
        tt = self.packed_torch_tensor.unpack()
        # tmin, tmax = tt.min(), tt.max()
        # max_upper_bound = self.packed_torch_tensor.max_val()
        # assert tmax == max_upper_bound
        # assert tmin == 0
        scale_factor = 2**current_n_bits_per_elm / 2**n_bits
        new_u8_tensor = (tt.to(torch.float32) / scale_factor).to(torch.uint8)
        return QTensorSepParamsWithPack(
            packed_tensor=PackingStrategy(bit_width=n_bits).pack(new_u8_tensor),
            qscheme=self.qscheme.clone_with_scale_factor(scale_factor),
            target_dtype=self.target_dtype,
        )

    @classmethod
    def from_torch_qtensor(
        cls,
        tensor,
        packing_strategy: T.Optional[PackingStrategy] = None,
        target_dtype=None,
    ) -> "QTensor":
        assert tensor.is_quantized
        if tensor.dtype in [torch.quint4x2, torch.quint2x4]:
            # TODO: handle torch.quint4x2 and torch.quint2x4
            raise TorchToNNEFNotImplementedError(tensor.dtype)
        if packing_strategy is None:
            # default
            packing_strategy = PackingStrategy(bit_width=8)
        assert isinstance(packing_strategy, PackingStrategy)
        torch_qscheme = tensor.qscheme()
        qscheme: T.Optional[QScheme] = None
        itensor = tensor.int_repr()
        offset_zp = 0
        if itensor.dtype == torch.int8:
            itensor = (itensor.to(torch.int16) + 128).to(torch.uint8)
            offset_zp += 128

        if torch_qscheme == torch.per_channel_affine:
            qscale = tensor.q_per_channel_scales()
            qzerop = tensor.q_per_channel_zero_points() + offset_zp
            dim = tensor.q_per_channel_axis()

            # reshapes to allow torch jit works
            qshape = [1] * len(tensor.shape)
            qshape[dim] = qscale.shape[0]
            qscale = qscale.reshape(qshape)
            qzerop = qzerop.reshape(qshape)

            qscheme = QZPScalePerChannel(qzerop, qscale, dim=dim)
        elif torch_qscheme == torch.per_tensor_affine:
            qscale = tensor.q_scale()
            qzerop = tensor.q_zero_point() + offset_zp
            qscheme = QZPScaleScalar(qzerop, qscale)
        else:
            raise TorchToNNEFNotImplementedError(
                f"not supported quantization scheme {qscheme}"
            )

        if target_dtype is None:
            target_dtype = TargetDType(tensor.dtype)
        return cls(
            packing_strategy.pack(itensor),
            qscheme=qscheme,
            target_dtype=target_dtype,
        )

    def forward(self):
        return self.to_torch_tensor()

    def __repr__(self) -> str:
        try:
            return f"{self.__class__.__name__}({self.packed_torch_tensor}, {self.qscheme})"
        except AttributeError:
            return f"{self.__class__.__name__}(?)"


class WeightInputedConv1d(nn.Module):
    CP_ATTRS = [
        "bias",
        "padding_mode",
        "stride",
        "dilation",
        "padding",
        "groups",
        "_reversed_padding_repeated_twice",
    ]

    def __init__(self, conv_nn: nn.Conv1d):
        super().__init__()
        for attr in self.CP_ATTRS:
            setattr(self, attr, getattr(conv_nn, attr))

    def _conv_forward(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: T.Optional[torch.Tensor],
    ):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    inp,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inp, weight, self.bias)


class WeightInputedLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    CP_ATTRS = ["bias", "in_features", "out_features"]

    def __init__(self, linear: nn.Linear):
        super().__init__()
        for attr in self.CP_ATTRS:
            setattr(self, attr, getattr(linear, attr))

    def forward(self, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(inp, weight, self.bias)


class QWeightedOp(nn.Module):
    def __init__(self, mod: nn.Module, weight_mod: QTensor):
        super().__init__()
        self.mod = mod
        self.weight_mod = weight_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x, self.weight_mod())

    def __getattr__(self, name):
        mod_dic = self.__dict__
        if name in mod_dic:
            return mod_dic[name]
        mod_dic = mod_dic["_modules"]
        if name in mod_dic:
            return mod_dic[name]
        mod_dic = mod_dic["mod"].__dict__
        if name in mod_dic:
            return mod_dic[name]
        raise AttributeError(f"{name} not found")


def replace_nn_ops(module: nn.Module, q_weight: QTensor) -> nn.Module:
    if isinstance(module, nn.Conv1d):
        assert (
            module.weight.shape == q_weight().shape
        ), f"{module.weight.shape} == {q_weight().shape}"
        return QWeightedOp(WeightInputedConv1d(module), q_weight)
    if isinstance(module, nn.Linear):
        assert (
            module.weight.shape == q_weight().shape
        ), f"{module.weight.shape} == {q_weight().shape}"
        return QWeightedOp(WeightInputedLinear(module), q_weight)
    raise NotImplementedError(module)
