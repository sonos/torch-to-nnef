""" Advanced QTensor (<= 8bits) with complex quant scheme non torch native """
import abc
import typing as T
from enum import Enum

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

    def to_zpscale_per_channel(self, tensor: torch.Tensor, dim: int = -1):
        return QZPScalePerChannel(
            zero_point=(torch.zeros(tensor.shape[dim]) + self.zero_point).to(
                torch.int32
            ),
            scale=torch.ones(tensor.shape[dim]) + self.scale,
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

    def dequantize(self, u8_tensor):
        return (u8_tensor.to(torch.float32) - self.zero_point) * self.scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zero_point={self.zero_point}, scale={self.scale})"


class QZPScalePerChunk(QScheme):
    def __init__(
        self, chunk_size: int, zero_point: torch.Tensor, scale: torch.Tensor
    ):
        assert zero_point.dtype == torch.int32
        assert scale.dtype == torch.float32
        assert len(zero_point.shape) == 1
        assert len(scale.shape) == 1
        assert zero_point.shape == scale.shape
        assert (-129 < zero_point & zero_point < 128).all(), zero_point
        assert (scale != 0).all(), scale
        self.chunk_size: int = chunk_size
        self.zero_point = zero_point
        self.scale = scale

    def quantize_as_torch(self, fp_tensor):
        raise TorchToNNEFNotImplementedError(
            "native torch does not suport per chunk"
        )

    def dequantize(self, u8_tensor):
        raise TorchToNNEFNotImplementedError(
            "not enough knowledge on chunk structure at this stage"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(chunk={self.chunk_size}, zero_point={self.zero_point}, scale={self.scale})"


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
    """Quantized Tensor

    Proposed export layout for this tensor:

    Example 1: (unquant dynamically weight at each graph run with rest of the graph quantized )
        x = tract_core_external(shape=(a, b), datum_type='u8')
        x_dyn = tract_core_dyn_bit_unpack(bit_width=4, layout='tiled')
        y = tract_core_cast(x_dyn) # with graph.quant containing =>  zero_point=0, scale=0.5

    Example 2: (unquant dynamically weight at each graph run with rest of the graph fp)
        x = tract_core_external(shape=(a, b), datum_type='u8')
        x_dyn = tract_core_dyn_bit_unpack(bit_width=4, layout='tiled')
        y = tract_core_cast(x_dyn) # with graph.quant containing =>  zero_point=0, scale=0.5
        z = tract_core_cast(y, to='f32')

    Example 3: (per channel quantization targeting f16)
        x = tract_core_external(shape=(a, b), datum_type='u8')
        x_dyn = tract_core_dyn_bit_unpack(bit_width=4, layout='tiled')
        zero_point_x = tract_core_external(shape=(b,), datum_type='i32')
        scale_x = tract_core_external(shape=(b,), datum_type='f32')
        y = tract_core_zpscale_per_channel(x_dyn, zero_point=zero_point_x, scale=scale_x, to='f16')

    Example 4: (per chunk quantization targeting f16)
        x = tract_core_external(shape=(a, b), datum_type='u8')
        x_dyn = tract_core_dyn_bit_unpack(bit_width=4, layout='tiled')
        zero_point_x = tract_core_external(shape=(a * b / 128,), datum_type='i32')
        scale_x = tract_core_external(shape=(a * b / 128,), datum_type='f32')
        y = tract_core_zpscale_per_chunk(x_dyn, zero_point=zero_point_x, scale=scale_x, to='f16')

    Drawback:
        -> per channel quantization targeting Q8 not handled (think quantized matmul)
        -> per chunk quantization targeting Q8 not handled
        This is because we would need to propagate these as tract tensor types instead of tract operators
        The handle those peculiar datum_type whould likely leads to heavy tract binary size

    NOTE:
        there is 3 main components it export to:
        - tract_core_dyn_bit_unpack: that allow to unpack in u8 tensor
        - tract_core_zpscale_per_channel: that allows casting into specific float
        - tract_core_zpscale_per_chunk: that allows casting into specific float

    WARNING:
        This tensor is only meant for storage and export DO not use for other torch operation
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
        return QTensor(
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

    #


def replace_nn_ops(module, q_weight):
    if isinstance(module, nn.Conv1d):
        assert module.weight.shape == q_weight.packed_torch_tensor.shape
        return QWeightedOp(WeightInputedConv1d(module), q_weight)
    raise NotImplementedError(module)
