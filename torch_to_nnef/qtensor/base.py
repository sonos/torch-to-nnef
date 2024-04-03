import abc
import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor


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
    """1 scale and 1 zero point per tensor"""

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


class QTensorBasicExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensorBasic

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        nnef_spec_strict: bool,
        **kwargs,
    ):
        raise NotImplementedError(
            "QTensorBasic are not meant to be exported."
            " please re-consider your quantization tensor type if you wish to export"
            " with (QTensorSepParamsWithPack | QTensorGGUF)"
        )
