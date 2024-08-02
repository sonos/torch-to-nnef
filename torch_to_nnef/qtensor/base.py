import abc
import logging
import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError

LOGGER = logging.getLogger(__name__)


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


class QScalePerGroupF16(QScheme):
    """Tract aligned

    using negative scales

    """

    def __init__(
        self,
        group_size: int,
        scale: torch.Tensor,
        n_bits: int,
    ):
        scale = scale.to(torch.float16)
        # assert (scale != 0).all(), scale
        self.group_size: int = group_size
        self.scale = scale
        self.n_bits = n_bits  # needed for bit-shift before packing

    def quantize_as_torch(self, fp_tensor):
        raise TorchToNNEFNotImplementedError(
            "native torch does not suport per chunk"
        )

    @classmethod
    def min_max_calibration(
        cls,
        fp_tensor,
        n_bits: int,
        group_size: int,
        percentile: float = 1.0,
    ) -> T.Tuple["QScalePerGroupF16", torch.Tensor]:
        assert 0.0 < percentile <= 1.0
        volume = 1
        for fp_dim in fp_tensor.shape:
            volume *= fp_dim
        if volume % group_size != 0:
            raise ValueError(
                f"tensor provided volume: {volume} but group size are {group_size} "
                "incomplete groups aren't supported."
            )
        fp_tensor_per_group = fp_tensor.flatten().reshape(-1, group_size)

        # we use full-range symmetric
        # like torch, ONNX, but oposed to restricted range from
        # TensorFlow, NVIDIA TensorRT and Intel DNNL
        scale = torch.quantile(fp_tensor_per_group.abs(), percentile, dim=1) / (
            -(2**n_bits) / 2
        )

        assert scale.shape[0] == fp_tensor_per_group.shape[0]
        qshape = [scale.shape[0]] + [1]
        qscheme = cls(
            group_size=group_size,
            scale=scale.reshape(qshape),
            n_bits=n_bits,
        )
        return (
            qscheme,
            qscheme.quantize_as_u8(fp_tensor_per_group).reshape(
                fp_tensor.shape
            ),
        )

    def quantize_as_u8(self, fp_tensor):
        assert (
            len(fp_tensor.shape) == 2 and fp_tensor.shape[1] == self.group_size
        )
        recip_scale = torch.where(self.scale == 0, self.scale, 1.0 / self.scale)
        fu8_tensor_per_group = (
            ((fp_tensor * recip_scale) + (2**self.n_bits + 1) / 2)
            .floor()
            .clamp(0, 2**self.n_bits - 1)
        )
        return fu8_tensor_per_group.to(torch.uint8)

    def dequantize(self, u8_tensor):
        u8_tensor_per_group = u8_tensor.flatten().reshape(-1, self.group_size)
        offset = 2**self.n_bits / 2
        fp_tensor_per_group = u8_tensor_per_group.to(torch.float16) - offset
        fp_tensor_per_group *= self.scale
        return fp_tensor_per_group.reshape(u8_tensor.shape)

    def clone_with_scale_factor(self, scale_factor):
        return self.__class__(
            scale=(self.scale / scale_factor).to(dtype=self.scale.dtype),
            group_size=self.group_size,
            n_bits=self.n_bits,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(group_size={self.group_size}, "
            f"scale={self.scale})"
        )


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


class QZPScalePerChannelFloat(QScheme):
    """Same as QZPScalePerChannel but with zero point in float"""

    def __init__(
        self, zero_point: torch.Tensor, scale: torch.Tensor, dim: int = -1
    ):
        assert zero_point.dtype == torch.float32, zero_point.dtype
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
        assert max_per_group.shape[0] == fp_tensor_per_group.shape[-1]

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


class QScalePerGroup(QScheme):
    def __init__(self, group_size: int, scale: torch.Tensor):
        assert scale.dtype == torch.float32
        assert (scale != 0).all(), scale
        self.group_size: int = group_size
        self.scale = scale

    @classmethod
    def min_max_quantize_float_tensor(
        cls, fp_tensor, group_size: int, n_bits: int
    ) -> T.Tuple[QScheme, torch.Tensor]:
        fp_tensor_per_group = fp_tensor.flatten().reshape(group_size, -1)
        min_per_group = fp_tensor_per_group.min(dim=0).values
        max_per_group = fp_tensor_per_group.max(dim=0).values
        assert max_per_group.shape[0] == fp_tensor_per_group.shape[-1]

        scale = (max_per_group - min_per_group) / ((2**n_bits) - 1)
        qshape = [1] + [scale.shape[0]]
        qscheme = cls(
            group_size=group_size,
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
        u8_tensor_per_group = (fp_tensor / self.scale).round()
        return u8_tensor_per_group

    def dequantize(self, u8_tensor):
        u8_tensor_per_group = u8_tensor.flatten().reshape(self.group_size, -1)
        fp_tensor_per_group = (u8_tensor_per_group.float()).round() * self.scale
        return fp_tensor_per_group.reshape(u8_tensor.shape)

    def quantize_as_torch(self, fp_tensor):
        raise TorchToNNEFNotImplementedError(
            "native torch does not suport per chunk"
        )

    def clone_with_scale_factor(self, scale_factor):
        return self.__class__(
            scale=(self.scale / scale_factor).to(dtype=self.scale.dtype),
            group_size=self.group_size,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(group_size={self.group_size}, "
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
