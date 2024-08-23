import abc
import logging
import typing as T

import torch

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


class QTensor(torch.Tensor):
    """Common interface for all Quantized storage"""

    def to_torch_float_tensor(self) -> torch.Tensor:
        raise NotImplementedError()
