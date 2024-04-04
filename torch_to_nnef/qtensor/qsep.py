import logging
import typing as T
from enum import Enum

import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor import bitpack
from torch_to_nnef.qtensor.base import (
    QScheme,
    QTensor,
    QZPScalePerChannel,
    QZPScaleScalar,
)

LOGGER = logging.getLogger(__name__)


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
            # idea would be to relax disibility criterion with padding
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
        # NOTE: not ideal as this imply:
        # custom_format -> dequant -> requant -> qtorch format
        return self.qscheme.quantize_as_torch(fp_tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(torch_dtype={self.torch_dtype}, qscheme={self.qscheme})"


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


# export logic {


class QTensorSepParamsWithPackExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensorSepParamsWithPack

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
        # why is this an empty ModuleNotFoundError ?
        # because provided node.output[0]
        # is a TensorVariable with data
        assert node.outputs[0].data is not None
        # this means that expansion to linear/conv/...
        # will handle this as a classical variable (weight/bias..)
        # and since node.output[0].data hold the dequantized
        # value from quant scheme selected. it is
        # sufficient to simulate weight quantization error
        LOGGER.warning(
            "Export of QTensorSepParamsWithPack to tract is only meant for simulation ! "
            "No memory benefit will be hold"
        )


# }
