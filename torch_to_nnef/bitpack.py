""" Base tensor bit-packing and unpack of values in  tensor lower than 8 bit.

For now there is no residual handling enhence packing is only possible
if divisibility match

Use packing by tile this means that internal colocated values are not
neccessary close to each other in reality, this allow for fast un/packing.

These base tensors are supported for NNEF export.

"""
import abc
import typing as T

import numpy as np
import torch

from torch_to_nnef.exceptions import BitPackingError


class BitPackedTensor(abc.ABC):
    def __init__(self, storage_tensor, shape: T.Tuple[int, ...]):
        self._shape = shape
        if storage_tensor.dtype != self.storage_dtype():
            raise BitPackingError(
                f"got {storage_tensor.dtype} but expected {self.storage_dtype()}"
            )
        self._tensor = storage_tensor

    @property
    def raw_tensor(self):
        return self._tensor

    @property
    def shape(self):
        return self._shape

    @staticmethod
    @abc.abstractmethod
    def storage_dtype():
        raise NotImplementedError("Need to implement storage_dtype")

    @staticmethod
    @abc.abstractmethod
    def n_bits():
        raise NotImplementedError("Need to implement n bits")

    @classmethod
    def max_val(cls):
        return 2 ** cls.n_bits() - 1

    @classmethod
    def storage_bit_size_per_element(cls):
        return torch.tensor([], dtype=cls.storage_dtype()).element_size() * 8

    @classmethod
    def pack(cls, tensor):
        if tensor.dtype != cls.storage_dtype():
            raise BitPackingError(
                f"Expected dtype:{cls.storage_dtype()} but provided:{tensor.dtype}"
            )

        divisor = int(cls.storage_bit_size_per_element() / cls.n_bits())
        if not len(tensor) % divisor == 0:
            raise BitPackingError(
                f"tensor must have shape[0] divisible by {divisor} to use bit packing but got {tensor.shape[0]}"
            )
        if (tensor > cls.max_val()).any():
            raise BitPackingError(
                f"all values must be bellow or equal {cls.max_val()} to use {cls}"
            )
        return cls(cls._pack(tensor), tensor.shape)

    @staticmethod
    @abc.abstractmethod
    def _pack(tensor):
        raise NotImplementedError("Need to implement pack in u8")

    @abc.abstractmethod
    def unpack(self):
        raise NotImplementedError("Need to implement unpack in u8")

    def forward(self):
        return self.unpack()

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self._tensor})"

    def bin_repr(self):
        """Useful for debuging"""
        txt = "[\n"
        rows_reprs = []
        for row in self._tensor.tolist():
            row_repr = (
                "[" + ", ".join(format(elm, "#010b") for elm in row) + "]"
            )
            rows_reprs.append(row_repr)
        txt += ",\n".join(rows_reprs)
        txt += "\n]"
        print(txt)


class BitPackedTensorU8(BitPackedTensor):
    @staticmethod
    def storage_dtype():
        return torch.uint8


class BitPackedTensorI32(BitPackedTensor):
    @staticmethod
    def storage_dtype():
        return torch.int32


class TensorB8(BitPackedTensorU8):
    """store 1 values for each u8"""

    @staticmethod
    def n_bits():
        return 8

    @staticmethod
    def _pack(tensor):
        return tensor

    def unpack(self):
        return self._tensor


class TensorB4(BitPackedTensorU8):
    """store 2 values for each u8"""

    @staticmethod
    def n_bits():
        return 4

    @staticmethod
    def _pack(tensor):  # uint8 > uint8/2
        step = int(len(tensor) / 2)
        return (tensor[:step] << 4) | tensor[step:]

    def unpack(self):  # uint8/2 > uint8
        step = self._tensor.shape[0]
        out = torch.empty(
            [2 * step] + list(self._tensor.shape[1:]),
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        out[:step] = (self._tensor & 0b11110000) >> 4
        out[step:] = self._tensor & 0b00001111
        return out


class TensorB2(BitPackedTensorU8):
    """store 4 values for each u8"""

    @staticmethod
    def n_bits():
        return 2

    @staticmethod
    def _pack(tensor):  # uint8 > uint8/4
        step = int(len(tensor) / 4)
        return (
            tensor[:step] << 6
            | tensor[step : 2 * step] << 4
            | tensor[2 * step : 3 * step] << 2
            | tensor[3 * step :]
        )

    def unpack(self):
        step = self._tensor.shape[0]
        out = torch.empty(
            [4 * step] + list(self._tensor.shape[1:]),
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        out[:step] = (self._tensor & 0b11000000) >> 6
        out[step : 2 * step] = (self._tensor & 0b00110000) >> 4
        out[2 * step : 3 * step] = (self._tensor & 0b00001100) >> 2
        out[3 * step :] = self._tensor & 0b00000011
        return out


class TensorB1(BitPackedTensorU8):
    """store 8 values for each u8"""

    @staticmethod
    def n_bits():
        return 1

    @staticmethod
    def _pack(tensor):  # uint8 > uint8/4
        tensor = tensor.to(torch.uint8)
        step = int(len(tensor) / 8)
        return (
            tensor[:step] << 7
            | tensor[step : 2 * step] << 6
            | tensor[2 * step : 3 * step] << 5
            | tensor[3 * step : 4 * step] << 4
            | tensor[4 * step : 5 * step] << 3
            | tensor[5 * step : 6 * step] << 2
            | tensor[6 * step : 7 * step] << 1
            | tensor[7 * step :]
        )

    def unpack(self):
        step = self._tensor.shape[0]
        out = torch.empty(
            [8 * step] + list(self._tensor.shape[1:]),
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        out[:step] = (self._tensor & 0b10000000) >> 7
        out[1 * step : 2 * step] = (self._tensor & 0b01000000) >> 6
        out[2 * step : 3 * step] = (self._tensor & 0b00100000) >> 5
        out[3 * step : 4 * step] = (self._tensor & 0b00010000) >> 4
        out[4 * step : 5 * step] = (self._tensor & 0b00001000) >> 3
        out[5 * step : 6 * step] = (self._tensor & 0b00000100) >> 2
        out[6 * step : 7 * step] = (self._tensor & 0b00000010) >> 1
        out[7 * step : 8 * step] = self._tensor & 0b00000001
        return out


class TensorB3(BitPackedTensorI32):
    """store 10 values for each i32

    loose 2 bits per i32 (6.25% useless store)
    """

    @staticmethod
    def n_bits():
        return 3

    @staticmethod
    def _pack(tensor):
        out = torch.zeros(
            [int(10 * np.ceil(tensor.shape[0] / 10.0))]
            + list(tensor.shape[1:]),
            device=tensor.device,
            dtype=torch.int32,
        )
        out[: len(tensor)] = tensor
        step = int(len(out) / 10)
        out = (
            (out[:step] << 27)
            | (out[step : step * 2] << 24)
            | (out[step * 2 : step * 3] << 21)
            | (out[step * 3 : step * 4] << 18)
            | (out[step * 4 : step * 5] << 15)
            | (out[step * 5 : step * 6] << 12)
            | (out[step * 6 : step * 7] << 9)
            | (out[7 * step : step * 8] << 6)
            | (out[step * 8 : step * 9] << 3)
            | (out[step * 9 :])
        )
        return out

    def unpack(self):
        step = self._tensor.shape[0]
        out = torch.empty(
            [10 * step] + list(self._tensor.shape[1:]),
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        out[:step] = (self._tensor & 0b00111000000000000000000000000000) >> 27
        out[1 * step : 2 * step] = (
            self._tensor & 0b00000111000000000000000000000000
        ) >> 24
        out[2 * step : 3 * step] = (
            self._tensor & 0b00000000111000000000000000000000
        ) >> 21
        out[3 * step : 4 * step] = (
            self._tensor & 0b00000000000111000000000000000000
        ) >> 18
        out[4 * step : 5 * step] = (
            self._tensor & 0b00000000000000111000000000000000
        ) >> 15
        out[5 * step : 6 * step] = (
            self._tensor & 0b00000000000000000111000000000000
        ) >> 12
        out[6 * step : 7 * step] = (
            self._tensor & 0b00000000000000000000111000000000
        ) >> 9
        out[7 * step : 8 * step] = (
            self._tensor & 0b00000000000000000000000111000000
        ) >> 6
        out[8 * step : 9 * step] = (
            self._tensor & 0b00000000000000000000000000111000
        ) >> 3
        out[9 * step :] = self._tensor & 0b00000000000000000000000000000111
        return out


class TensorB3InU8(BitPackedTensorU8):
    """store 2 values for each u8

    Easier to use compared to i32 version, as it accept any divisible 2 tensor.shape[0]

    Loose 2 bit per u8 (25% useless store).

    """

    @staticmethod
    def n_bits():
        return 3

    @staticmethod
    def _pack(tensor):
        out = torch.zeros(
            [int(2 * np.ceil(tensor.shape[0] / 2.0))] + list(tensor.shape[1:]),
            device=tensor.device,
            dtype=torch.uint8,
        )
        out[: len(tensor)] = tensor
        step = int(len(tensor) / 2)
        out = out[:step] << 5 | out[1 * step : 2 * step] << 2
        return out

    def unpack(self):
        step = self._tensor.shape[0]
        out = torch.empty(
            [2 * step] + list(self._tensor.shape[1:]),
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        out[:step] = (self._tensor & 0b11100000) >> 5
        out[1 * step : 2 * step] = (self._tensor & 0b00011100) >> 2
        return out
