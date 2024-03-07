""" Base tensor to pack and unpack values in Q tensor lower than 8.

For now there is no residual handling enhence packing is only possible
if divisibility match

These base tensors are supported for NNEF export.

"""
import abc
import typing as T

import numpy as np
import torch
from torch import nn


class QTensorPacked(nn.Module, abc.ABC):
    def __init__(self, storage_tensor, shape: T.Tuple[int, ...]):
        super().__init__()
        self._shape = shape
        assert storage_tensor.dtype == self.storage_dtype(), storage_tensor
        self._tensor = storage_tensor

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
    def pack(cls, tensor):
        if tensor.dtype != cls.storage_dtype():
            raise ValueError(
                f"Expected dtype:{cls.storage_dtype()} but provided:{tensor.dtype}"
            )
        if len(tensor.shape) != 2:
            raise ValueError(
                f"only 2d tensor supported for QTensorPacked but provided shape:{tensor.shape}"
            )

        divisor = int(cls.storage_dtype().itemsize * 8 / cls.n_bits())
        if not len(tensor) % divisor == 0:
            raise ValueError(
                f"tensor must have shape[0] divisible by {divisor} to use bit packing but got {tensor.shape[0]}"
            )
        if (tensor > cls.max_val()).any():
            raise ValueError(
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


class QTensorPackedU8(QTensorPacked):
    @staticmethod
    def storage_dtype():
        return torch.uint8


class QTensorPackedI32(QTensorPacked):
    @staticmethod
    def storage_dtype():
        return torch.int32


class Q8Tensor(QTensorPackedU8):
    """store 1 values for each u8"""

    @staticmethod
    def n_bits():
        return 8

    @staticmethod
    def _pack(tensor):
        return tensor

    def unpack(self):
        return self._tensor


class Q4Tensor(QTensorPackedU8):
    """store 2 values for each u8"""

    @staticmethod
    def n_bits():
        return 4

    @staticmethod
    def _pack(tensor):  # uint8 > uint8/2
        _step = int(len(tensor) / 2)
        return (tensor[:_step] << 4) | tensor[_step:]

    def unpack(self):  # uint8/2 > uint8
        _step = self._tensor.shape[0]
        tmp = torch.empty(
            [2 * _step, self._tensor.shape[1]],
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        tmp[:_step] = (self._tensor & 0b11110000) >> 4
        tmp[_step:] = self._tensor & 0b00001111
        return tmp


class Q2Tensor(QTensorPackedU8):
    """store 4 values for each u8"""

    @staticmethod
    def n_bits():
        return 2

    @staticmethod
    def _pack(tensor):  # uint8 > uint8/4
        _step = int(len(tensor) / 4)
        return (
            tensor[:_step] << 6
            | tensor[_step : 2 * _step] << 4
            | tensor[2 * _step : 3 * _step] << 2
            | tensor[3 * _step :]
        )

    def unpack(self):
        _step = self._tensor.shape[0]
        tmp = torch.empty(
            [4 * _step, self._tensor.shape[1]],
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        tmp[:_step] = (self._tensor & 0b11000000) >> 6
        tmp[_step : 2 * _step] = (self._tensor & 0b00110000) >> 4
        tmp[2 * _step : 3 * _step] = (self._tensor & 0b00001100) >> 2
        tmp[3 * _step :] = self._tensor & 0b00000011
        return tmp


class Q1Tensor(QTensorPackedU8):
    """store 8 values for each u8"""

    @staticmethod
    def n_bits():
        return 1

    @staticmethod
    def _pack(tensor):  # uint8 > uint8/4
        tensor = tensor.to(torch.uint8)
        _step = int(len(tensor) / 8)
        return (
            tensor[:_step] << 7
            | tensor[_step : 2 * _step] << 6
            | tensor[2 * _step : 3 * _step] << 5
            | tensor[3 * _step : 4 * _step] << 4
            | tensor[4 * _step : 5 * _step] << 3
            | tensor[5 * _step : 6 * _step] << 2
            | tensor[6 * _step : 7 * _step] << 1
            | tensor[7 * _step :]
        )

    def unpack(self):
        _step = self._tensor.shape[0]
        tmp = torch.empty(
            [8 * _step, self._tensor.shape[1]],
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        tmp[:_step] = (self._tensor & 0b10000000) >> 7
        tmp[1 * _step : 2 * _step] = (self._tensor & 0b01000000) >> 6
        tmp[2 * _step : 3 * _step] = (self._tensor & 0b00100000) >> 5
        tmp[3 * _step : 4 * _step] = (self._tensor & 0b00010000) >> 4
        tmp[4 * _step : 5 * _step] = (self._tensor & 0b00001000) >> 3
        tmp[5 * _step : 6 * _step] = (self._tensor & 0b00000100) >> 2
        tmp[6 * _step : 7 * _step] = (self._tensor & 0b00000010) >> 1
        tmp[7 * _step : 8 * _step] = self._tensor & 0b00000001
        return tmp


class Q3Tensor(QTensorPackedI32):
    """store 10 values for each i32

    loose 2 bits per i32 (6.25% useless store)
    """

    @staticmethod
    def n_bits():
        return 3

    @staticmethod
    def _pack(tensor):
        out = torch.zeros(
            [int(10 * np.ceil(tensor.shape[0] / 10.0)), tensor.shape[1]],
            device=tensor.device,
            dtype=torch.int32,
        )
        out[: len(tensor)] = tensor
        _step = int(len(out) / 10)
        out = (
            (out[:_step] << 27)
            | (out[_step : _step * 2] << 24)
            | (out[_step * 2 : _step * 3] << 21)
            | (out[_step * 3 : _step * 4] << 18)
            | (out[_step * 4 : _step * 5] << 15)
            | (out[_step * 5 : _step * 6] << 12)
            | (out[_step * 6 : _step * 7] << 9)
            | (out[7 * _step : _step * 8] << 6)
            | (out[_step * 8 : _step * 9] << 3)
            | (out[_step * 9 :])
        )
        return out

    def unpack(self):
        _step = self._tensor.shape[0]
        tmp = torch.empty(
            [10 * _step, self._tensor.shape[1]],
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        tmp[:_step] = (self._tensor & 0b00111000000000000000000000000000) >> 27
        tmp[1 * _step : 2 * _step] = (
            self._tensor & 0b00000111000000000000000000000000
        ) >> 24
        tmp[2 * _step : 3 * _step] = (
            self._tensor & 0b00000000111000000000000000000000
        ) >> 21
        tmp[3 * _step : 4 * _step] = (
            self._tensor & 0b00000000000111000000000000000000
        ) >> 18
        tmp[4 * _step : 5 * _step] = (
            self._tensor & 0b00000000000000111000000000000000
        ) >> 15
        tmp[5 * _step : 6 * _step] = (
            self._tensor & 0b00000000000000000111000000000000
        ) >> 12
        tmp[6 * _step : 7 * _step] = (
            self._tensor & 0b00000000000000000000111000000000
        ) >> 9
        tmp[7 * _step : 8 * _step] = (
            self._tensor & 0b00000000000000000000000111000000
        ) >> 6
        tmp[8 * _step : 9 * _step] = (
            self._tensor & 0b00000000000000000000000000111000
        ) >> 3
        tmp[9 * _step :] = self._tensor & 0b00000000000000000000000000000111
        return tmp


class Q3TensorInU8(QTensorPackedU8):
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
            [int(2 * np.ceil(tensor.shape[0] / 2.0)), tensor.shape[1]],
            device=tensor.device,
            dtype=torch.uint8,
        )
        out[: len(tensor)] = tensor
        _step = int(len(tensor) / 2)
        out = out[:_step] << 5 | out[1 * _step : 2 * _step] << 2
        return out

    def unpack(self):
        _step = self._tensor.shape[0]
        tmp = torch.empty(
            [2 * _step, self._tensor.shape[1]],
            dtype=torch.uint8,
            device=self._tensor.device,
        )
        tmp[:_step] = (self._tensor & 0b11100000) >> 5
        tmp[1 * _step : 2 * _step] = (self._tensor & 0b00011100) >> 2
        return tmp
