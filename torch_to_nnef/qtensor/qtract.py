import struct
import typing as T
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.qtensor.base import QScalePerGroupF16, QTensor


class DatBinHeaderBuilder:
    TRACT_ITEM_TYPE_VENDOR = struct.pack(
        "h", struct.unpack("B", b"T")[0] << 8 | struct.unpack("B", b"R")[0]
    )

    def __init__(self, item_type, shape):
        # magic: [u8; 2],
        self.magic = bytes.fromhex("4eef")

        # version_maj: u8,
        self.version_maj = struct.pack("B", 1)

        # version_min: u8,
        self.version_min = struct.pack("B", 0)

        # data_size_bytes: u32,
        vol = 1
        for s in shape:
            vol *= s
        data_size_in_bytes = int((vol * 4 + vol / 32 * 16) / 8)
        self.data_size_bytes = struct.pack("I", data_size_in_bytes)

        # rank: u32
        self.rank = struct.pack("I", len(shape))

        # dims: [u32; 8],
        sh = list(shape)
        sh += [0] * (8 - len(sh))
        self.dims = struct.pack("8I", *sh)

        # bits_per_item: u32,
        self.bits_per_item = bytes.fromhex("FFFFFFFF")

        # item_type: u16,
        self.item_type = bytes.fromhex(item_type)

        # item_type_vendor: u16,
        self.item_type_vendor = self.TRACT_ITEM_TYPE_VENDOR

        # item_type_params_deprecated: [u8; 32],
        self.item_type_params_deprecated = struct.pack("32B", *([0] * 32))

        # padding: [u32; 11],
        self.padding = struct.pack("11I", *([0] * 11))

    def to_bytes(self):
        b_arr = bytearray(b"")
        b_arr.extend(self.magic)
        b_arr.extend(self.version_maj)
        b_arr.extend(self.version_min)
        b_arr.extend(self.data_size_bytes)
        b_arr.extend(self.rank)
        b_arr.extend(self.dims)
        b_arr.extend(self.bits_per_item)
        b_arr.extend(self.item_type)
        b_arr.extend(self.item_type_vendor)
        b_arr.extend(self.item_type_params_deprecated)
        b_arr.extend(self.padding)
        binheader = bytes(b_arr)
        assert len(binheader) == 128, len(binheader)
        return binheader


class QTensorTract(QTensor):
    """All QTensorTract implementations"""


class QTensorTractScaleOnly(QTensorTract):
    """

    Tract data format it serialize to: Q4_0

    """

    qscheme: QScalePerGroupF16  # type notation for mypy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.qscheme, QScalePerGroupF16), self.qscheme

        # tract limited support of packing
        assert self.qscheme.n_bits == 4, self.qscheme.n_bits
        assert (
            len(self.u8_values_tensor.shape) == 2
        ), self.u8_values_tensor.shape
        assert (
            self.u8_values_tensor.shape[1] % 32 == 0
        ), self.u8_values_tensor.shape

    def _build_binary_dat_header(self) -> bytes:
        q4_0_hex_code = "4020"
        return DatBinHeaderBuilder(
            q4_0_hex_code, self.u8_values_tensor.shape
        ).to_bytes()

    def _build_binary_dat_content(self) -> bytes:
        # NOTE: implementation with multiple call to .tobytes, not tested if bottleneck

        n_bytes_per_group = 18
        tensor_per_group = (
            self.u8_values_tensor.clone().flatten().reshape(-1, 16, 2)
        )
        tensor_per_group[:, :, 0] <<= 4
        tensor_per_group = tensor_per_group.sum(dim=2).numpy().astype(np.uint8)

        b_arr = bytearray(b"")
        for values, scale in zip(
            tensor_per_group, self.qscheme.scale.flatten().numpy()
        ):
            b_arr.extend(scale.tobytes("F"))
            b_arr.extend(values.tobytes("F"))
            assert len(b_arr) % n_bytes_per_group == 0
        return bytes(b_arr)

    def write_in_file(self, dirpath: T.Union[str, Path], label: str):
        path = Path(dirpath) / f"{label}.dat"
        assert not path.exists(), path
        bin_header = self._build_binary_dat_header()
        bin_content = self._build_binary_dat_content()
        with path.open("wb") as fh:
            fh.write(bin_header)
            fh.write(bin_content)


def fp_to_tract_q4_0_with_min_max_calibration(
    fp_tensor, percentile: float = 1.0
) -> QTensorTractScaleOnly:
    """Min-Max method to quantize float tensor to Q4_0"""
    if isinstance(fp_tensor, torch.nn.Parameter):
        fp_tensor = fp_tensor.data
    if len(fp_tensor.shape) != 2:
        raise TorchToNNEFImpossibleQuantization(
            f"tract Q4_0 does only support weight of shape 2d but found {fp_tensor.shape}"
        )
    if fp_tensor.shape[1] % 32 != 0:
        raise TorchToNNEFImpossibleQuantization(
            f"tract Q4_0 does only support weight with 2nd dim "
            f"divisible by 32 but found {fp_tensor.shape[1]}"
        )
    with torch.no_grad():
        q_scheme, u8_values_tensor = QScalePerGroupF16.min_max_calibration(
            fp_tensor, n_bits=4, group_size=32, percentile=percentile
        )
        return QTensorTractScaleOnly(
            u8_values_tensor=u8_values_tensor,
            qscheme=q_scheme,
            dequant_to_dtype=fp_tensor.dtype,
        )
