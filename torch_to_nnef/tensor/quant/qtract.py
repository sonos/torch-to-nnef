import filecmp
import operator
import platform
import struct
import tempfile
import typing as T
from functools import reduce
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    TorchToNNEFImpossibleQuantization,
    TorchToNNEFNotImplementedError,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.tensor.quant.base import (
    QScalePerGroupF16,
    QTensor,
    qscale_per_group_f16_min_max_calibration,
)


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

    def __init__(
        self, *args, specific_machine: T.Optional[str] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.qscheme, QScalePerGroupF16), self.qscheme
        # tract limited support of packing
        assert self.qscheme.n_bits == 4, self.qscheme.n_bits
        self.decompressed_shape = self.decompress_to_u8().shape
        assert len(self.decompressed_shape) in [2, 3, 4], (
            self.decompressed_shape
        )
        assert reduce(operator.mul, self.decompressed_shape[1:]) % 32 == 0, (
            self.decompressed_shape
        )
        self.specific_machine = specific_machine

    def decompress(self):
        """tract dequantization depends on hardware

        typically dequantization happen with ops in f16
        on ARM and f32 (scale directly casted) on others

        so we overwrite the function to be consistant
        with tract.

        """
        assert self.qscheme is not None
        decompress_u8 = self.u8_blob
        for u8_compressor in reversed(self.u8_compressors):
            decompress_u8 = u8_compressor.decompress(decompress_u8)
        if (self.specific_machine and "arm" in self.specific_machine) or (
            self.specific_machine is None and "arm" in platform.machine()
        ):
            return self.qscheme.dequantize(
                decompress_u8, target_dtype=torch.float16
            ).to(self.dequant_to_dtype)
        return self.qscheme.dequantize(
            decompress_u8, target_dtype=self.dequant_to_dtype
        )

    def _build_binary_dat_header(self, post_tract_21_11: bool = False) -> bytes:
        if post_tract_21_11:
            q4_0_hex_code = "4030"
        else:
            q4_0_hex_code = "4020"
        return DatBinHeaderBuilder(
            q4_0_hex_code, self.decompressed_shape
        ).to_bytes()

    def _build_binary_dat_content(
        self, post_tract_21_11: bool = False
    ) -> bytes:
        # NOTE: implementation with multiple call to .tobytes, not tested if bottleneck

        n_bytes_per_group = 18
        tensor_flat = self.decompress_to_u8().clone().flatten()
        if post_tract_21_11:
            tensor_per_group = tensor_flat.reshape(-1, 2, 16)
            tensor_per_group[:, 1, :] <<= 4
            tensor_per_group = (
                tensor_per_group.sum(dim=1).numpy().astype(np.uint8)
            )
        else:
            tensor_per_group = tensor_flat.reshape(-1, 16, 2)
            tensor_per_group[:, :, 1] <<= 4
            tensor_per_group = (
                tensor_per_group.sum(dim=2).numpy().astype(np.uint8)
            )

        b_scales = self.qscheme.scale.numpy().view(np.byte)
        b_vals = tensor_per_group.view(np.byte)
        b_all = np.hstack([b_scales, b_vals])
        b_arr = b_all.tobytes()
        assert len(b_arr) % n_bytes_per_group == 0
        return b_arr

    def write_in_file(
        self,
        dirpath: T.Union[str, Path],
        label: str,
        inference_target: InferenceTarget = TractNNEF.latest(),
    ):
        path = Path(dirpath) / f"{label}.dat"
        if path.exists():
            # already created a variable dump with that name.
            # check we would produce identical serialized data
            with tempfile.TemporaryDirectory() as _td:
                td = Path(_td)
                self.write_in_file(td, label, inference_target=inference_target)
                new_path = td / f"{label}.dat"
                assert new_path.exists(), new_path
                if filecmp.cmp(path, new_path):
                    return
            raise TorchToNNEFNotImplementedError(
                "At least 2 variables in the NNEF graph, "
                f"share same Parameters: '{label}' but they try "
                "to use different data-type (likely quantization format). "
                "This variable collision as no resolution strategy yet."
            )
        assert not path.exists(), path
        assert isinstance(inference_target, TractNNEF), inference_target
        post_tract_21_11 = inference_target.version >= "0.21.11"
        bin_header = self._build_binary_dat_header(post_tract_21_11)
        bin_content = self._build_binary_dat_content(post_tract_21_11)
        with path.open("wb") as fh:
            fh.write(bin_header)
            fh.write(bin_content)


def fp_to_tract_q4_0_with_min_max_calibration(
    fp_tensor, percentile: float = 1.0
) -> QTensorTractScaleOnly:
    """Min-Max method to quantize float tensor to tract supported Q4_0"""
    q4_group_size = 32
    if isinstance(fp_tensor, torch.nn.Parameter):
        fp_tensor = fp_tensor.data

    if len(fp_tensor.shape) not in [2, 3, 4]:
        raise TorchToNNEFImpossibleQuantization(
            "tract Q4_0 does only support weight "
            f"of shape 2d, 3d or 4d but found {fp_tensor.shape}"
        )

    multiple_axis = 1
    if (
        reduce(operator.mul, fp_tensor.shape[multiple_axis:]) % q4_group_size
        != 0
    ):
        raise TorchToNNEFImpossibleQuantization(
            f"tract Q4_0 does only support weight with dim={multiple_axis} "
            f"divisible by {q4_group_size} but "
            f"found {fp_tensor.shape[multiple_axis:]}"
        )
    with torch.no_grad():
        q_scheme = qscale_per_group_f16_min_max_calibration(
            fp_tensor, n_bits=4, group_size=q4_group_size, percentile=percentile
        )
        return QTensorTractScaleOnly(
            fp_tensor,
            qscheme=q_scheme,
            dequant_to_dtype=fp_tensor.dtype,
        )
