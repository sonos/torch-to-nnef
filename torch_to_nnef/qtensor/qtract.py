import struct
import typing as T
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor.base import QScalePerGroupF16, QScheme, QTensor

# header encoded in 2 bytes


class TractQuantDataType(str, Enum):
    """tract weight quantization formats

    * 2024-07-29 : q4_0 same as the GGML tensor format available

    """

    Q4_0 = "q4_0"


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
    def write_in_tract_dat_file(self, filepath: T.Union[str, Path]):
        raise TorchToNNEFNotImplementedError()


class QTensorTractScaleOnly(QTensorTract):
    """

    u8_values_tensor: is a "non-packed" tensor where each value
        is stored in 8bits regardless of the final storage format

    """

    @staticmethod
    def __new__(
        cls,
        fp_tensor,
        u8_values_tensor,
        qscheme,
        tract_quant_data_type,
        dequant_to_dtype,
        *args,
        **kwargs,
    ):
        return super().__new__(cls, fp_tensor, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return QTensorTractScaleOnly(
            super().clone(*args, **kwargs),
            self.u8_values_tensor,
            self.qscheme,
            self.tract_quant_data_type,
            self.dequant_to_dtype,
        )

    def to(self, *args, **kwargs):
        new_obj = QTensorTractScaleOnly(
            [],
            self.u8_values_tensor,
            self.qscheme,
            self.tract_quant_data_type,
            self.dequant_to_dtype,  # TODO: fix with .to params
        )
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = False
        return new_obj

    def __init__(
        self,
        fp_tensor: torch.Tensor,
        u8_values_tensor: torch.Tensor,
        qscheme: QScheme,
        tract_quant_data_type: TractQuantDataType,
        dequant_to_dtype=torch.float32,
    ):
        assert isinstance(
            tract_quant_data_type, TractQuantDataType
        ) and tract_quant_data_type.value.endswith("_0"), tract_quant_data_type
        assert isinstance(qscheme, QScalePerGroupF16), qscheme
        assert len(u8_values_tensor.shape) == 2, u8_values_tensor.shape

        # tract limited support of packing
        assert u8_values_tensor.shape[1] % 32 == 0, u8_values_tensor.shape
        super().__init__()
        self.u8_values_tensor = u8_values_tensor
        self.qscheme = qscheme
        self.tract_quant_data_type = tract_quant_data_type
        self.dequant_to_dtype = dequant_to_dtype
        self.requires_grad = False  # since it's a

    @classmethod
    def build_q4_0_from_min_max_calibration(
        cls, fp_tensor, percentile: float = 1.0
    ) -> "QTensorTractScaleOnly":
        if isinstance(fp_tensor, torch.nn.Parameter):
            fp_tensor = fp_tensor.data
        if len(fp_tensor.shape) != 2:
            raise TorchToNNEFNotImplementedError(
                f"tract does only support weight of shape 2d but found {fp_tensor.shape}"
            )
        with torch.no_grad():
            q_scheme, u8_values_tensor = QScalePerGroupF16.min_max_calibration(
                fp_tensor, n_bits=4, group_size=32, percentile=percentile
            )
            return cls(
                fp_tensor=fp_tensor,
                u8_values_tensor=u8_values_tensor,
                qscheme=q_scheme,
                tract_quant_data_type=TractQuantDataType.Q4_0,
                dequant_to_dtype=fp_tensor.dtype,
            )

    def to_torch_float_tensor(self):
        # TODO: dequant_to_dtype should be an arg and passed to .dequantize
        # only self.forward(...) should use self.dequant_to_dtype
        return self.qscheme.dequantize(self.u8_values_tensor).to(
            self.dequant_to_dtype
        )

    def _build_binary_dat_header(self) -> bytes:
        q4_0_hex_code = "4020"
        return DatBinHeaderBuilder(
            q4_0_hex_code, self.u8_values_tensor.shape
        ).to_bytes()

    def _build_binary_dat_content(self) -> bytes:
        # NOTE: implementation with multiple call to .tobytes, not tested if bottleneck

        # Q40 block: 1 scale (f16) followed by 16bytes, each one storing 2 values rank == 2.
        assert self.tract_quant_data_type == TractQuantDataType.Q4_0

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

    def write_in_tract_dat_file(self, filepath: T.Union[str, Path]):
        path = Path(filepath)
        assert not path.exists(), path
        bin_header = self._build_binary_dat_header()
        bin_content = self._build_binary_dat_content()
        with path.open("wb") as fh:
            fh.write(bin_header)
            fh.write(bin_content)


class QTensorTractExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensorTractScaleOnly

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kwargs,
    ):
        """implementation with storage"""

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.helper import add_nnef_operation

        q_tensor = node.op_ref
        out_node = node.outputs[0]
        out_node.data = (
            None  # very important to avoid linear/conv relying on q issues
        )
        nnef_tensor_ref = helper.add_tensor_variable_node_as_nnef_tensor(
            g, out_node, name_to_tensor, prevent_variable=True
        )
        nnef_tensor_ref.qtensor = q_tensor  # main assign to allow corect dump
        add_nnef_operation(
            graph=g,
            type="variable",
            inputs=None,
            outputs=nnef_tensor_ref,
            attribs={
                "custom_datatype": "tract_quant",
                "label": out_node.export_name,
                "shape": list(nnef_tensor_ref.shape),
            },
        )
        return ["tract_core"]
