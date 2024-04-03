import io
import sys
import tempfile
import typing as T
from pathlib import Path

import gguf
import numpy as np
import torch
from torch import nn

from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor.base import QTensor


class QTensorGGUF(QTensor):
    """GGUF tensor storage

    Aka tensor format used in Llama.cpp & GGML
    (2024-04-02)

    we define:
        bpw = bit per weight

    ./ggml/src/ggml-common.h
    ./ggml/src/ggml-quants.c

    This storage is heavily tweaked/optimal for LLM so,
    no guaranty it will perform best on other NN arch / ML tasks

    To our knowedge there is 3 kinds of formats:
    - old legacy formats (still in use in many models):
        (tensor shape need to be divisible by 32)
        Q{X}_0 -> symetric quant with per group quantization of 32 elements
            Q8_0 -> [f16 scale, 32x(8bits element)] -> 34 bytes per group -> 8,5 bpw
            Q4_0 -> [f16 scale, 32x(4bits element)] -> 18 bytes per group -> 4.5 bpw
            Q5_0 -> [f16 scale, 32x(5th bit of each element), 32x(4 first bits of each element)]
                        -> 22 bytes per group
                        -> 5.5 bpw

        Q{X}_1 -> asymetric quant with per group quantization of 32 elements
            Q8_1 -> [f16 scale, f16 min, 32x(8bits element)] -> 36 bytes per group -> 9 bpw
            Q4_1 -> [f16 scale, f16 min, 32x(4bits element)] -> 20 bytes per group -> 5 bpw
            Q5_1 -> [f16 scale, f16 min, 32x(5th bit of each element), 32x(4 first bit of each elements)]
                        -> 24 bytes per group
                        -> 6 bpw

    - new formats: With double quantization formats where qparams are quantized themselves,
            macro quantization parameters are used to dequantize qparams

        Format group elements by 256 (so tensor shape need to be divisible by 256)

        Q{X}_K:
            Q2_K -> [16x(4bits min, 4bits scale), 256x(2bits element), f16 macro scale, f16 macro min]
                        -> 84 bytes per group
                        -> 2,625 bpw

            Q3_K -> [
                    256x(3rd bit per element),
                    256x(2 first bits bit per element),
                    16x(6bits quantized scales),
                    f16 macro scale
                    ]
                        -> 110 bytes per group
                        -> 3,4375 bpw
            Q4_K -> [f16 macro scale, f16 macro min, 8x(6bits min, 6bits scale), 128x(4bits element)]
                        -> 80 bytes per group
                        -> 5 bpw
            Q5_K -> [
                    f16 macro scale,
                    f16 macro min,
                    8x(6bits min, 6bits scale),
                    256x(5th bit per element),
                    256x(4 first bits per element)
                    ]
                        -> 176 bytes per group
                        -> 5,5 bpw
            Q6_K -> ..
            Q8_K -> ..

    - new formats: with non-linearity or very low bit-width
        IQ{X}_{SIZE}:
            where SIZE can be:
                XXS, XS, S, M
            and X can be 1, 2, 3

            format not studied but:
                iq1_s --> ... --> 1.56 bpw
                iq1_m -> ... --> 1.75 bpw
                iq2_xxs --> [f16 scale, 256x(2bits element)] -> 2.0625 bpw
                iq2_xs --> [f16 macro scale, 256x(2bits element), ~n x qscale~] -> 2.3125 bpw
                iq2_xs --> ... -> 2.5625 bpw
                iq3_xxs --> ... -> 3.0625 bpw

            i-quants familly is also providing in some case non linear quantization, ending with "_nl" notation
            see: https://github.com/ggerganov/llama.cpp/discussions/5063#discussioncomment-8383732
            for performance

    However to date gguf 0.6.0 only reference familly Q_{X}_0, Q_{X}_1 and Q_{X}_K
        other are on main but not yet released

    warning!: elements order is not maintained packing is applied in tile
        in 4 bit by example on 32 element stored index would be in store:
            [0, 16, 1, 17, ..., 15, 31]

    Implementation details:
        As of 2024-04-02 we only rely on gguf python library,

        ggml modified: https://github.com/JulienBalianSonos/ggml.git
        is only meant for tract unittest generation purpose

        This limit us to only quantization but not dequantization implementation
        for production export, we do this because:
        GGML python library (on mainstream), is not well supported (more a POC)
        and it needs you to provide specific .so library to link against library via
        env variable.

    """

    def __init__(
        self,
        float_torch_tensor: torch.Tensor,
        gguf_data_type: int,  # : "GGUFDataType"
    ):
        super().__init__()
        if isinstance(float_torch_tensor, nn.Parameter):
            float_torch_tensor = float_torch_tensor.data
        self._float_torch_tensor = float_torch_tensor
        self.gguf_data_type = gguf_data_type

    @property
    def gguf_data_type_name(self) -> str:
        return next(
            name
            for name, value in vars(gguf.constants.GGMLQuantizationType).items()
            if value == self.gguf_data_type
        )

    def _write_tensor_in_gguf_file(
        self,
        filepath: Path,
        variable_name: str,
        np_float_tensor: np.ndarray,
        dtype: int,
    ):
        sys.stdout = io.StringIO()
        # silent noise message:
        # "gguf: This GGUF file is for little Endian only"
        try:
            gguf_writer = gguf.GGUFWriter(filepath, "tract_custom")
        finally:
            sys.stdout = sys.__stdout__  # restore stdout function

        # gguf_writer.add_block_count(1)
        gguf_writer.add_tensor(variable_name, np_float_tensor, raw_dtype=dtype)

        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()
        return filepath

    def _get_tensor_data_from_gguf_file(
        self, gguf_file_path: T.Union[Path, str], variable_name: str
    ):
        reader = gguf.GGUFReader(gguf_file_path)
        for tensor in reader.tensors:
            if tensor.name == variable_name:
                return tensor.data
        raise ValueError(
            f"not found tensor '{variable_name}' in gguf file: {gguf_file_path}"
        )

    @property
    def ggml_data_np_tensor(self) -> np.ndarray:
        with tempfile.TemporaryDirectory() as dir_path:
            filepath = Path(dir_path) / "a.gguf"
            self._write_tensor_in_gguf_file(
                filepath,
                "a",
                self._float_torch_tensor.numpy(),
                self.gguf_data_type,
            )
            qdata = self._get_tensor_data_from_gguf_file(filepath, "a")
        return qdata

    def to_torch_float_tensor(self):
        return self._float_torch_tensor

    def __repr__(self) -> str:
        try:
            return (
                f"{self.__class__.__name__}(shape={tuple(self._float_torch_tensor.shape)},"
                f" gguf_target_dtype={self.gguf_data_type_name})"
            )
        except AttributeError:
            return f"{self.__class__.__name__}(?)"


class QTensorGGUFExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensorGGUF

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
        """implementation with storage"""
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.dtypes import numpy_dtype_to_tract_str

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.primitive import base

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.primitive.base import add_nnef_operation

        q_tensor = node.op_ref
        out_node = node.outputs[0]
        nnef_tensor_ref = base.add_tensor_variable_node_as_nnef_tensor(
            g, out_node, name_to_tensor, prevent_variable=True
        )
        nnef_tensor_ref.qtensor = q_tensor  # main assign to allow corect dump
        add_nnef_operation(
            graph=g,
            type="tract_core_gguf_variable",
            inputs=None,
            outputs=nnef_tensor_ref,
            attribs={
                "gguf_filename": f"{out_node.export_name}.gguf",
                "gguf_tensor_name": out_node.export_name,
                "gguf_dtype": q_tensor.gguf_data_type_name,
                "shape": list(nnef_tensor_ref.shape),
                "output_datum_type": numpy_dtype_to_tract_str(
                    nnef_tensor_ref.dtype
                ),
            },
        )
        return ["tract_core"]
