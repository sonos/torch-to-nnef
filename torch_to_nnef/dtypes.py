"""Module referencing all data types conversions between libraries.

List of libraries being:PyTorch, Numpy, tract
"""

import typing as T

import numpy as np
import torch

from torch_to_nnef.utils import torch_version

WHOLE_NUMBER_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
]
if torch_version() >= "2.4.0":
    WHOLE_NUMBER_DTYPES.extend(
        [
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ]
    )

NUMPY_TO_TORCH_DTYPE = {
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.double: torch.double,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.bool_: torch.bool,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    # np.float: torch.float,
    # np.bool: torch.bool,
    # next mapping are avoided as they map silently to builtin python type
    # np.int: torch.int,
    # np.short: torch.short,
    # np.long: torch.long,
}
if torch_version() >= "2.4.0":
    NUMPY_TO_TORCH_DTYPE[np.uint16] = torch.uint16
    NUMPY_TO_TORCH_DTYPE[np.uint32] = torch.uint32
    NUMPY_TO_TORCH_DTYPE[np.uint64] = torch.uint64

TORCH_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_TO_TORCH_DTYPE.items()}
# In both direction it's not a mapping 1<->1 so update is needed
TORCH_TO_NUMPY_DTYPE.update(
    {
        torch.quint8: np.uint8,
        torch.qint8: np.int8,
        torch.qint32: np.int32,
    }
)

# borrowed from torch.onnx.symbolic_helper
SCALAR_TYPE_TO_PYTORCH_TYPE = [
    torch.uint8,  # 0
    torch.int8,  # 1
    torch.short,  # 2
    torch.int,  # 3
    torch.int64,  # 4
    torch.half,  # 5
    torch.float,  # 6
    torch.double,  # 7
    None,  # 8
    torch.complex64,  # 9
    torch.complex128,  # 10
    torch.bool,  # 11
    torch.qint8,  # 12
    torch.quint8,  # 13
    torch.qint32,  # 14
    torch.bfloat16,  # 15
]


STR_TO_NUMPY_DTYPE = {
    "QUInt8": np.int8,
    "Byte": np.uint8,
    "Long": np.int64,
    "Float": np.float32,
    "float": np.float32,
    "Short": np.int16,
    "short": np.int16,
    "Char": np.int8,
    "char": np.int8,
    "Double": np.float64,
    "double": np.float64,
    "i64": np.int64,
    "I64": np.int64,
    "int": np.int32,
    "Int": np.int32,
    "Bool": np.bool_,
    "bool": np.bool_,
    "Half": np.float16,
    "ComplexFloat": np.complex64,
    "TDim": np.bool_,
    "tdim": np.bool_,
    "UInt16": np.uint16,
    "uint16": np.uint16,
}

NUMPY_DTYPE_TO_STR = {v: k for k, v in STR_TO_NUMPY_DTYPE.items()}
NUMPY_DTYPE_TO_STR.update({int: "Long"})


TORCH_DTYPE_TO_NNEF_STR = {
    torch.int8: "integer",
    torch.int16: "integer",
    torch.int32: "integer",
    torch.int64: "integer",
    torch.uint8: "integer",
    torch.float: "scalar",
    torch.double: "scalar",
    torch.float16: "scalar",
    torch.float32: "scalar",
    torch.float64: "scalar",
    torch.bool: "logical",
}
if torch_version() >= "2.4.0":
    TORCH_DTYPE_TO_NNEF_STR[torch.uint16] = "integer"

TORCH_DTYPE_TO_TRACT_STR = {
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.float16: "f16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.complex64: "complexf64",
    torch.complex128: "complexf128",
    torch.bool: "bool",
}

if torch_version() >= "2.4.0":
    TORCH_DTYPE_TO_TRACT_STR[torch.uint16] = "u16"
    TORCH_DTYPE_TO_TRACT_STR[torch.uint32] = "u32"
    TORCH_DTYPE_TO_TRACT_STR[torch.uint64] = "u64"


def str_to_torch_dtype(torch_type_str: str):
    return NUMPY_TO_TORCH_DTYPE[STR_TO_NUMPY_DTYPE[torch_type_str]]


def torch_dtype_to_str(torch_type):
    if torch_type == torch.quint8:
        torch_type = torch.int8
    if torch_type == torch.qint8:
        torch_type = torch.int8
    return NUMPY_DTYPE_TO_STR[TORCH_TO_NUMPY_DTYPE[torch_type]]


def numpy_dtype_to_tract_str(dtype) -> str:
    if dtype in [
        np.uint16,
        np.uint32,
        np.uint64,
    ]:
        return {np.uint16: "u16", np.uint32: "u32", np.uint64: "u64"}[dtype]
    torch_dtype = NUMPY_TO_TORCH_DTYPE[dtype]
    return TORCH_DTYPE_TO_TRACT_STR[torch_dtype]


def is_quantized_dtype(dtype: T.Optional[torch.dtype]):
    return dtype in [torch.quint8, torch.qint8, torch.qint32]


def dtype_is_whole_number(dtype):
    if "numpy" in str(dtype):
        dtype = NUMPY_TO_TORCH_DTYPE[dtype]
    return dtype in WHOLE_NUMBER_DTYPES
