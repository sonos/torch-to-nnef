import typing as T

import numpy as np
import torch

NUMPY_TO_TORCH_DTYPE = {
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.float: torch.float,
    np.double: torch.double,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int: torch.int,
    np.short: torch.short,
    np.long: torch.long,
    np.bool: torch.bool,
    np.bool_: torch.bool,
}
TORCH_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_TO_TORCH_DTYPE.items()}
# In both direction it's not a mapping 1<->1 so update is needed
TORCH_TO_NUMPY_DTYPE.update(
    {
        torch.quint8: np.uint8,
        torch.qint8: np.int8,
        torch.qint32: np.int32,
    }
)


STR_TO_NUMPY_DTYPE = {
    "QUInt8": np.int8,
    "Long": np.int64,
    "Float": np.float32,
    "float": np.float32,
    "int": np.int32,
    "Bool": np.bool_,
    "bool": np.bool_,
    "Half": np.float16,
}
NUMPY_DTYPE_TO_STR = {v: k for k, v in STR_TO_NUMPY_DTYPE.items()}
NUMPY_DTYPE_TO_STR.update({int: "Long"})


def str_to_torch_dtype(torch_type_str: str):
    return NUMPY_TO_TORCH_DTYPE[STR_TO_NUMPY_DTYPE[torch_type_str]]


def torch_dtype_to_str(torch_type):
    if torch_type == torch.quint8:
        torch_type = torch.int8
    if torch_type == torch.qint8:
        torch_type = torch.int8
    return NUMPY_DTYPE_TO_STR[TORCH_TO_NUMPY_DTYPE[torch_type]]


def is_quantized_dtype(dtype: T.Optional[torch.dtype]):
    return dtype in [torch.quint8, torch.qint8, torch.qint32]
