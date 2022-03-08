import torch
import numpy as np


INT_TO_TORCH_DTYPE = {
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
    # TODO add qint8
    13: torch.quint8,
}
NUMPY_DTYPE_TO_TORCH = {
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
TORCH_DTYPE_TO_NUMPY = {v: k for k, v in NUMPY_DTYPE_TO_TORCH.items()}


def torch_typestr_to_type(torch_type_str: str):
    return NUMPY_DTYPE_TO_TORCH[torch_typestr_to_nptype(torch_type_str)]


def torch_typestr_to_nptype(torch_type_str: str):
    if torch_type_str == "QUInt8":
        return np.int8
    if torch_type_str == "Long":
        return np.int64
    if torch_type_str in ["Float", "float"]:
        return np.float32
    if torch_type_str == "int":
        return np.int32
    if torch_type_str in ["Bool", "bool"]:
        return np.bool_
    if torch_type_str in ["Half"]:
        return np.float16

    raise NotImplementedError(torch_type_str)
