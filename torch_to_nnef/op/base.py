import typing as T

import numpy as np
import torch
from torch import nn

from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.torch_graph import NodeTensorSized

__all__ = ["ModuleInfoExtractor"]


def _torch_to_nnef_typestr(torch_type_str: str):

    if torch_type_str == "QUInt8":
        return np.int8
    if torch_type_str == "Long":
        return np.int64
    if torch_type_str == "Float":
        return np.float32
    if torch_type_str == "int":
        return np.int32
    if torch_type_str == "bool":
        return np.bool_

    raise NotImplementedError(torch_type_str)


class _ModuleInfoRegistery(type):

    """Allow extract in NNEF behavior from specific nn.Module"""

    JIT_CLASS_NAME = None

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        if new_cls.JIT_CLASS_NAME is not None:
            cls.REGISTRY[new_cls.JIT_CLASS_NAME] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


class ModuleInfoExtractor(metaclass=_ModuleInfoRegistery):

    """Allow extract in NNEF behavior from specific nn.Module"""

    JIT_CLASS_NAME = None

    def __init__(self, node: NodeTensorSized, module: nn.Module, g: NGraph):
        assert self.JIT_CLASS_NAME is not None
        self.node = node
        self.mod = module
        self.g = g

    def _quant_from_ref(self, tensor, ref_qtensor):
        return torch.quantize_per_tensor(
            tensor,
            scale=ref_qtensor.q_scale(),
            zero_point=ref_qtensor.q_zero_point(),
            dtype=ref_qtensor.dtype,
        )

    def _torch_qtensor_to_ntensor(self, tensor, name):
        np_int_tensor = tensor.int_repr().numpy()
        return NTensor(
            self.g,
            name=name,
            shape=tuple(tensor.shape),
            dtype=np_int_tensor.dtype.type,
            data=np_int_tensor,
            quant={
                "scale": tensor.q_scale(),
                "zero_point": tensor.q_zero_point(),
                "bits": 8,
                "signed": True,
                "symmetric": False,
                "op-name": "zero_point_linear_quantize",
            },
        )

    def _add_idem_type_shape_output_tensor(
        self, name_to_tensor, force_dtype=None
    ):
        out_tensor_name = self.node.export_name
        input_node = name_to_tensor[self.node.export_inputs[1]]
        output_tensor = NTensor(
            graph=self.g,
            name=out_tensor_name,
            dtype=force_dtype or input_node.dtype,
            shape=input_node.shape,
        )
        name_to_tensor[out_tensor_name] = output_tensor
        return output_tensor

    def add_quantized_tensor_to_ngraph(
        self,
        qtensor: torch.Tensor,
        tensor_name: str,
        name_to_tensors: T.Dict[str, NTensor],
    ):
        name = f"{self.node.export_name}_{tensor_name}"
        ntensor = self._torch_qtensor_to_ntensor(qtensor, name)
        name_to_tensors[name] = ntensor
        return ntensor

    def add_tensor_to_ngraph(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
        name_to_tensors: T.Dict[str, NTensor],
    ):
        name = f"{self.node.export_name}_{tensor_name}"
        tensor_np = tensor.numpy()
        ntensor = NTensor(
            self.g,
            name=name,
            shape=tuple(tensor.shape),
            dtype=tensor_np.dtype.type,
            data=tensor_np,
        )
        name_to_tensors[name] = ntensor
        return ntensor

    def extract_extra_tensor_from_module(
        self, name_to_tensor: T.Dict[str, NTensor]
    ):
        return name_to_tensor

    def extract_operations(self, name_to_tensor: T.Dict[str, NTensor]):
        raise NotImplementedError()

    def extract(self, name_to_tensor: T.Dict[str, NTensor]):
        self.extract_extra_tensor_from_module(name_to_tensor)
        self.extract_operations(name_to_tensor)
