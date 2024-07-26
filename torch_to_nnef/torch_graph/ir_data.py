"""Abstractions used in torch_to_nnef internal graph data IR

The goal is that these elements are:
- extracted/parsed from PyTorch graph data structs
- translated to NNEF graph data structs

"""

import typing as T
from dataclasses import dataclass

import numpy as np
import torch

from torch_to_nnef.dtypes import (
    TORCH_TO_NUMPY_DTYPE,
    is_quantized_dtype,
    str_to_torch_dtype,
)
from torch_to_nnef.exceptions import (
    TorchToNNEFError,
    TorchToNNEFNotImplementedError,
    TorchUnableToTraceData,
)
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_SCALARIMPLICIT,
    INTTYPE_KIND,
    NUMBERTYPE_KIND,
    TUPLETYPE_KIND,
)
from torch_to_nnef.utils import NamedItem

UNKNOWN_TRACE_SHAPE_VALUE = 321


def cleanup_data_name(name: str) -> str:
    for sep in ["/", "[", "]", ".", "-"]:
        name = name.replace(sep, "_")
    return name.lower()


@dataclass
class Data(NamedItem):
    name: str
    data: T.Any

    def __post_init__(self):
        self.debug_name = self.name
        self.name = cleanup_data_name(self.name)

    @property
    def is_container(self) -> bool:
        return False

    @property
    def export_name(self) -> str:
        return cleanup_data_name(self.name)

    @property
    def shaped(self) -> bool:
        return True

    @property
    def typed(self):
        return True

    @property
    def shaped_and_typed(self) -> bool:
        return self.shaped and self.typed

    @property
    def tracable(self) -> bool:
        return self.shaped_and_typed

    @property
    def is_constant(self):
        raise TorchToNNEFNotImplementedError()

    def __hash__(self):
        return hash(self.name)


@dataclass
class TensorVariable(Data):
    shape: T.Optional[T.List[int]]
    dtype: T.Optional[torch.dtype]

    # used as reference in case of Op outputs
    data: T.Optional[torch.Tensor]

    quant: T.Optional[T.Dict[str, T.Any]] = None

    @property
    def slug(self) -> str:
        return (
            f"{self.export_name}: {self.dtype}@{self.shape}" + ""
            if not self.quant
            else f"q8(scale={self.quant['scale']}, "
            f"zerop={self.quant['zero_point']})"
        )

    def cast_float_inplace(self):
        if self.data is not None:
            self.data = self.data.float()
            self.dtype = self.data.dtype

    @property
    def np_dtype(self):
        assert self.dtype is not None
        return TORCH_TO_NUMPY_DTYPE[self.dtype]

    @property
    def rank(self) -> T.Optional[int]:
        if self.data is not None:
            return len(self.data.shape)
        return len(self.shape) if self.shape is not None else None

    @property
    def shaped(self) -> bool:
        return self.shape is not None

    @property
    def typed(self) -> bool:
        return bool(self.dtype)

    @property
    def is_constant(self) -> bool:
        return self.data is not None

    @property
    def tracable(self) -> bool:
        if is_quantized_dtype(self.dtype) and self.quant is None:
            return False
        return self.shaped_and_typed

    @property
    def tracing_data(self):
        """Generate data if is not fixed based on tensor information

        we use it to produce computation trace

        """
        if not self.tracable:
            raise TorchUnableToTraceData(self)

        if self.data is not None:
            return self.data

        data = torch.rand(
            [
                UNKNOWN_TRACE_SHAPE_VALUE if x is None else x
                for x in (self.shape or [])
            ]
        )
        if is_quantized_dtype(self.dtype):
            return torch.quantize_per_tensor(
                data,
                scale=self.quant["scale"],
                zero_point=self.quant["zero_point"],
                dtype=self.dtype,
            )
        return data.to(self.dtype)

    @classmethod
    def parse(cls, node_c_value: torch._C.Value) -> "TensorVariable":
        node_type = node_c_value.type()
        if node_type.kind() == INTTYPE_KIND:
            dtype = torch.int32
        else:
            if node_type.kind() == NUMBERTYPE_KIND:
                parent_node = node_c_value.node()
                if parent_node.kind() == ATEN_SCALARIMPLICIT:
                    node_type = parent_node.input().type()
                else:
                    raise NotImplementedError()
            stype = node_type.scalarType()
            dtype = str_to_torch_dtype(stype) if stype else None
        return cls(
            name=node_c_value.debugName(),
            shape=[1]
            if node_type.kind() == INTTYPE_KIND
            else node_type.sizes(),
            dtype=dtype,
            data=node_c_value.toIValue(),
            quant=None,
        )

    def into_tensor_variable(self):
        return self

    def __hash__(self):
        return hash(self.name)


@dataclass
class PythonConstant(Data):
    data: T.Any

    @property
    def is_constant(self) -> bool:
        return True

    @property
    def np_dtype(self) -> np.dtype:
        raise TorchToNNEFNotImplementedError()

    @property
    def tracable(self) -> bool:
        return True

    @property
    def tracing_data(self):
        return self.data

    def __hash__(self):
        return hash(self.name)

    def cast_float_inplace(self):
        self.data = float(self.data)

    def into_tensor_variable(self):
        data = self.data
        if not isinstance(data, torch.Tensor):
            if self.data == "none":
                raise TorchToNNEFError(
                    "'None' can not be transformed TensorVariable"
                )
            data = torch.tensor(self.data)
        return TensorVariable(
            name=self.name, data=data, shape=list(data.shape), dtype=data.dtype
        )


@dataclass
class BlobTorchScriptObject(Data):
    """Used only in Quantized Operators

    from our current obervation

    """

    @property
    def np_dtype(self) -> np.dtype:
        raise TorchToNNEFNotImplementedError()

    @property
    def tracing_data(self):
        return self.data

    def __hash__(self):
        return hash(self.name)


@dataclass
class TupleTensors(Data):
    """Used as transition object only

    None should be remaining once graph is fully expanded

    """

    data: T.List[TensorVariable]

    @property
    def slug(self) -> str:
        slugs = ", ".join(_.slug for _ in self.data)
        return f"tupleTensor({self.export_name})({slugs})"

    @property
    def dtype(self):
        return None

    @property
    def is_constant(self) -> bool:
        return all(data.is_constant for data in self.data)

    @property
    def is_container(self) -> bool:
        return True

    @classmethod
    def parse_from_tuple_type(
        cls, node_c_value: torch._C.Value
    ) -> "TupleTensors":
        node_type = node_c_value.type()
        name = node_c_value.debugName()
        assert node_type.kind() == TUPLETYPE_KIND
        elements = []
        for idx, elm in enumerate(node_type.elements()):
            stype = elm.scalarType()
            dtype = str_to_torch_dtype(stype) if stype else None
            elm_data = TensorVariable(
                name=f"{name}_{idx}",
                shape=elm.sizes(),
                dtype=dtype,
                data=None,
            )
            elements.append(elm_data)
        return TupleTensors(name, elements)

    def __hash__(self):
        return hash(self.slug)


TtupleOrVar = T.Union[TensorVariable, TupleTensors]


@dataclass
class FixedTensorList(Data):
    """FixedTensorList is a list that contains tensor constant or not"""

    data: T.Sequence[T.Union[TensorVariable, PythonConstant]]

    @property
    def slug(self) -> str:
        slugs = ", ".join(_.slug for _ in self.data)
        return f"fixedTensorList({self.export_name})({slugs})"

    @property
    def is_constant(self) -> bool:
        return all(data.is_constant for data in self.data)

    @property
    def tracing_data(self) -> T.List[torch.Tensor]:
        return [d.tracing_data for d in self.data]

    @property
    def is_container(self) -> bool:
        return True

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        datas = "".join(f"\t\t\t{d},\n" for d in self.data)
        return f"FixedTensorList(name='{self.name}', data=[\n{datas}\t\t])"
