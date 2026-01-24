"""Abstractions used in torch_to_nnef internal graph data IR.

The goal is that these elements are:
- extracted/parsed from PyTorch graph data structs
- translated to NNEF graph data structs

"""

import re
import typing as T
from dataclasses import dataclass

import numpy as np
import torch

from torch_to_nnef.dtypes import (
    TORCH_TO_NUMPY_DTYPE,
    dtype_is_whole_number,
    is_quantized_dtype,
    str_to_torch_dtype,
)
from torch_to_nnef.exceptions import (
    T2NError,
    T2NErrorConsistency,
    T2NErrorIRDataConsistency,
    T2NErrorNotImplemented,
    T2NErrorTorchNotFoundDataNode,
    T2NErrorTorchUnableToTraceData,
)
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_SCALARIMPLICIT,
    CONSTANT_KIND,
    DICTTYPE_KIND,
    INTTYPE_KIND,
    NUMBERTYPE_KIND,
    TUPLETYPE_KIND,
)
from torch_to_nnef.utils import NamedItem, ReactiveNamedItemDict

UNKNOWN_TRACE_SHAPE_VALUE = 321


def cleanup_data_name(name: str) -> str:
    for sep in ["/", "[", "]", ".", "-"]:
        name = re.sub(r"\s+", "_", name.replace(sep, "_"))
    return name


@dataclass
class Data(NamedItem):
    """Base abstract T2N IR data holder."""

    name: str
    data: T.Any

    def set_data(self, data: T.Any, *args, **kwargs):
        self.data = data

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
        return False

    @property
    def typed(self) -> bool:
        return False

    @property
    def shaped_and_typed(self) -> bool:
        return self.shaped and self.typed

    @property
    def tracable(self) -> bool:
        return self.shaped_and_typed

    @property
    def is_constant(self):
        raise T2NErrorNotImplemented()

    def __hash__(self):
        return hash(self.name)


class TensorDataSlot:
    """Prevent uncontrolled assignation of tensor data in Data nodes."""

    def __init__(self):
        self.private = "_data"

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private, None)

    def __set__(self, obj, value):
        # Allow first assignment only
        if hasattr(obj, self.private):
            raise RuntimeError(
                f"{obj.name}: direct reassignment of .data is forbidden; "
                f"use set_data(...) instead"
            )
        setattr(obj, self.private, value)


@dataclass
class TensorVariable(Data):
    shape: T.Optional[T.List[int]]
    dtype: T.Optional[torch.dtype]

    # used as reference in case of Op outputs
    data = TensorDataSlot()

    quant: T.Optional[T.Dict[str, T.Any]] = None
    _traced_data: T.Optional[torch.Tensor] = None

    def set_data(  # pylint: disable=arguments-differ
        self, data, *args, force_shape=False, force_dtype=False, **kwargs
    ):
        if force_shape and self.shape is not None:
            self.shape = list(data.shape)

        if force_dtype and self.dtype is not None:
            self.dtype = data.dtype

        # Explicit mutation path
        self._validate_data(data)
        object.__setattr__(self, "_data", data)

    def _validate_data(self, value):
        if value is None:
            return

        if isinstance(value, torch.Tensor):
            if (
                not (
                    len(value.shape) == 0
                    and len(self.shape) == 1
                    and self.shape[0] == 1
                )
                or len(value.shape) == 1
                and len(self.shape) == 0
                and value.shape[0] == 1
            ) and list(value.shape) != self.shape:
                raise T2NErrorIRDataConsistency(
                    f"{self.name}: shape mismatch {value.shape} != {self.shape}"
                )
            if value.dtype != self.dtype:
                raise T2NErrorIRDataConsistency(
                    f"{self.name}: dtype mismatch {value.dtype} != {self.dtype}"
                )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.data is None:
            return

        if self.dtype is None or self.shape is None:
            raise T2NErrorIRDataConsistency(
                "dtype and shape must be specified when data is provided"
            )

        if self.dtype != self.data.dtype:
            raise T2NErrorIRDataConsistency(
                f"dtype mismatch: declared {self.dtype}, "
                f"but data has dtype {self.data.dtype}"
            )

        # shape validation / inference
        data_shape = list(self.data.shape)
        if self.shape is None:
            self.shape = data_shape
        if list(self.shape) != data_shape:
            raise T2NErrorIRDataConsistency(
                f"shape mismatch: declared {self.shape}, "
                f"but data has shape {data_shape}"
            )

    @property
    def slug(self) -> str:
        return (
            f"{self.export_name}: {self.dtype}@{self.shape}" + ""
            if not self.quant
            else (
                f"q8(scale={self.quant['scale']}, "
                f"zerop={self.quant['zero_point']})"
            )
        )

    def cast_float_inplace(self):
        if self.data is not None:
            self.set_data(self.data.float(), force_dtype=True)

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
    def volume(self) -> T.Optional[int]:
        if self.shape is None:
            return None
        vol = 1
        for s in self.shape:
            vol *= s
        return vol

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
        st = self.shaped_and_typed
        # whole number can be used for indexing or tensor gen ...
        # those may be significan for subsequent op in network ...
        need_data_realisation = (
            dtype_is_whole_number(self.dtype)
            and self._traced_data is None
            and self.data is None
        )
        return st and not need_data_realisation

    @property
    def tracing_data(self):
        """Generate data if is not fixed based on tensor information.

        we use it to produce computation trace

        """
        if not self.tracable:
            raise T2NErrorTorchUnableToTraceData(self)

        if self.data is not None:
            return self.data
        if self._traced_data is None:
            if dtype_is_whole_number(self.dtype):
                raise T2NErrorConsistency(f"whole number need {self}")
        else:
            return self._traced_data

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
            dtype = torch.int64
        else:
            if node_type.kind() == NUMBERTYPE_KIND:
                parent_node = node_c_value.node()
                if parent_node.kind() == ATEN_SCALARIMPLICIT:
                    node_type = parent_node.input().type()
                else:
                    raise T2NErrorNotImplemented()
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

    def __eq__(self, other) -> bool:
        # check by name first
        # since faster than isinstance
        try:
            if self.name != other.name or not isinstance(other, TensorVariable):
                return False
        except AttributeError:
            return False
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.dtype == other.dtype
            and (
                (self.data == other.data).all().item()
                if self.data is not None and other.data is not None
                else self.data == other.data
            )
            and self.quant == other.quant
        )

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
        raise T2NErrorNotImplemented()

    @property
    def shaped(self) -> bool:
        return True

    @property
    def typed(self) -> bool:
        return True

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
                raise T2NError("'None' can not be transformed TensorVariable")
            data = torch.tensor(self.data)
        return TensorVariable(
            name=self.name, data=data, shape=list(data.shape), dtype=data.dtype
        )


@dataclass
class BlobTorchScriptObject(Data):
    """Used only in Quantized Operators from our current obervation."""

    @property
    def np_dtype(self) -> np.dtype:
        raise T2NErrorNotImplemented()

    @property
    def tracable(self) -> bool:
        return True

    @property
    def tracing_data(self):
        return self.data

    def __hash__(self):
        return hash(self.name)


@dataclass
class TupleTensors(Data):
    """Used as transition object only.

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
    def shaped(self) -> bool:
        return all(_.shaped for _ in self.data)

    @property
    def typed(self) -> bool:
        return all(_.typed for _ in self.data)

    @property
    def is_constant(self) -> bool:
        return all(data.is_constant for data in self.data)

    @property
    def is_container(self) -> bool:
        return True

    def iter(self):
        return self.data

    @classmethod
    def parse_from_tuple_type(
        cls, node_c_value: torch._C.Value
    ) -> "TupleTensors":
        node_type = node_c_value.type()
        name = node_c_value.debugName()
        assert node_type.kind() == TUPLETYPE_KIND
        elements = []
        for idx, elm in enumerate(node_type.elements()):
            if elm.kind() == "TensorType":
                stype = elm.scalarType()
                shape = elm.sizes()
            elif elm.kind() == "IntType":
                stype = "int"
                shape = [1]
            else:
                raise T2NErrorNotImplemented(elm.kind())
            dtype = str_to_torch_dtype(stype) if stype else None
            elm_data = TensorVariable(
                name=f"{name}_{idx}",
                shape=shape,
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
    """FixedTensorList is a list that contains tensor constant or not."""

    data: T.Sequence[T.Union[TensorVariable, PythonConstant]]

    @property
    def slug(self) -> str:
        slugs = ", ".join(_.slug for _ in self.data)
        return f"fixedTensorList({self.export_name})({slugs})"

    @property
    def is_constant(self) -> bool:
        return all(data.is_constant for data in self.data)

    @property
    def shaped(self) -> bool:
        return all(_.shaped for _ in self.data)

    @property
    def typed(self) -> bool:
        return all(_.typed for _ in self.data)

    @property
    def tracing_data(self) -> T.List[torch.Tensor]:
        return [d.tracing_data for d in self.data]

    @property
    def is_container(self) -> bool:
        return True

    def iter(self):
        return self.data

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        datas = "".join(f"\t\t\t{d},\n" for d in self.data)
        return f"FixedTensorList(name='{self.name}', data=[\n{datas}\t\t])"


@dataclass
class DictTensors(Data):
    """Used as transition object only.

    None should be remaining once graph is fully expanded

    """

    data: T.Dict[str, TensorVariable]

    @property
    def slug(self) -> str:
        slugs = ", ".join(f"{k}: {n.slug}" for k, n in self.data.items())
        return f"dictTensor({self.export_name})({slugs})"

    @property
    def dtype(self):
        return None

    @property
    def shaped(self) -> bool:
        return all(_.shaped for _ in self.data.values())

    @property
    def typed(self) -> bool:
        return all(_.typed for _ in self.data.values())

    @property
    def is_constant(self) -> bool:
        return all(data.is_constant for data in self.data.values())

    @property
    def is_container(self) -> bool:
        return True

    def iter(self):
        return self.data.values()

    @classmethod
    def parse_from_dic_node_c_value(
        cls, node_c_value: torch._C.Value, data_nodes: ReactiveNamedItemDict
    ) -> "DictTensors":
        node_type = node_c_value.type()
        name = node_c_value.debugName()
        assert node_type.kind() == DICTTYPE_KIND
        elements = {}
        key = None
        for idx, c_val in enumerate(node_c_value.node().inputs()):
            if idx % 2 == 0:
                if (
                    str(c_val.type()) in ["str", "int"]
                    and c_val.node().kind() == CONSTANT_KIND
                ):
                    key = c_val.node()["value"]
                else:
                    raise T2NErrorNotImplemented()
            else:
                assert key is not None
                try:
                    elm_data = data_nodes.get_by_name(
                        cleanup_data_name(c_val.debugName())
                    )
                except T2NErrorTorchNotFoundDataNode:
                    ctype = c_val.type()
                    stype = ctype.scalarType()
                    dtype = str_to_torch_dtype(stype) if stype else None
                    elm_data = TensorVariable(
                        name=c_val.debugName(),
                        shape=ctype.sizes(),
                        dtype=dtype,
                        data=None,
                    )
                elements[key] = elm_data
        return DictTensors(name, elements)

    def __hash__(self):
        return hash(self.slug)
