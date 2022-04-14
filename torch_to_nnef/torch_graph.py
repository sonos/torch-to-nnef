# pylint: disable=too-many-lines
"""
torch_graph is intended to extract full representation of pytorch Graph
into a stable intermediate representation suitable to then apply translation
operation to NNEF. This means that not all Pytorch orginal graph is translated.
By example, we ignore part linked to device location informations,
memory specific operation or parameters linked to gradients.

This choice which is different compared to torch.onnx module due to the
absence of control (on our side) over evolution of Pytorch internals.
If some of the Pytorch internals are modified only this module should idealy
be impacted.

"""


import logging
import typing as T
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.jit._trace
from torch import jit, nn

from torch_to_nnef.console import Console
from torch_to_nnef.dtypes import (
    TORCH_TO_NUMPY_DTYPE,
    is_quantized_dtype,
    str_to_torch_dtype,
)
from torch_to_nnef.op.custom_extractors import (
    ModuleInfoExtractor,
    NotFoundModuleExtractor,
)
from torch_to_nnef.tract import nop
from torch_to_nnef.utils import cache

LOGGER = logging.getLogger(__name__)


class JitTraceFailed(RuntimeError):
    pass


class UnableToTraceData(ValueError):
    pass


class TorchOpTranslatedDifferently(ValueError):
    pass


class NotFoundDataNode(ValueError):
    pass


class NotFoundTorchOp(ValueError):
    pass


class CheckError(ValueError):
    pass


UNKNOWN_TRACE_SHAPE_VALUE = 321

PRIM_STARTID = "prim::"
CALL_KIND = "prim::CallMethod"
CONSTANT_KIND = "prim::Constant"
GETATTR_KIND = "prim::GetAttr"
LISTCONSTRUCT_KIND = "prim::ListConstruct"
PARAM_KIND = "prim::Param"
TUPLECONSTRUCT_KIND = "prim::TupleConstruct"
TUPLEUNPACK_KIND = "prim::TupleUnpack"
LISTUNPACK_KIND = "prim::ListUnpack"
NUMTOTENSOR_KIND = "prim::NumToTensor"

ATEN_STARTID = "aten::"
ATEN_CONTIGUOUS_KIND = "aten::contiguous"
ATEN_VIEW_KIND = "aten::view"
ATEN_SIZE_KIND = "aten::size"
ATEN_INT = "aten::Int"
ATEN_ARANGE = "aten::arange"


CLASSTYPE_KIND = "ClassType"
TUPLETYPE_KIND = "TupleType"
LISTTYPE_KIND = "ListType"
NONETYPE_KIND = "NoneType"
INTTYPE_kIND = "IntType"

MODULE_PATH_ATEN = "TORCH_INTERNAL_ATEN"
MODULE_PATH_QUANTIZED = "TORCH_INTERNAL_QUANTIZED"
SPECIAL_ATEN_REMAP_PYTORCH = {"__and__": "bitwise_and", "__or__": "bitwise_or"}

MAP_TO_NOP = [NUMTOTENSOR_KIND, LISTCONSTRUCT_KIND]
MAP_TO_TENSOR_FN = [ATEN_CONTIGUOUS_KIND, ATEN_VIEW_KIND]


def aten_name_to_torch_fn(aten_name):
    name = aten_name.replace(ATEN_STARTID, "")
    return getattr(torch.ops.aten, name)


def quantized_name_to_torch_fn(aten_name):
    name = aten_name.replace("quantized::", "")
    return getattr(torch.ops.quantized, name)


def _add_prefix_if_start_with_digit(text: str, prefix: str) -> str:
    """ensure we do not start with integer a text"""
    _prefix = ""
    if text[0] in "0123456789":
        _prefix = prefix
    return _prefix + text


def _refid_clean(name: str) -> str:
    for sep in ["/", "[", "]", ".", "-"]:
        name = name.replace(sep, "_")
    return name.lower()


def _parse_traced_name(module):
    if isinstance(module, jit.TracedModule):
        module_name = module._name
    else:
        module_name = getattr(module, "original_name", "Module")
    return module_name


def _is_container(data_node: "Data"):
    return isinstance(data_node, (FixedTensorList, TupleTensors))


def _expand_containers_if_exists(data_items):
    for data_item in data_items:
        if _is_container(data_item):
            yield from data_item.data
        yield data_item


def _unfold_graph_getattr_by_node(
    module: T.Union[nn.Module, torch.jit.TracedModule],
    getattr_node: torch._C.Node,
) -> T.Tuple[str, T.Union[nn.Module, torch.jit.TracedModule]]:
    """Unfold  nn.Module python code reference to sub...sub modules in graph"""
    getter_sequence: T.List[str] = []

    def getter_fn(getattr_node: torch._C.Node, getter_sequence: T.List[str]):
        try:
            getter_sequence.append(getattr_node.s("name"))
        except RuntimeError:
            # ensure we are at root reference
            dname = next(getattr_node.outputs()).debugName()
            assert dname.startswith("self"), dname

    getter_fn(getattr_node, getter_sequence)
    while getattr_node.kind() in [GETATTR_KIND, CALL_KIND]:
        c_value = next(getattr_node.inputs())
        getattr_node = c_value.node()
        getter_fn(getattr_node, getter_sequence)

    getter_sequence = getter_sequence[::-1]
    submodule = module
    for getter_item in getter_sequence:
        submodule = getattr(submodule, getter_item)

    return ".".join(getter_sequence), submodule


def _reconstruct_view_dims(
    original_shape: T.Tuple[int, ...], wished_view: T.Tuple[int, ...]
) -> T.Tuple[int, ...]:
    """Reconstruct shapes of whished view

    By example:
        x_reshape = x.contiguous().view((-1,) + x.shape[2:])

        Provide incomplete graph information about shapes filing info with None
        as such:

        view((-1, None, None))

        but we know input by example:

        (4, 64, 16, 128)

        so we can solve real view dims to be:

        (-1, 16, 128)

    """
    assert len(original_shape) >= len(wished_view)
    assert sum(_ == -1 for _ in wished_view) == 1, "impossible to guess ?"
    completed_wished_view = list(wished_view)[:]

    # try forward
    for rank, dim_at_rank in enumerate(wished_view):
        if dim_at_rank == -1:
            # unable to guess further dimensions
            break
        completed_wished_view[rank] = original_shape[rank]

    # try backward
    for rank, dim_at_rank in enumerate(wished_view[::-1]):
        if dim_at_rank == -1:
            # unable to guess further dimensions
            break
        completed_wished_view[-rank - 1] = original_shape[-rank - 1]

    assert len(wished_view) == len(completed_wished_view)
    assert None not in completed_wished_view
    return tuple(completed_wished_view)


def _find_common_root(
    elements: T.Iterable[str], sep: str, shortest: bool = False, base: str = ""
) -> str:
    common_root = base

    # `scope_name_appeared` is recursive tree
    # so not handled by mypy type anotation yet
    # see https://github.com/python/mypy/issues/731
    scope_name_appeared: T.Dict[str, T.Any] = {}
    for path in elements:
        tree_scope = scope_name_appeared
        for subscope in path.split(sep):
            if subscope not in tree_scope:
                tree_scope[subscope] = {}
            tree_scope = tree_scope[subscope]
    tree_scope = scope_name_appeared
    while len(tree_scope) == 1:
        subscope = list(scope_name_appeared.keys())[0]
        common_root += (sep if common_root else "") + subscope
        tree_scope = tree_scope[subscope]
        if shortest:
            break
    return common_root


def _replacement_to_relative_module_path(replacements: T.List[str]):
    return ".".join(
        [rep.split("[")[1][:-1] if "[" in rep else rep for rep in replacements]
    )


def _is_io_quantized_module(module):
    if isinstance(module, nn.Sequential):
        module = module[0]
    return not isinstance(module, torch.nn.quantized.Quantize) and any(
        _ in str(module.__class__)
        for _ in [
            "torch.nn.quantized",
            "torch.nn.intrinsic.quantized",
        ]
    )


def maybe_quantize_args_tensor(module, args):
    if _is_io_quantized_module(module):
        args = [
            # force cast in quantized form
            torch.quantize_per_tensor(
                in_item.float(),
                torch.tensor(0.1),
                torch.tensor(0),
                dtype=torch.quint8,
            )
            if isinstance(in_item, torch.Tensor)
            and not is_quantized_dtype(in_item.dtype)
            else in_item
            for in_item in args
        ]
    return args


@dataclass
class Data:
    name: str
    data: T.Any

    @property
    def export_name(self) -> str:
        return _refid_clean(self.name)

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
            else "q8(scale={self.quant['scale']}, zerop={self.quant['zero_point']})"
        )

    def cast_float_inplace(self):
        if self.data is not None:
            self.data = self.data.float()
            self.dtype = self.data.dtype

    @property
    def np_dtype(self) -> np.dtype:
        assert self.dtype is not None
        return TORCH_TO_NUMPY_DTYPE[self.dtype]

    @property
    def rank(self) -> T.Optional[int]:
        if self.data is not None:
            return len(self.data.shape)
        return len(self.shape) if self.shape else None

    @property
    def shaped(self) -> bool:
        return self.shape is not None

    @property
    def typed(self) -> bool:
        return bool(self.dtype)

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
            raise UnableToTraceData(self)

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
        if node_type.kind() == INTTYPE_kIND:
            dtype = torch.int32
        else:
            stype = node_type.scalarType()
            dtype = str_to_torch_dtype(stype) if stype else None
        return cls(
            name=node_c_value.debugName(),
            shape=[1]
            if node_type.kind() == INTTYPE_kIND
            else node_type.sizes(),
            dtype=dtype,
            data=node_c_value.toIValue(),
            quant=None,
        )

    def __hash__(self):
        return hash(self.name)


@dataclass
class PythonConstant(Data):
    data: T.Any

    @property
    def np_dtype(self) -> np.dtype:
        raise NotImplementedError()

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
        raise NotImplementedError()

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

    data: T.List[TensorVariable]

    @property
    def tracing_data(self) -> T.List[torch.Tensor]:
        return [d.tracing_data for d in self.data]

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        datas = ""
        for d in self.data:
            datas += f"\t\t\t{d},\n"
        return f"FixedTensorList(name='{self.name}', data=[\n{datas}\t\t])"


def dynamic_tensor_list_parse(node_c_value: torch._C.Value):
    """Hold outputs of aten::chunk and other pytorch graph Tensor[]"""

    node_type = node_c_value.type()
    assert node_type.kind() == LISTTYPE_KIND
    LOGGER.warning(
        "ListType can be of arbitrary length "
        "but we can not handle this dynamism at inference "
        " so 'split and other ops' will generate array "
        "of tensor with fixed size"
    )
    used_in = node_c_value.uses()
    if len(used_in) != 1:
        raise NotImplementedError()
    use_op = used_in[0].user
    if use_op.kind() != LISTUNPACK_KIND:
        raise NotImplementedError()

    return FixedTensorList(
        name=node_c_value.debugName(),
        data=[TensorVariable.parse(_) for _ in use_op.outputs()],
    )


@dataclass
class TorchConstant(Data):
    data: torch.Tensor

    @property
    def np_dtype(self) -> np.dtype:
        return TORCH_TO_NUMPY_DTYPE[self.data.dtype]

    @property
    def tracing_data(self):
        return self.data

    def __hash__(self):
        return hash(self.name)


def _find_data_node(data_nodes: T.List[Data], name: str):
    try:
        return next(d for d in data_nodes if d.name == name)
    except StopIteration as exp:
        names = [dnode.name for dnode in data_nodes]
        raise NotFoundDataNode(f"'{name}' not found in {names}") from exp


def _parse_getattr_tensor(node: torch._C.Node, module, data_nodes):
    data_state = _unfold_graph_getattr_by_node(module, node)[1].data
    data_nodes.append(
        TensorVariable(
            name=node.output().debugName(),
            shape=list(data_state.shape),
            dtype=data_state.dtype,
            data=data_state,
        )
    )


def _parse_getattr_script_obj(node: torch._C.Node, module, data_nodes):
    pack_name = node.output().debugName()
    pack = _unfold_graph_getattr_by_node(module, node)[1]
    assert isinstance(pack, torch._C.ScriptObject)
    data_nodes.append(
        BlobTorchScriptObject(
            name=pack_name,
            data=pack,
        )
    )


def _parse_constant(node: torch._C.Node, data_nodes) -> T.Optional[Data]:
    try:
        data = node["value"]
    except RuntimeError:
        data = None
    name = node.output().debugName()
    dtype = node.output().type().annotation_str
    if dtype == "bool":
        data = bool(data)
    elif dtype == "int":
        data = int(data)
    elif dtype == "float":
        data = float(data)
    elif dtype == "str":
        data = str(data)
    elif dtype == "NoneType":
        data = None
    elif dtype == "Tensor":
        assert isinstance(data, torch.Tensor)
        data_nodes.append(
            TensorVariable(
                name=name, data=data, shape=list(data.shape), dtype=data.dtype
            )
        )
        return data_nodes[-1]
    elif dtype == "Device":
        # Device will not be useful info for us but we pass it to avoid
        # dereferencing it from full graph
        pass
    else:
        raise NotImplementedError(dtype)
    data_nodes.append(PythonConstant(name=name, data=data))
    return data_nodes[-1]


def _fetch_backward(data_nodes, c_node: torch._C.Node):
    """backward search of final resolution argument from list_construct"""
    if c_node.kind() in [ATEN_INT, NUMTOTENSOR_KIND]:
        return _fetch_backward(data_nodes, c_node.input().node())
    try:
        return _find_data_node(data_nodes, c_node.output().debugName())
    except NotFoundDataNode as exp:
        raise NotImplementedError("_fetch_backward c_node:", c_node) from exp


def _parse_list_construct_values(node, data_nodes):
    values = []
    contains_tensors = False
    for cvalue in node.inputs():
        if cvalue.node().kind() == CONSTANT_KIND:
            value = _parse_constant(
                cvalue.node(), []  # data_nodes empty as added later
            )
            if isinstance(value, TensorVariable):
                contains_tensors = True
        else:
            contains_tensors = True
            if cvalue.node().kind() == ATEN_INT:
                value = _fetch_backward(data_nodes, cvalue.node())
            elif str(cvalue.type()) == "Tensor":
                try:
                    value = _find_data_node(data_nodes, cvalue.debugName())
                except NotFoundDataNode:
                    value = TensorVariable.parse(cvalue)
            else:
                raise NotImplementedError(
                    "parse list construct argument", cvalue
                )
        values.append(value)
    return contains_tensors, values


def _parse_list_construct(node, data_nodes):
    # should build a Data
    contains_tensors, values = _parse_list_construct_values(node, data_nodes)

    if contains_tensors:
        tensor_values = []
        for value in values:
            if isinstance(value, TensorVariable):
                if not any(_.name == value.name for _ in data_nodes):
                    data_nodes.append(value)
            elif isinstance(value, PythonConstant):
                pass
            else:
                raise NotImplementedError()
            tensor_values.append(value)
        data_node = FixedTensorList(
            name=node.output().debugName(), data=tensor_values
        )
    else:
        data_node = PythonConstant(
            name=node.output().debugName(), data=[v.data for v in values]
        )
    data_nodes.append(data_node)

    # }


def _prepare_arguments(kind: str, inputs: T.List[torch._C.Value], data_nodes):
    abstracted_inputs = [
        _find_data_node(data_nodes, inp.debugName())
        for inp in inputs
        if not isinstance(inp, Data)
    ]
    if kind in [
        "aten::sub",
        "aten::sub_",
        "aten::add",
        "aten::add_",
    ]:
        # remove useless ref to scaling (probably never used)
        abstracted_inputs = abstracted_inputs[:2]

    if kind in ["aten::mean", "aten::sum"]:
        abstracted_inputs = abstracted_inputs[:3]

    if kind == "aten::elu":
        # difference between aten and python API
        abstracted_inputs = abstracted_inputs[:2]

    if kind == "aten::clone":
        # remove useless ref to memory_format (for us)
        abstracted_inputs = abstracted_inputs[:1]

    if kind == "aten::expand":
        # remove useless ref to inplicit (for us)
        abstracted_inputs = abstracted_inputs[:-1]

    if kind == "aten::ones":
        # remove useless ref even dtype for now
        abstracted_inputs = abstracted_inputs[:1]

    if kind == ATEN_CONTIGUOUS_KIND:
        abstracted_inputs = abstracted_inputs[:1]

    if kind == ATEN_ARANGE:
        # we guess from observation that provided arange params are either:
        # [start, end, step, dtype, layout_type, device, requires_grad]
        # [start, end, dtype, layout_type, device, requires_grad]
        # [end, dtype, layout_type, device, requires_grad]

        abstracted_inputs = abstracted_inputs[:-4]  # skip even dtype

        # cast to torch dtype
        # abstracted_inputs[-1].data = SCALAR_TYPE_TO_PYTORCH_TYPE[
        # abstracted_inputs[-1].data
        # ]

        n_inputs = len(abstracted_inputs)
        if n_inputs < 1 or n_inputs > 3:
            raise NotImplementedError(n_inputs, abstracted_inputs)
        if n_inputs == 1:
            abstracted_inputs.insert(
                0,
                PythonConstant(
                    abstracted_inputs[0].name + "_stub_start", data=0
                ),
            )
        abstracted_inputs.insert(
            2,
            PythonConstant(abstracted_inputs[0].name + "_stub_step", data=1),
        )

    return abstracted_inputs


def _aten_inputs_and_op_ref(kind, inputs, data_nodes):
    abstracted_inputs = _prepare_arguments(kind, inputs, data_nodes)
    op_ref = None
    try:
        op_ref = aten_name_to_torch_fn(kind)
    except AttributeError:
        pass
    return op_ref, abstracted_inputs


def _rerouted_parsing(node: torch._C.Node, data_nodes: T.List[Data], module):
    """Specific torch kind operation are transformed

    to improve readability of intermediate representation

        If specific kind matched it raise TorchOpTranslatedDifferently
        meaning it is handled differently than vanilla torch graph

    """
    kind: str = node.kind()
    if kind == GETATTR_KIND:
        _parse_getattr_tensor(node, module, data_nodes)
        raise TorchOpTranslatedDifferently("geattr handled as TensorVariable")
    if kind == CONSTANT_KIND:
        _parse_constant(node, data_nodes)
        raise TorchOpTranslatedDifferently("constant handled as PythonConstant")
    if kind == LISTCONSTRUCT_KIND:
        _parse_list_construct(node, data_nodes)
        raise TorchOpTranslatedDifferently(
            "List Construct handled as PythonConstant"
        )
    if kind.startswith(PRIM_STARTID):
        if kind == TUPLEUNPACK_KIND:
            dnodes = _find_data_node(data_nodes, node.input().debugName()).data
            for dnode, o_node_c_value in zip(dnodes, node.outputs()):
                o_type = o_node_c_value.type()
                stype = o_type.scalarType()
                dtype = str_to_torch_dtype(stype) if stype else None
                dnode.name = o_node_c_value.debugName()
                dnode.shape = o_type.sizes()
                dnode.dtype = dtype
                dnode.data = o_node_c_value.toIValue()
            raise TorchOpTranslatedDifferently("Tuple unpacked")
        if kind == TUPLECONSTRUCT_KIND:
            data_nodes.append(
                TupleTensors(
                    name=node.output().debugName(),
                    data=[
                        _find_data_node(data_nodes, i_node_c_value.debugName())
                        for i_node_c_value in node.inputs()
                    ],
                )
            )
            raise TorchOpTranslatedDifferently("Tuple Construct")
        if kind == NUMTOTENSOR_KIND:
            return
        if kind == LISTUNPACK_KIND:
            # note: maybe should be replace dataNode to a FixedTensorList
            raise TorchOpTranslatedDifferently("List unpacked")
        if kind != CALL_KIND:
            raise NotImplementedError(node)
    if kind == CALL_KIND and not any(
        bool(use) for onode in node.outputs() for use in onode.uses()
    ):
        raise TorchOpTranslatedDifferently(
            "This method outputs are not used anywhere in graph"
        )


def _extract_op_infos(
    module,
    data_nodes: T.List[Data],
    node: torch._C.Node,
    module_traced: torch.jit.TracedModule,
) -> T.Tuple[
    str, T.Optional[str], str, T.Callable[[T.Any], T.Any], T.List[Data]
]:
    """Extract informations from module or torch operation"""
    call_name = None
    kind: str = node.kind()
    inputs = list(node.inputs())

    if kind == CALL_KIND:
        value_call_ref = inputs[0]
        module_getter_ref, op_ref = _unfold_graph_getattr_by_node(
            module, value_call_ref.node()
        )
        call_name = value_call_ref.debugName()
        inputs = inputs[1:]
        # use appropriate graph
        _, op_ref_traced = _unfold_graph_getattr_by_node(
            module_traced, value_call_ref.node()
        )
        op_ref = TracedModuleCallBox(
            op_ref, op_ref_traced, fn_name=node.s("name")
        )

    elif kind.startswith("quantized::"):
        module_getter_ref = MODULE_PATH_QUANTIZED
        op_ref = quantized_name_to_torch_fn(kind)
        for inp in inputs:
            in_name = inp.debugName()
            try:
                _find_data_node(data_nodes, in_name)
            except NotFoundDataNode:
                _parse_getattr_script_obj(inp.node(), module, data_nodes)
    else:
        module_getter_ref = MODULE_PATH_ATEN
        if kind in MAP_TO_NOP:
            op_ref = nop  # type: ignore
        elif kind.startswith(ATEN_STARTID):
            op_ref, inputs = _aten_inputs_and_op_ref(kind, inputs, data_nodes)
        else:
            raise NotImplementedError(
                f"Unable to extract operation from {kind}"
            )

    abstracted_inputs: T.List[Data] = [
        inp
        if isinstance(inp, Data)
        else _find_data_node(data_nodes, inp.debugName())
        for inp in inputs
    ]

    return (kind, call_name, module_getter_ref, op_ref, abstracted_inputs)


class TracedModuleCallBox:
    """Evaluate Optimized traced Function code so that signature always match

    original Module is passed to do proper un-boxing later on.
    This is needed because we have a re-routing based on actual module classtype.

    """

    def __init__(
        self,
        module: nn.Module,
        traced_module: torch.jit.TracedModule,
        fn_name: str,
    ):
        self.mod = module
        self.traced_module = traced_module
        self.fn_name = fn_name

    def __call__(self, *args, **kwargs):
        # _actual_script_module is an implementation details
        # from torch/jit/_trace.py:l1076 in TracedModule
        if self.fn_name == "forward":
            traced_op_call = self.traced_module._actual_script_module.forward
        else:
            traced_op_call = getattr(self.traced_module, self.fn_name)
        return traced_op_call(*args, **kwargs)


@dataclass
class TorchOp:
    kind: str
    module_path: str
    inputs: T.List[Data]
    outputs: T.List[TtupleOrVar]
    scope: str
    op_ref: T.Optional[T.Callable]  # multiple ins and outs possible
    call_name: T.Optional[str]

    def __hash__(self):
        return hash(f"{self.kind}{self.inputs}{self.outputs}")

    @property
    def is_callmethod(self) -> bool:
        return self.kind == CALL_KIND

    @classmethod
    def _parse_outputs(cls, node: torch._C.Node, data_nodes: T.List[Data]):
        outputs: T.List[TtupleOrVar] = []
        for out_node in node.outputs():  #: torch._C.Value
            if out_node.type().annotation_str != NONETYPE_KIND:
                if out_node.type().kind() == LISTTYPE_KIND:
                    fixed_tensor_list = dynamic_tensor_list_parse(out_node)
                    data_nodes += fixed_tensor_list.data
                    data_nodes.append(fixed_tensor_list)
                    outputs += fixed_tensor_list.data
                elif out_node.type().kind() == TUPLETYPE_KIND:
                    tuple_out = TupleTensors.parse_from_tuple_type(out_node)
                    for tupitem in tuple_out.data:
                        data_nodes.append(tupitem)
                    # ducktyping/factorize tensor_out & tuple_out
                    # lead to mypy complain hence repeated
                    outputs.append(tuple_out)
                    data_nodes.append(tuple_out)
                else:
                    tensor_out = TensorVariable.parse(out_node)
                    outputs.append(tensor_out)
                    data_nodes.append(tensor_out)
        return outputs

    @classmethod
    def parse(
        cls,
        module,
        node: torch._C.Node,
        scope: str,
        data_nodes: T.List[Data],
        module_traced,
    ) -> "TorchOp":
        op_ref = None
        _rerouted_parsing(node, data_nodes, module)

        (
            kind,
            call_name,
            module_getter_ref,
            op_ref,
            inputs,
        ) = _extract_op_infos(module, data_nodes, node, module_traced)
        outputs = cls._parse_outputs(node, data_nodes)

        if not outputs:
            raise TorchOpTranslatedDifferently(
                "Avoid reccording no return operations"
            )

        return cls(
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            module_path=module_getter_ref,
            op_ref=op_ref,
            call_name=call_name,
        )

    def call_op(self):
        """Produce operation output based on traced inputs with real torch call

        This operation call is done via self._args arguments (for now).
        Which means that we need to have all args needed in parameters order,
        following at least 1 underling torch operation signature.

        NOTE: we use a different approach than original torch.onnx which pass
        parameter by keyword arguments, this is due to the fact that we are not
        aware of argument name being provided in exported graph (
            from what we understand torch.onnx solve this via explicit
            rerouting of all signatures, which might be a bit bulky in most case
        ).

        """
        if self.op_ref is not None:
            if self.kind in MAP_TO_TENSOR_FN:
                args = self._args
                tensor = args[0]
                subargs = args[1:]
                if self.kind == ATEN_VIEW_KIND and None in subargs[0]:
                    # custom reconstruction of missing dimensions infos
                    subargs = list(subargs)
                    subargs[0] = _reconstruct_view_dims(
                        tensor.shape, subargs[0]
                    )
                    self.inputs[1].data = subargs[0]
                    subargs = tuple(subargs)
                return getattr(tensor, self.kind.replace(ATEN_STARTID, ""))(
                    *subargs
                )
            args = self._args
            # hacky/bad way to pass argument that are named argument only {
            kwargs = {}
            if self.kind == "aten::div" and len(args) >= 3:
                kwargs["rounding_mode"] = args[2]
                args = args[:-1]
                self.op_ref = torch.div
            # }
            return self.op_ref(*args, **kwargs)
        raise NotImplementedError(self)

    @property
    def _args(self) -> T.Tuple[T.Any, ...]:
        return tuple(_.tracing_data for _ in self.inputs)

    def realise_output_type_and_size(self) -> bool:
        """This trace output and try to find out type shape and even constant realisation"""
        if not all(_.tracable for _ in self.inputs):
            return False

        # generate all data and call ops to infer missing infos
        results = self.call_op()

        if isinstance(results, int):
            results = torch.tensor(results, dtype=torch.int32)

        if isinstance(results, torch.Tensor):
            results = (results,)

        if len(self.outputs) != len(results):
            raise CheckError(
                f"Arity Missmatch between extracted from graph len({len(self.outputs)}) "
                f"and the one experienced in tracing simulation len({len(results)}) "
                f"for {self.op_ref}"
            )
        is_constant_inputs = all(
            input_node.data is not None for input_node in self.inputs
        )
        for data_node, result in zip(self.outputs, results):
            if is_constant_inputs:
                data_node.data = result
            if isinstance(data_node, TensorVariable):
                if self.kind == ATEN_SIZE_KIND:
                    # note this is a special case where we fix variable value
                    data_node.data = result
                data_node.dtype = result.dtype
                data_node.shape = list(result.shape)
                if is_quantized_dtype(result.dtype):
                    data_node.quant = {
                        "scale": result.q_scale(),
                        "zero_point": result.q_zero_point(),
                    }
        return True

    def __repr__(self):
        body = f"\tkind={self.kind}\n"
        inputs = ""
        for input_ in self.inputs:
            inputs += f"\t\t{input_},\n"
        body += f"\tinputs=(\n{inputs}\n\t)\n"

        outputs = ""
        for output in self.outputs:
            outputs += f"\t\t{output},\n"
        body += f"\toutputs=(\n{outputs}\t)\n"
        return f"TorchOp(\n{body}\n)".replace("\t", " " * 2)


class _OrderedStrictSet(list):
    """Data Structure aimed to detect code implementation bugs

    Indeed it checks that no 2 nodes are inserted with same name

    Warning! only aimed at Data items (but work if provided item as name attr).

    """

    def append(self, item):
        assert all(elm.name != item.name for elm in self), item.name
        return super().append(item)


class TorchModuleIRGraph:

    """Torch Graph intermediate representation from: jit.trace with recursion

    This is not direct torch._C.Graph but simpler abstraction, with:

    A list of data nodes in `self.data_nodes`
    A list of operations nodes in `self.op_nodes`

    `self.inputs` is a list of reference of some `self.data_nodes`
    `self.outputs` is a list of reference of some `self.data_nodes`

    This abstraction of the vanilla Torch Graph allow to manipulate graph
    in order to check/complete missing data informations and ignore
    useless operations for our transcription needs.

    It's also allows to be less reliant on base graph in case
    of modification of Pytorch Internals (think Adapter Pattern).

    Warning !
        Only NOT nested data container (TupleTensors, FixedTensorList, ...)
        are supported for now

    """

    SEP = "/"

    def __init__(
        self,
        module: nn.Module,
        args: T.Tuple[T.Any, ...],  # likely mostly torch tensor
        omit_useless_nodes: bool = True,
        auto_parse: bool = True,
        inputs: T.Optional[T.List[Data]] = None,
        outputs: T.Optional[T.List[TtupleOrVar]] = None,
        renaming_scheme: str = "numeric",
        module_traced: T.Optional[torch.jit.TracedModule] = None,
        fn_trace_call_name: T.Optional[str] = None,
    ):
        self.op_nodes: T.List[TorchOp] = []
        self.inputs: T.List[Data] = []
        self.outputs: T.List[TtupleOrVar] = []
        self._data_nodes: _OrderedStrictSet = _OrderedStrictSet()
        self._module = module
        self._module_traced = module_traced
        self._fn_trace_call_name = fn_trace_call_name
        self._args = maybe_quantize_args_tensor(module, args)
        self._omit_useless_nodes = omit_useless_nodes
        if auto_parse:
            self.parse(
                provided_inputs=inputs,
                provided_outputs=outputs,
                renaming_scheme=renaming_scheme,
            )

    @property
    def data_nodes(self):
        return self._data_nodes

    @data_nodes.setter
    def data_nodes(self, other):
        oss = _OrderedStrictSet()
        for item in other:
            oss.append(item)
        self._data_nodes = oss

    def _check_container_items_rely_on_data_nodes(self):
        """container items reference must exists in `data_nodes`"""
        for dnode in self.data_nodes:
            if _is_container(dnode):
                for subdnode in dnode.data:
                    assert any(
                        subdnode is _ for _ in self.data_nodes
                    ), f"not referenced correctly sub item: {subdnode}"

    def _check_io_rely_on_data_nodes(self):
        """`inputs` or `outputs` reference items must exists in `data_nodes`"""
        for inode in self.inputs:
            if not any(_ is inode for _ in self.data_nodes):
                raise CheckError(f"not referenced correctly input: {inode}")

        for onode in self.outputs:
            if not any(_ is onode for _ in self.data_nodes):
                raise CheckError(f"not referenced correctly output: {onode}")

    @property  # type: ignore
    @cache
    def _torch_trace(self):
        try:
            if self._module_traced is not None:
                return self._module_traced
            return jit.trace(
                self._module,
                self._args,
                check_trace=True,
            )
        except RuntimeError as exp:
            raise JitTraceFailed(
                "Unable to trace with jit one of following submodule:"
                f"{[(k, v.__class__) for k,v in self._module.named_children()]} "
                f"with original error:\n\n'{exp}'\n\n"
                "This maybe due to provided input dimension. "
                "If not, you can aleviate this issue by applying a special hook"
                "this module (explaination available in torch_to_nnef README)"
            ) from exp

    @property  # type: ignore
    @cache
    def _original_torch_graph(self):
        trace = self._torch_trace
        if self._fn_trace_call_name and self._fn_trace_call_name != "forward":
            trace = getattr(trace, self._fn_trace_call_name)
        return trace.graph

    def remap_node(self, from_node, to_node):
        assert isinstance(from_node, Data)
        assert isinstance(to_node, Data)
        from_node.name = to_node.name
        for op in self.op_nodes:
            op.inputs = [to_node if _ is from_node else _ for _ in op.inputs]
            op.outputs = [to_node if _ is from_node else _ for _ in op.outputs]
        self.data_nodes = [_ for _ in self.data_nodes if _ is not from_node]

        # allow to remap item in containers as well
        for dnode in self.data_nodes:
            if _is_container(dnode):
                new_data = []
                for subdnode in dnode.data:
                    if subdnode is from_node:
                        value = to_node
                    else:
                        value = subdnode
                    new_data.append(value)
                dnode.data = new_data

        # add if not exists in graph
        if not any(to_node is _ for _ in self.data_nodes):
            self.data_nodes.append(to_node)

    def _parse_inputs(
        self, provided_inputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph inputs"""
        idx = 0
        for node_c_value in self._original_torch_graph.inputs():
            if self._omit_useless_nodes:
                if (
                    len(node_c_value.uses()) == 0
                ):  # number of user of the node_c_value (= number of outputs/ fanout)
                    continue

            if node_c_value.type().kind() != CLASSTYPE_KIND:
                tv = TensorVariable.parse(node_c_value)
                if provided_inputs is not None:
                    original_input = provided_inputs[idx]
                    tv.shape = original_input.shape
                    tv.dtype = original_input.dtype
                    tv.quant = original_input.quant
                    idx += 1
                self.inputs.append(tv)
                self.data_nodes.append(tv)

    def _parse_core(self):
        """Parse all Operations and collect the scope infos"""
        attr_to_scope: T.Dict[T.Any, str] = {}
        for node in self._original_torch_graph.nodes():
            if node.kind() == GETATTR_KIND:
                attr_name = node.s("name")
                parent = node.input().node()
                if (
                    parent.kind() == GETATTR_KIND
                ):  # If the parent node is not the top-level "self" node
                    parent_attr_name = parent.s("name")
                    parent_scope = attr_to_scope[parent_attr_name]
                    attr_scope = parent_scope.split("/")[-1]
                    attr_to_scope[
                        attr_name
                    ] = f"{parent_scope}/{attr_scope}.{attr_name}"
                else:
                    attr_to_scope[attr_name] = f"__module.{attr_name}"
                # We don't need classtype nodes; scope will provide this information
                if node.output().type().kind() != CLASSTYPE_KIND:
                    try:
                        op = TorchOp.parse(
                            self._module,
                            node,
                            scope=attr_to_scope[attr_name],
                            data_nodes=self.data_nodes,
                            module_traced=self._torch_trace,
                        )
                        self.op_nodes.append(op)
                    except TorchOpTranslatedDifferently:
                        pass

            else:
                try:
                    op = TorchOp.parse(
                        self._module,
                        node,
                        scope="",
                        data_nodes=self.data_nodes,
                        module_traced=self._torch_trace,
                    )
                    self.op_nodes.append(op)
                except TorchOpTranslatedDifferently:
                    pass

    def _parse_outputs(
        self, provided_outputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph outputs"""
        torch_graph_outputs = self._original_torch_graph.outputs()
        outputs = [
            _find_data_node(self.data_nodes, _.debugName())
            for _ in torch_graph_outputs
        ]

        if provided_outputs is not None:
            expanded_output = list(_expand_containers_if_exists(outputs))
            original_outputs = list(
                _expand_containers_if_exists(provided_outputs)
            )
            if len(expanded_output) != len(original_outputs):
                CheckError(f"{len(expanded_output)} == {len(original_outputs)}")
            for original_output, output in zip(
                original_outputs, expanded_output
            ):
                if _is_container(original_output) and _is_container(output):
                    # can be safely explored
                    continue
                if isinstance(output, TensorVariable):
                    output.shape = original_output.shape
                    output.dtype = original_output.dtype
                    output.quant = original_output.quant
                else:
                    raise NotImplementedError(
                        f"output={output}\ncompared to:\n"
                        f"original_output={original_output}"
                    )
        self.outputs = outputs

    def _update_scope_reference(self):
        """Update scope in op_nodes with additional infos"""
        alias_to_name = {}
        base_name = _parse_traced_name(self._torch_trace)
        for name, module in self._torch_trace.named_modules(prefix="__module"):
            mod_name = _parse_traced_name(module)
            attr_name = name.split(".")[-1]
            alias_to_name[name] = f"{mod_name}[{attr_name}]"

        for node in self.op_nodes:
            module_aliases = node.scope.split(self.SEP)
            replacements = [
                alias_to_name[alias]
                if alias in alias_to_name
                else alias.split(".")[-1]
                for alias in module_aliases
            ]
            node.scope = base_name
            if any(replacements):
                node.module_path = _replacement_to_relative_module_path(
                    replacements
                )
                node.scope += self.SEP + self.SEP.join(replacements)

    def _update_data_node_name_with_base_context(self):
        unique_name_to_scoped_name = {}
        selected_scope_name = _find_common_root(
            elements=(node.scope for node in self.op_nodes),
            sep=self.SEP,
            shortest=True,
        )

        for node in self.op_nodes:
            for input_node in _expand_containers_if_exists(node.inputs):
                unique_name_to_scoped_name[input_node.name] = (
                    node.scope + self.SEP + input_node.name
                )

        for node in _expand_containers_if_exists(self.data_nodes):
            if not node.name.startswith(selected_scope_name + self.SEP):
                node.name = selected_scope_name + self.SEP + node.name

        for node in _expand_containers_if_exists(self.inputs):
            if not node.name.startswith(selected_scope_name + self.SEP):
                node.name = selected_scope_name + self.SEP + node.name

    def _infer_missing_shapes_from_ops_outputs(self):
        unshaped_data = {}
        for node in self.data_nodes:
            if not node.tracable:
                unshaped_data[node.name] = node

        remaining_ops = self.op_nodes
        while unshaped_data:
            ops_to_rm = []
            start_len = len(unshaped_data)
            for op_node in remaining_ops:
                worked = op_node.realise_output_type_and_size()
                if worked:
                    ops_to_rm.append(op_node)
                    for _ in op_node.outputs:
                        if _.name in unshaped_data:
                            del unshaped_data[_.name]
            remaining_ops = [op for op in remaining_ops if op not in ops_to_rm]
            end_len = len(unshaped_data)
            if start_len == end_len:
                raise NotImplementedError(
                    f"missing unshaped_data: {unshaped_data}"
                )

    def _merge_subraph(
        self, submodule_graph, callmethod_node, prefix: str, module_prefix: str
    ):
        # Re-Wire input and output naming => {

        def search_and_replace_data_nodes(
            node_subgraph_to_wire: T.List[Data],
            node_graph_to_wire: T.List[Data],
            datas_attr: str,
        ):
            for node, ref_node in zip(
                _expand_containers_if_exists(node_subgraph_to_wire),
                _expand_containers_if_exists(node_graph_to_wire),
            ):
                for op_node in submodule_graph.op_nodes:
                    datas = []
                    for dnode in getattr(op_node, datas_attr):
                        ref = dnode
                        if dnode.name == node.name:
                            ref = ref_node
                        datas.append(ref)
                    setattr(op_node, datas_attr, datas)

                for data_node in submodule_graph.data_nodes:
                    if _is_container(data_node):  # update ref
                        data_node.data = [
                            ref_node if dnode.name == node.name else dnode
                            for dnode in data_node.data
                        ]

        search_and_replace_data_nodes(
            submodule_graph.inputs, callmethod_node.inputs, "inputs"
        )
        search_and_replace_data_nodes(
            submodule_graph.outputs, callmethod_node.outputs, "outputs"
        )

        # }

        to_del_nodes = submodule_graph.inputs + submodule_graph.outputs
        submodule_graph.data_nodes = [
            _ for _ in submodule_graph.data_nodes if _ not in to_del_nodes
        ]
        for _ in submodule_graph.op_nodes:
            res = _.scope.split(self.SEP, maxsplit=1)
            if len(res) >= 2 and isinstance(res, list):
                _.scope = f"{res[0]}[{prefix}]{self.SEP}{res[1]}"
            else:
                _.scope = f"{res}[{prefix}]"
            _.module_path = f"{module_prefix}.{_.module_path}"

        for _ in submodule_graph.data_nodes:
            _.name = f"{prefix}.{_.name}"

        self.op_nodes = [op for op in self.op_nodes if op != callmethod_node]
        self.op_nodes += submodule_graph.op_nodes
        self.data_nodes += submodule_graph.data_nodes

    def _recursive_call_method(self, renaming_scheme: str):
        """In case prim::CallMethod is encountered it tries to trace it

        It does this by recursive call to parse_module on linked submodule.

        Some part of the submodule may not be serializable to JIT
        this is for this very API limitation that we do not use directly
        the method torch.jit._get_trace_graph that is used in
        ONNX builtin pytorch serialization and instead build on recursive jit.parse.

        If the serialization to jit FAIL you will be able to put a full hook
        on the concerned sub-module with declarative enonciation of what
        is needed.

        """
        ref_count: T.Dict[str, int] = defaultdict(int)
        for op in self.op_nodes:
            if op.is_callmethod:
                cname = op.call_name or ""
                ref_count[cname] += 1
                assert isinstance(op, TorchOp)
                module_traced = None
                fn_trace_call_name = None
                if isinstance(op.op_ref, TracedModuleCallBox):
                    module_traced = op.op_ref.traced_module
                    fn_trace_call_name = op.op_ref.fn_name
                    op.op_ref = op.op_ref.mod

                assert isinstance(op.op_ref, nn.Module)
                submodule_graph = TorchModuleIRGraph(
                    op.op_ref,
                    op._args,
                    omit_useless_nodes=self._omit_useless_nodes,
                    inputs=op.inputs,
                    outputs=op.outputs,
                    renaming_scheme=renaming_scheme,
                    module_traced=module_traced,
                    fn_trace_call_name=fn_trace_call_name,
                )
                prefix = ""
                if cname is not None:
                    prefix = _add_prefix_if_start_with_digit(cname, "s")
                prefix += f"_c{ref_count[cname]}"
                self._merge_subraph(
                    submodule_graph,
                    prefix=prefix,
                    module_prefix=op.module_path,
                    callmethod_node=op,
                )

    def apply_renaming_scheme(self, scheme="natural_verbose"):
        """Rename availlable data node following a scheme

        by default the natural_verbose pattern built is as close as possible
        to Pytorch graph context info. This pattern might come as too verbose.

        we propose a more concise numeric pattern that allow easier debug
        when looking at NNEF export correctness.

        """
        if scheme == "natural_verbose":
            return
        if scheme == "numeric":
            count_ref = defaultdict(int)
            mapping = {}
            prefix_map = {
                TensorVariable: "v",
                PythonConstant: "c",
                BlobTorchScriptObject: "b",
                TorchConstant: "t",
                FixedTensorList: "l",
                TupleTensors: "tt",
                Data: "d",  # not used, avoid static analysis complain
            }
            for dnode in self.data_nodes:
                prefix = prefix_map[dnode.__class__]
                if dnode.name in mapping:
                    dnode.name = mapping[dnode.name]
                    continue
                suffix = count_ref[prefix]
                count_ref[prefix] += 1
                mapping[dnode.name] = prefix + str(suffix)
                dnode.name = mapping[dnode.name]
            return

        raise NotImplementedError(f"renaming scheme: {scheme}")

    def _filter_tuple_tensor_from_data_nodes(self):
        new_data_nodes = []
        for dnode in self.data_nodes:
            if isinstance(dnode, TupleTensors):
                continue
            new_data_nodes.append(dnode)
        self.data_nodes = new_data_nodes

    def _expand_tuple_in(self, iterable):
        expanded_data_nodes = []
        for dnode in iterable:
            if isinstance(dnode, TupleTensors):
                for sdnode in dnode.data:
                    expanded_data_nodes.append(sdnode)
            else:
                expanded_data_nodes.append(dnode)
        return expanded_data_nodes

    def _avoid_reference_to_tuples(self):
        """Remove all references to tuple by using only unpacked variables"""
        self._filter_tuple_tensor_from_data_nodes()
        self.inputs = self._expand_tuple_in(self.inputs)
        self.outputs = self._expand_tuple_in(self.outputs)
        for op in self.op_nodes:
            op.inputs = self._expand_tuple_in(op.inputs)
            op.outputs = self._expand_tuple_in(op.outputs)

    def _filter_nodes_not_in_trace_between_inputs_and_outputs(self):
        """remove all unused graph nodes

        Backward propagation from graph output to input to select kept nodes

        """
        used_data_nodes = set(self.outputs)
        used_op_nodes = set()
        remaining_op_nodes = set(self.op_nodes)
        remaining_data_nodes = set(self.data_nodes).difference(used_data_nodes)

        new_frontier = True
        while new_frontier:
            new_frontier = False
            for op in remaining_op_nodes:
                for data_node in used_data_nodes:
                    if data_node in op.outputs:
                        used_op_nodes.add(op)
                        used_data_nodes.update(op.inputs)
                        new_frontier = True
                        break  # look at next op

            remaining_data_nodes.difference_update(used_data_nodes)
            remaining_op_nodes.difference_update(used_op_nodes)

            if len(remaining_data_nodes):
                # at each loop try to add new expansion of
                # maybe added FixedTensorList, TupleTensors
                additional_data_node_from_list = set()
                for used_data_node in used_data_nodes:
                    if _is_container(used_data_node):
                        additional_data_node_from_list.update(
                            used_data_node.data
                        )
                remaining_data_nodes.difference_update(
                    additional_data_node_from_list
                )
                used_data_nodes.update(additional_data_node_from_list)

        # filtered bug with original order
        ordered_op_nodes_hashs = [hash(_) for _ in self.op_nodes]
        self.op_nodes = sorted(
            list(used_op_nodes),
            key=lambda _: ordered_op_nodes_hashs.index(hash(_)),
        )

        ordered_data_nodes_hashs = [hash(_) for _ in self.data_nodes]
        self.data_nodes = sorted(
            list(used_data_nodes),
            key=lambda _: ordered_data_nodes_hashs.index(hash(_))
            if _ in ordered_data_nodes_hashs
            else -1,
        )

    def parse(
        self,
        renaming_scheme: str = "numeric",
        provided_inputs=None,
        provided_outputs=None,
    ):
        try:
            extractor = ModuleInfoExtractor.get_by_module(self._module)
            extractor.generate_in_torch_graph(
                self, provided_inputs, provided_outputs
            )
            return self
        except NotFoundModuleExtractor:
            pass

        self._parse_inputs(provided_inputs)
        self._parse_core()
        self._parse_outputs(provided_outputs)
        self._update_scope_reference()
        self._update_data_node_name_with_base_context()
        self._infer_missing_shapes_from_ops_outputs()
        self._recursive_call_method(renaming_scheme=renaming_scheme)
        self._avoid_reference_to_tuples()
        self._filter_nodes_not_in_trace_between_inputs_and_outputs()

        self._check_container_items_rely_on_data_nodes()
        self._check_io_rely_on_data_nodes()

        if renaming_scheme:
            self.apply_renaming_scheme(renaming_scheme)

        return self

    def find_data_node_producer(self, data_node: Data):
        for op in self.op_nodes:
            for op_out_dnode in _expand_containers_if_exists(op.outputs):
                if op_out_dnode is data_node:
                    return op
        raise NotFoundTorchOp("Did not find operation node")

    def printall(self):
        """Display Helper Graph infos in stdout of your tty"""
        console = Console(
            theme={
                "type": "blue",
                "var": "grey82",
                "kind": "yellow",
                "subsection": "red",  # dim bold
            }
        )
        cprint = console.print
        cprint(
            "\n\n[type]"
            + "_" * 35
            + "[Pytorch JIT Graph]"
            + "_" * 35
            + "[/type]"
        )
        inputs_str = ", ".join(_.slug for _ in self.inputs)
        cprint(f"inputs: ({inputs_str})")
        cprint("")
        cprint("\t[subsection]Static Constants:[/subsection]")
        for _ in self.data_nodes:
            if isinstance(_, PythonConstant):
                cprint(
                    f"\t\t[type]{type(_.data).__name__}[/type] "
                    f"[var]{_.export_name}[/var] := {_.data}"
                )

        cprint()
        cprint("\t[subsection]Static Tensor:[/subsection]")
        for _ in self.data_nodes:
            if isinstance(_, TensorVariable) and _.data is not None:
                cprint(
                    f"\t\t[type]{_.dtype}[/type] "
                    f"[var]{_.export_name}[/var] := shape({_.shape})"
                )

        cprint()
        cprint("\t[subsection]Blob TorchScript:[/subsection]")
        for _ in self.data_nodes:
            if isinstance(_, BlobTorchScriptObject):
                cprint(
                    f"\t\t[type]{type(_.data).__name__}[/type] "
                    f"[var]{_.export_name}[/var] := ?"
                )
        cprint("")
        cprint("\t[subsection]List:[/subsection]")
        for _ in self.data_nodes:
            if isinstance(_, FixedTensorList):
                refs = ", ".join([d.export_name for d in _.data])
                cprint(
                    f"\t\t[type]List[/type] "
                    f"[var]{_.export_name}[/var] := ({refs})"
                )

        cprint("")
        cprint("\t[subsection]TupleTensors:[/subsection]")
        for _ in self.data_nodes:
            if isinstance(_, TupleTensors):
                refs = ", ".join([d.export_name for d in _.data])
                cprint(
                    f"\t\t[type]TupleTensors[/type] "
                    f"[var]{_.export_name}[/var] := ({refs})"
                )

        cprint("")
        cprint("\t[subsection]Directed Acyclic Graph:[/subsection]")
        for _ in self.op_nodes:
            inputs_str = ""
            if _.inputs:
                inputs_str = ", ".join(
                    f"[var]{i.export_name}[/var]" for i in _.inputs
                )
                inputs_str = f"( {inputs_str} )"
            cls_name = ""
            outputs_str = ", ".join(
                [
                    f"[type]{o.dtype if hasattr(o, 'dtype') else type(o.data)}[/type] [var]{o.export_name}[/var]"
                    for o in _.outputs
                ]
            )
            cprint(
                "\t\t "
                f"{outputs_str} := [kind]{_.kind}[/kind]{inputs_str}{cls_name}"
            )

        outputs_str = ", ".join(_.slug for _ in self.outputs)
        cprint("")
        cprint(f"outputs: ({outputs_str})")
        cprint("[type]" + "_" * 100 + "[/type]")
