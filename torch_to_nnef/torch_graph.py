# pylint: disable=too-many-lines
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
from torch_to_nnef.utils import cache


class JitTraceFailed(RuntimeError):
    pass


class UnableToTraceData(ValueError):
    pass


class TorchOpTranslatedDifferently(ValueError):
    pass


class NotFoundDataNode(ValueError):
    pass


UNKNOWN_TRACE_SHAPE_VALUE = 321
CALL_KIND = "prim::CallMethod"
CONSTANT_KIND = "prim::Constant"
GETATTR_KIND = "prim::GetAttr"
LISTCONSTRUCT_KIND = "prim::ListConstruct"
PARAM_KIND = "prim::Param"
TUPLECONSTRUCT_KIND = "prim::TupleConstruct"
TUPLEUNPACK_KIND = "prim::TupleUnpack"
NUMTOTENSOR_KIND = "prim::NumToTensor"

CLASSTYPE_KIND = "ClassType"
TUPLETYPE_KIND = "TupleType"
LISTTYPE_KIND = "ListType"
NONETYPE_KIND = "NoneType"
INTTYPE_kIND = "IntType"

MODULE_PATH_ATEN = "TORCH_INTERNAL_ATEN"
MODULE_PATH_QUANTIZED = "TORCH_INTERNAL_QUANTIZED"
SPECIAL_ATEN_REMAP_PYTORCH = {"__and__": "bitwise_and", "__or__": "bitwise_or"}


def aten_name_to_torch_fn(aten_name):
    name = aten_name.replace("aten::", "")
    return getattr(torch.ops.aten, name)


def quantized_name_to_torch_fn(aten_name):
    name = aten_name.replace("quantized::", "")
    return getattr(torch.ops.quantized, name)


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


def _expand_tuples_if_exists(data_items):
    for data_item in data_items:
        if isinstance(data_item, TupleTensors):
            yield from data_item.data
        yield data_item


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


def _is_io_qantized_module(module):
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
    if _is_io_qantized_module(module):
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

    @property
    def np_dtype(self) -> np.dtype:
        assert self.dtype is not None
        return TORCH_TO_NUMPY_DTYPE[self.dtype]

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
                scale=self.quant['scale'],
                zero_point=self.quant['zero_point'],
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


@dataclass
class PythonConstant(Data):
    data: T.Any

    @property
    def np_dtype(self) -> np.dtype:
        raise NotImplementedError()

    @property
    def tracing_data(self):
        return self.data


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


class TupleTensors(Data):
    """Should be only used as transition

    None should be remaining once graph is fully expanded
    """

    data: T.List[TensorVariable]

    @property
    def slug(self) -> str:
        slugs = ", ".join(_.slug for _ in self.data)
        return f'tupleTensor({self.export_name})({slugs})'

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


@dataclass
class ListWithTensor(Data):
    """ListWithTensor is a list that contains tensor constant or not"""

    data: T.List[Data]


@dataclass
class TorchConstant(Data):
    data: torch.Tensor

    @property
    def np_dtype(self) -> np.dtype:
        return TORCH_TO_NUMPY_DTYPE[self.data.dtype]

    @property
    def tracing_data(self):
        return self.data


def _find_data_node(data_nodes: T.List[Data], name: str):
    try:
        return next(d for d in data_nodes if d.name == name)
    except StopIteration as exp:
        names = [dnode.name for dnode in data_nodes]
        raise NotFoundDataNode(f"'{name}' not found in {names}") from exp


def _parse_getattr_tensor(node: torch._C.Node, module, data_nodes):
    tensor_name = node['name']
    data_state = getattr(module, tensor_name).data
    data_nodes.append(
        TensorVariable(
            name=tensor_name,
            shape=list(data_state.shape),
            dtype=data_state.dtype,
            data=data_state,
        )
    )


def _parse_getattr_script_obj(node: torch._C.Node, module, data_nodes):
    pack_name = node['name']
    pack = getattr(module, pack_name)
    assert isinstance(pack, torch._C.ScriptObject)
    data_nodes.append(
        BlobTorchScriptObject(
            name=pack_name,
            data=pack,
        )
    )


def _parse_contant(node: torch._C.Node, data_nodes):
    try:
        data = node['value']
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
    elif dtype == "NoneType":
        data = None
    elif dtype == "Tensor":
        assert isinstance(data, torch.Tensor)
        data_nodes.append(
            TensorVariable(
                name=name, data=data, shape=list(data.shape), dtype=data.dtype
            )
        )
        return
    elif dtype == "Device":
        # Device will not be useful info for us but we pass it to avoid
        # dereferencing it from full graph
        pass
    else:
        raise NotImplementedError(dtype)
    data_nodes.append(PythonConstant(name=name, data=data))


def _parse_list_construct(node, data_nodes):
    # should build a Data
    values = []
    contains_tensors = False
    for cvalue in node.inputs():
        value = cvalue.toIValue()
        if str(cvalue.type()) == "Tensor":
            contains_tensors = True
            value = TensorVariable.parse(cvalue)

        values.append(value)

    if contains_tensors:
        for value in values:
            if not isinstance(value, TensorVariable):
                raise NotImplementedError()
            data_nodes.append(value)
        data_node = ListWithTensor(name=node.output().debugName(), data=values)
    else:
        data_node = PythonConstant(name=node.output().debugName(), data=values)
    data_nodes.append(data_node)

    # }


def _aten_inputs_and_op_ref(kind, inputs):
    # HERE we remove unecessary OPS
    if kind in [
        "aten::sub",
        "aten::sub_",
        "aten::add",
        "aten::add_",
    ]:
        # remove useless ref to scaling (probably never used)
        inputs = inputs[:2]

    if kind in ["aten::mean", "aten::sum"]:
        inputs = inputs[:3]

    if kind == "aten::elu":
        # difference between aten and python API
        inputs = inputs[:2]

    if kind == "aten::clone":
        # remove useless ref to memory_format (for us)
        inputs = inputs[:1]
    op_ref = None
    try:
        op_ref = aten_name_to_torch_fn(kind)
    except AttributeError:
        pass
    return op_ref, inputs


def _rerouted_parsing(
    kind: str, node: torch._C.Node, data_nodes: T.List[Data], module
):
    """Specific torch Kind operation are transformed

    to improve readability of intermediate representation

        If specific kind matched it raise TorchOpTranslatedDifferently
        meaning it is handled differently than vanilla torch graph

    """
    if kind == GETATTR_KIND:
        _parse_getattr_tensor(node, module, data_nodes)
        raise TorchOpTranslatedDifferently("geattr handled as TensorVariable")
    if kind == CONSTANT_KIND:
        _parse_contant(node, data_nodes)
        raise TorchOpTranslatedDifferently("constant handled as PythonConstant")
    if kind == LISTCONSTRUCT_KIND:
        _parse_list_construct(node, data_nodes)
        raise TorchOpTranslatedDifferently(
            "List Construct handled as PythonConstant"
        )
    if kind.startswith("prim::"):
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
            # Warning ! this hard wire will only work if input
            # is only used for this cast to tensor
            dnode = _find_data_node(data_nodes, node.input().debugName())
            dnode.name = node.output().debugName()
            raise TorchOpTranslatedDifferently("NumToTensor re-wired")
        if kind != CALL_KIND:
            raise NotImplementedError(node)


def _extract_op_infos(kind: str, module, inputs, data_nodes: T.List[Data]):
    """Extract informations from module or torch operation"""
    call_name = None
    if kind == CALL_KIND:
        module_getter_ref = inputs[0].node()['name']
        op_ref = getattr(module, module_getter_ref)
        call_name = inputs[0].debugName()
        inputs = inputs[1:]
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
        op_ref, inputs = _aten_inputs_and_op_ref(kind, inputs)
    return (call_name, module_getter_ref, op_ref, inputs)


@dataclass
class TorchOp:
    kind: str
    module_path: str
    inputs: T.List[Data]
    outputs: T.List[T.Union[TensorVariable, TupleTensors]]
    scope: str
    op_ref: T.Optional[
        T.Callable[[T.Any], T.Any]
    ]  # multiple ins and outs possible
    call_name: T.Optional[str]

    @property
    def is_callmethod(self) -> bool:
        return self.kind == CALL_KIND

    @classmethod
    def _parse_outputs(cls, node: torch._C.Node, data_nodes: T.List[Data]):
        outputs: T.List[T.Union[TensorVariable, TupleTensors]] = []
        for out_node in node.outputs():  #: torch._C.Value
            if out_node.type().annotation_str != NONETYPE_KIND:
                if out_node.type().kind() == LISTTYPE_KIND:
                    raise NotImplementedError(
                        "ListType can be of arbitrary length "
                        "and will probably need a special Class to handle it"
                    )
                if out_node.type().kind() == TUPLETYPE_KIND:
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
        cls, module, node: torch._C.Node, scope: str, data_nodes: T.List[Data]
    ) -> "TorchOp":
        op_ref = None
        inputs = list(node.inputs())
        kind = node.kind()

        _rerouted_parsing(kind, node, data_nodes, module)
        (call_name, module_getter_ref, op_ref, inputs) = _extract_op_infos(
            kind, module, inputs, data_nodes
        )

        inputs = [
            _find_data_node(data_nodes, inp.debugName()) for inp in inputs
        ]

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
        if self.op_ref is not None:
            return self.op_ref(*self._args)
        raise NotImplementedError(self)

    @property
    def _args(self) -> T.Tuple[T.Any, ...]:
        return tuple(_.tracing_data for _ in self.inputs)

    def realise_output_type_and_size(self) -> bool:
        if not all(_.tracable for _ in self.inputs):
            return False

        # generate all data and call ops to infer missing infos
        results = self.call_op()

        if isinstance(results, int):
            results = torch.tensor(results, dtype=torch.int32)

        if isinstance(results, torch.Tensor):
            results = (results,)

        for data_node, result in zip(self.outputs, results):
            if isinstance(data_node, TensorVariable):
                data_node.dtype = result.dtype
                data_node.shape = list(result.shape)
                if is_quantized_dtype(result.dtype):
                    data_node.quant = {
                        "scale": result.q_scale(),
                        "zero_point": result.q_zero_point(),
                    }
        return True


class TorchModuleTraceHelper:
    SEP = "/"

    def __init__(
        self,
        module: nn.Module,
        args: T.Tuple[T.Any],
        omit_useless_nodes: bool = True,
        auto_parse: bool = True,
        inputs: T.Optional[T.List[TensorVariable]] = None,
        outputs: T.Optional[T.List[TensorVariable]] = None,
    ):
        self.op_nodes: T.List[TorchOp] = []
        self.inputs: T.List[TensorVariable] = []
        self.outputs: T.List[TensorVariable] = []
        self.data_nodes: T.List[Data] = []
        self._module = module
        self._args = maybe_quantize_args_tensor(module, args)
        self._omit_useless_nodes = omit_useless_nodes
        if auto_parse:
            self.parse(provided_inputs=inputs, provided_outputs=outputs)

    @property  # type: ignore
    @cache
    def _torch_trace(self):
        try:
            return jit.trace(self._module, self._args)
        except RuntimeError as exp:
            raise JitTraceFailed(
                "Unable to trace with jit one of following submodule:"
                f"{[(k, v.__class__) for k,v in self._module.named_children()]} "
                f"with original error:\n\n'{exp}'\n\n"
                "You can aleviate this issue by applying a special hook"
                "this module (explaination available in README)"
            ) from exp

    @property  # type: ignore
    @cache
    def _torch_graph(self):
        return self._torch_trace.graph

    def remap_node(self, from_node, to_node):
        assert isinstance(from_node, Data)
        assert isinstance(to_node, Data)
        from_node.name = to_node.name
        for op in self.op_nodes:
            op.inputs = [to_node if _ == from_node else _ for _ in op.inputs]
            op.outputs = [to_node if _ == from_node else _ for _ in op.outputs]
        self.data_nodes = [_ for _ in self.data_nodes if _ != from_node]

    def _parse_inputs(
        self, provided_inputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph inputs"""
        idx = 0
        for node_c_value in self._torch_graph.inputs():
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
        for node in self._torch_graph.nodes():
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
                    )
                    self.op_nodes.append(op)
                except TorchOpTranslatedDifferently:
                    pass

    def _parse_outputs(
        self, provided_outputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph outputs"""
        for idx, node in enumerate(self._torch_graph.outputs()):
            output = _find_data_node(self.data_nodes, node.debugName())
            if provided_outputs is not None:
                original_output = provided_outputs[idx]
                if isinstance(original_output, TupleTensors):
                    for soriginal, sout in zip(
                        original_output.data, output.data
                    ):
                        sout.shape = soriginal.shape
                        sout.dtype = soriginal.dtype
                        sout.quant = soriginal.quant
                else:
                    output.shape = original_output.shape
                    output.dtype = original_output.dtype
                    output.quant = original_output.quant
            self.outputs.append(output)

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
            for input_node in node.inputs:
                unique_name_to_scoped_name[input_node.name] = (
                    node.scope + self.SEP + input_node.name
                )

        for node in self.data_nodes:
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
        wire_inputs = callmethod_node.inputs
        wire_outputs = callmethod_node.outputs

        for node, ref_node in zip(
            _expand_tuples_if_exists(submodule_graph.inputs),
            _expand_tuples_if_exists(wire_inputs),
        ):
            for op_node in submodule_graph.op_nodes:
                op_node.inputs = [
                    innode if innode != node else ref_node
                    for innode in op_node.inputs
                ]
            for data_node in self.data_nodes:
                if isinstance(
                    data_node, (ListWithTensor, TupleTensors)
                ):  # update ref
                    data_node.data = [
                        dnode if dnode != node else ref_node
                        for dnode in data_node.data
                    ]

        for node, ref_node in zip(
            _expand_tuples_if_exists(submodule_graph.outputs),
            _expand_tuples_if_exists(wire_outputs),
        ):
            for op_node in submodule_graph.op_nodes:
                op_node.outputs = [
                    outnode if outnode != node else ref_node
                    for outnode in op_node.outputs
                ]
            for data_node in self.data_nodes:
                if isinstance(
                    data_node, (ListWithTensor, TupleTensors)
                ):  # update ref
                    data_node.data = [
                        dnode if dnode != node else ref_node
                        for dnode in data_node.data
                    ]

        to_del_nodes = submodule_graph.inputs + submodule_graph.outputs
        submodule_graph.data_nodes = [
            _ for _ in submodule_graph.data_nodes if _ not in to_del_nodes
        ]
        for _ in submodule_graph.op_nodes:
            res = _.scope.split("/", maxsplit=1)
            if len(res) >= 2 and isinstance(res, list):
                _.scope = f"{res[0]}[{prefix}]/{res[1]}"
            else:
                _.scope = f"{res}[{prefix}]"
            _.module_path = f"{module_prefix}.{_.module_path}"

        for _ in submodule_graph.data_nodes:
            _.name = f"{prefix}.{_.name}"

        self.op_nodes = [op for op in self.op_nodes if op != callmethod_node]
        self.op_nodes += submodule_graph.op_nodes
        self.data_nodes += submodule_graph.data_nodes

    def _recursive_call_method(self):
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
        ref_count = defaultdict(int)
        for op in self.op_nodes:
            if op.is_callmethod:
                ref_count[op.call_name] += 1
                assert isinstance(op, TorchOp)
                assert isinstance(op.op_ref, nn.Module)
                submodule_graph = TorchModuleTraceHelper(
                    op.op_ref,
                    op._args,
                    omit_useless_nodes=self._omit_useless_nodes,
                    inputs=op.inputs,
                    outputs=op.outputs,
                )
                self._merge_subraph(
                    submodule_graph,
                    prefix="s"  # ensure we do not start with integer varname
                    + op.call_name
                    + f"_c{ref_count[op.call_name]}",
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
            for dnode in self.data_nodes:
                prefix = {
                    TensorVariable: "v",
                    PythonConstant: "c",
                    BlobTorchScriptObject: "b",
                    TorchConstant: "t",
                    ListWithTensor: "l",
                    TupleTensors: "tt",
                    Data: "d",  # not used, avoid static analysis complain
                }[dnode.__class__]
                if dnode.name in mapping:
                    dnode.name = mapping[dnode.name]
                    continue
                suffix = count_ref[prefix]
                count_ref[prefix] += 1
                mapping[dnode.name] = prefix + str(suffix)
                dnode.name = mapping[dnode.name]

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
        self._recursive_call_method()
        self._avoid_reference_to_tuples()
        if renaming_scheme:
            self.apply_renaming_scheme(renaming_scheme)
        return self

    def printall(self):
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
            if isinstance(_, ListWithTensor):
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
                    f"[type]{o.dtype}[/type] [var]{o.export_name}[/var]"
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
