from collections import defaultdict
from functools import lru_cache
import logging
import typing as T
from dataclasses import dataclass

import numpy as np
import torch
from torch import jit, nn
import torch.jit._trace
from torch_to_nnef.dtypes import (
    TORCH_TO_NUMPY_DTYPE,
    str_to_torch_dtype,
    INT_TO_TORCH_DTYPE,
    torch_dtype_to_str,
)

from .console import Console

LOGGER = logging.getLogger(__name__)


class NodeNotFound(ValueError):
    pass


class JitTraceFailed(RuntimeError):
    pass


class UnableToTraceData(ValueError):
    pass


CALL_KIND = "prim::CallMethod"
GETATTR_KIND = "prim::GetAttr"
CONSTANT_KIND = "prim::Constant"
LISTCONSTRUCT_KIND = "prim::ListConstruct"
MODULE_PATH_ATEN = "TORCH_INTERNAL"

CLASSTYPE_KIND = "ClassType"

SPECIAL_ATEN_REMAP_PYTORCH = {"__and__": "bitwise_and", "__or__": "bitwise_or"}


def aten_name_to_torch_fn(aten_name):
    name = aten_name.replace("aten::", "")
    name = SPECIAL_ATEN_REMAP_PYTORCH.get(name, name)
    try:
        return getattr(torch, name)
    except AttributeError:
        return getattr(torch.nn.functional, name)


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


def _find_common_root(
    elements: T.Iterable[str], sep: str, shortest: bool = False, base: str = ""
) -> str:
    common_root = base
    scope_name_appeared = {}
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
            torch.quantize_per_tensor(
                in_item.float(),
                torch.tensor(0.1),
                torch.tensor(0),
                dtype=torch.quint8,
            )
            if isinstance(in_item, torch.Tensor)
            and (
                in_item.dtype
                not in [
                    torch.qint32,
                    torch.qint8,
                    torch.quint4x2,
                    torch.quint8,
                ]
            )
            else in_item
            for in_item in args
        ]
    return args


@dataclass
class Data:
    name: str

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


@dataclass
class TensorVariable(Data):

    shape: T.Optional[T.List[int]]
    dtype: T.Optional[torch.dtype]
    data: T.Optional[torch.Tensor] = None  # serve as reference

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
    def tracing_data(self):
        if not self.shaped_and_typed:
            raise UnableToTraceData(self)
        if self.data is not None:
            return self.data
        return torch.rand(
            ([321 if x is None else x for x in (self.shape or [])])
        ).to(self.dtype)

    @classmethod
    def parse(cls, node_c_value: torch._C.Value) -> "TensorVariable":
        node_type = node_c_value.type()
        stype = node_type.scalarType()
        return cls(
            name=node_c_value.debugName(),
            shape=node_type.sizes(),
            dtype=str_to_torch_dtype(stype) if stype else None,
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
class TorchConstant(Data):
    data: torch.Tensor

    @property
    def np_dtype(self) -> np.dtype:
        return TORCH_TO_NUMPY_DTYPE[self.data.dtype]

    @property
    def tracing_data(self):
        return self.data


@dataclass
class TorchOp:
    kind: str
    module_path: str
    inputs: T.List[Data]
    outputs: T.List[Data]
    scope: str
    op_ref: T.Callable[[T.Any], T.Any]  # multiple ins and outs possible
    call_name: T.Optional[str]

    @property
    def is_callmethod(self) -> bool:
        return self.kind == CALL_KIND

    @classmethod
    def parse(
        cls, module, node: torch._C.Node, scope: str, data_nodes: T.List[Data]
    ) -> "TorchOp":
        op_ref = None

        inputs = list(node.inputs())
        call_name = None
        kind = node.kind()
        if kind == CALL_KIND:
            module_getter_ref = inputs[0].node()['name']
            op_ref = getattr(module, module_getter_ref)
            inputs = inputs[1:]
            call_name = inputs[0].debugName()
        elif kind == GETATTR_KIND:
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
            return
        elif kind == CONSTANT_KIND:
            try:
                data = node['value']
            except RuntimeError:
                data = None
            data_nodes.append(
                PythonConstant(name=node.output().debugName(), data=data)
            )
            return
        elif kind == LISTCONSTRUCT_KIND:
            # should build a Data
            values = []
            for cvalue in node.inputs():
                values.append(cvalue.toIValue())
            data_nodes.append(
                PythonConstant(name=node.output().debugName(), data=values)
            )
            # }
            return
        else:
            module_getter_ref = MODULE_PATH_ATEN
            # HERE we remove unecessary OPS
            if kind.endswith("_"):
                # allow to find correct pytorch API fn
                kind = kind[:-1]
            if kind in [
                "aten::sub",
                "aten::add",
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
            try:
                op_ref = aten_name_to_torch_fn(kind)
            except AttributeError:
                pass

        outputs = []
        for out_node in node.outputs():  #: torch._C.Value
            out = TensorVariable.parse(out_node)
            data_nodes.append(out)
            outputs.append(out)

        return cls(
            kind=kind,
            inputs=[
                next(d for d in data_nodes if d.name == inp.debugName())
                for inp in inputs
            ],
            outputs=outputs,
            scope=scope,
            module_path=module_getter_ref,
            op_ref=op_ref,
            call_name=call_name,
        )

    def call_op(self, input_args):
        if self.kind == "aten::to":
            # note wrong type
            results = input_args[0].to(INT_TO_TORCH_DTYPE[input_args[1]])
        elif self.kind == "aten::repeat":
            results = input_args[0].repeat(input_args[1])
        elif self.kind in [
            "aten::reflection_pad1d",
            "aten::reflection_padnd",
        ]:
            results = nn.functional.pad(
                input_args[0], pad=input_args[1], mode="reflect"
            )
        elif self.kind in [
            "aten::replication_pad1d",
            "aten::replication_padnd",
        ]:
            results = nn.functional.pad(
                input_args[0], pad=input_args[1], mode="replicate"
            )
        else:
            if self.op_ref is not None:
                results = self.op_ref(*self._args)
            else:
                raise NotImplementedError(self)
        return results

    @property
    def _args(self) -> T.Tuple[T.Any]:
        return tuple(_.tracing_data for _ in self.inputs)

    def realise_output_type_and_size(self) -> bool:
        if not all(_.shaped_and_typed for _ in self.inputs):
            return False
        # generate all data
        # and call ops to infer missing infos
        results = self.call_op(self._args)
        if isinstance(results, torch.Tensor):
            results = (results,)
        for data_node, result in zip(self.outputs, results):
            data_node.dtype = result.dtype
            data_node.shape = list(result.shape)
        return True


class TorchModuleTraceHelper:
    SEP = "/"

    def __init__(
        self,
        module: nn.Module,
        args: T.Tuple[T.Any],
        omit_useless_nodes: bool = True,
        auto_parse: bool = True,
    ):
        self.op_nodes: T.List[TorchOp] = []
        self.data_nodes: T.List[Data] = []
        self.inputs: T.List[TensorVariable] = []
        self.outputs: T.List[TensorVariable] = []
        self._module = module
        self._args = maybe_quantize_args_tensor(module, args)
        self._omit_useless_nodes = omit_useless_nodes
        if auto_parse:
            self.parse()

    @property
    @lru_cache(1)
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

    @property
    @lru_cache(1)
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

    def _parse_inputs(self):
        """Parse traced graph inputs"""
        for node_c_value in self._torch_graph.inputs():
            if self._omit_useless_nodes:
                if (
                    len(node_c_value.uses()) == 0
                ):  # number of user of the node_c_value (= number of outputs/ fanout)
                    continue

            if node_c_value.type().kind() != CLASSTYPE_KIND:
                tv = TensorVariable.parse(node_c_value)
                self.inputs.append(tv)
                self.data_nodes.append(tv)

    def _parse_core(self):
        """Parse all Operations and collect the scope infos"""
        attr_to_scope: T.Dict[T.Any, str] = dict()
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
                    attr_to_scope[attr_name] = "{}/{}.{}".format(
                        parent_scope, attr_scope, attr_name
                    )
                else:
                    attr_to_scope[attr_name] = f"__module.{attr_name}"
                # We don't need classtype nodes; scope will provide this information
                if node.output().type().kind() != CLASSTYPE_KIND:
                    op = TorchOp.parse(
                        self._module,
                        node,
                        scope=attr_to_scope[attr_name],
                        data_nodes=self.data_nodes,
                    )
                    if op is not None:
                        self.op_nodes.append(op)

            else:
                op = TorchOp.parse(
                    self._module,
                    node,
                    scope="",
                    data_nodes=self.data_nodes,
                )
                if op is not None:
                    self.op_nodes.append(op)

    def _parse_outputs(self):
        """Parse traced graph outputs"""
        for node in self._torch_graph.outputs():
            self.outputs.append(
                next(d for d in self.data_nodes if d.name == node.debugName())
            )

    def _update_scope_reference(self):
        """Update scope in op_nodes with additional infos"""
        alias_to_name = dict()
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
            if not node.shaped_and_typed:
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
                LOGGER.warning(
                    "following nodes doesn't have shape concretized:"
                )
                for _ in unshaped_data.values():
                    LOGGER.warning("\t%s", _)
                break

    def merge_subraph(
        self, submodule_graph, callmethod_node, prefix: str, module_prefix: str
    ):
        # TODO <<====
        # Re-Wire input and output naming => {
        wire_inputs = callmethod_node.inputs
        wire_outputs = callmethod_node.outputs

        for node, ref_node in zip(submodule_graph.inputs, wire_inputs):
            for op_node in submodule_graph.op_nodes:
                op_node.inputs = [
                    innode if innode != node else ref_node
                    for innode in op_node.inputs
                ]

        for node, ref_node in zip(submodule_graph.outputs, wire_outputs):
            for op_node in submodule_graph.op_nodes:
                op_node.outputs = [
                    outnode if outnode != node else ref_node
                    for outnode in op_node.outputs
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

    def recursive_call_method(self):
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
                )
                self.merge_subraph(
                    submodule_graph,
                    prefix="s"  # ensure we do not start with integer varname
                    + op.call_name
                    + f"_c{ref_count[op.call_name]}",
                    module_prefix=op.module_path,
                    callmethod_node=op,
                )

    def parse(self):
        self._parse_inputs()
        self._parse_core()
        self._parse_outputs()
        self._update_scope_reference()
        self._update_data_node_name_with_base_context()
        self._infer_missing_shapes_from_ops_outputs()
        self.recursive_call_method()
        return self

    @classmethod
    def parse_model(cls, module, args, omit_useless_nodes=True):
        """This method parses a PyTorch model graph and produces
        a list of nodes and node stats for eventual conversion to NNEF format.
        """
        graph_helper = cls(
            module,
            args,
            omit_useless_nodes=omit_useless_nodes,
        )
        graph_helper.parse()
        return graph_helper

    def printall(self):
        raise NotImplementedError()
