import logging
import typing as T
from collections import OrderedDict
from itertools import chain
from dataclasses import dataclass

import torch
from torch import jit, nn
import torch.jit._trace

from .console import Console

LOGGER = logging.getLogger(__name__)

methods_OP = [
    "attributeNames",
    "hasMultipleOutputs",
    "hasUses",
    "inputs",
    "kind",
    "outputs",
    "outputsSize",
    "scopeName",
]
# Some additional methods to explure for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.
methods_IO = ["debugName"]  # "node", "offset",

GETATTR_KIND = "prim::GetAttr"
CLASSTYPE_KIND = "ClassType"

UNKNOWN_SHAPE = "unknown_shape"
UNKNOWN_DTYPE = "unknown_type"


def aten_name_to_torch_fn(aten_name):
    name = aten_name.replace("aten::", "")
    try:
        return getattr(torch, name)
    except AttributeError:
        return getattr(torch.nn.functional, name)


def _replacement_to_relative_module_path(replacements: T.List[str]):
    return ".".join(
        [rep.split("[")[1][:-1] if "[" in rep else rep for rep in replacements]
    )


def _access_module(module_path: str, model: nn.Module):
    current_context = model
    for next_context_str in module_path.split("."):
        try:
            intcasted = int(next_context_str)
            current_context = current_context[intcasted]
            continue
        except ValueError:
            pass
        current_context = getattr(current_context, next_context_str)
    return current_context


@dataclass
class NodeBase:
    debugName: str
    inputs: T.List[str]
    scope: T.Optional[str]
    kind: T.Optional[str]

    def apply_prefix(
        self, prefix: str, skip_names: T.Optional[T.List[str]] = None
    ):
        skip_names = skip_names or []
        if self.debugName not in skip_names:
            self.debugName = f"{prefix}.{self.debugName}"

            res = self.scope.split("/", maxsplit=1)
            if len(res) >= 2 and isinstance(res, list):
                self.scope = f"{res[0]}[{prefix}]/{res[1]}"
            else:
                self.scope = f"{res}[{prefix}]"
        for idx, _ in enumerate(self.inputs):
            if _ not in skip_names:
                self.inputs[idx] = f"{prefix}.{_}"

        if hasattr(self, "outputs"):
            for idx, _ in enumerate(self.outputs):
                if _ not in skip_names:
                    self.outputs[idx] = f"{prefix}.{_}"
        if hasattr(self, "module_path"):
            self.module_path = f"{prefix}.{self.module_path}"

    @property
    def is_callmethod(self) -> bool:
        return self.kind == "prim::CallMethod"

    @property
    def is_getattr(self) -> bool:
        return self.kind == "prim::GetAttr"

    def _refid_clean(self, name: str) -> str:
        for sep in ["/", "[", "]", ".", "-"]:
            name = name.replace(sep, "_")
        return name

    @property
    def export_name(self) -> str:
        return self._refid_clean(self.debugName)

    @property
    def export_inputs(self) -> T.List[str]:
        return [self._refid_clean(i) for i in self.inputs]

    @classmethod
    def _parse_debug_name(cls, node_cpp) -> str:
        if hasattr(node_cpp, "debugName"):
            return node_cpp.debugName().strip()
        return str(node_cpp).strip()

    @classmethod
    def parse_args(cls, node_cpp, valid_methods):
        kwargs = {"debugName": cls._parse_debug_name(node_cpp), "inputs": []}
        valid_methods = valid_methods[:]

        for m in valid_methods:
            if m == "inputs" or m == "outputs":
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())
                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)

                kwargs[m] = io_unique_names
                kwargs[f"{m}_tensor_size"] = io_tensor_sizes
            else:
                kwargs[m] = getattr(node_cpp, m)()
        if "scopeName" in kwargs:
            kwargs["scope"] = kwargs["scopeName"]
            del kwargs["scopeName"]
        return kwargs

    @classmethod
    def parse(cls, node_cpp, valid_methods):
        return cls(**cls.parse_args(node_cpp, valid_methods))


@dataclass
class NodeIO(NodeBase):
    tensor_size: T.Union[
        None,
        T.List[int],
        T.Tuple[T.Union[T.List[T.Union[int, None]], T.Tuple[int]], ...],
    ]
    dtype: str
    subtype: str

    @property
    def tracing_data(self):
        return torch.rand(
            tuple([321 if x is None else x for x in self.tensor_size])
        )

    @classmethod
    def _cpp_tensor_size_from_type(cls, node_type):
        final_size = None
        if node_type.kind() == "TupleType":
            final_size = []
            for sub_node_type in node_type.elements():
                final_size.append(cls._cpp_tensor_size_from_type(sub_node_type))
            final_size = tuple(final_size)
        elif node_type.kind() == "ListType":
            final_size = [
                cls._cpp_tensor_size_from_type(node_type.getElementType())
            ]
        else:
            # if isinstance(ntype.kind(), "Tensor"):
            final_size = node_type.sizes()
        return final_size

    @classmethod
    def parse_args(cls, node_cpp):
        node_args = super().parse_args(node_cpp, methods_IO)
        try:
            tensor_size = cls._cpp_tensor_size_from_type(node_cpp.type())
        except RuntimeError:
            tensor_size = [
                1,
            ]  # fail when constant model is used.
        node_args["tensor_size"] = tensor_size
        # Kind attribute string is purely descriptive
        # node_args["kind"] = "Parameter"
        node_args["kind"] = "IO Node"

        dtype = node_cpp.type().annotation_str
        node_args["dtype"] = dtype

        subtype = (
            node_cpp.type().scalarType() if dtype == "Tensor" else ""
        ) or ""
        node_args["subtype"] = subtype
        if "scope" not in node_args:
            node_args["scope"] = ""
        return node_args

    @classmethod
    def parse(cls, node_cpp):
        return cls(**cls.parse_args(node_cpp))

    @property
    def print_slug(self):
        if self.tensor_size:
            shape_str = f"({','.join(str(_) for _ in self.tensor_size)})"
        else:
            shape_str = UNKNOWN_SHAPE
        return f"[var]{self.export_name}[/var]={shape_str}"

    @property
    def input_or_output(self):
        return "input" if isinstance(self, NodeInput) else "output"


@dataclass
class NodeInput(NodeIO):
    pass


@dataclass
class NodeOutput(NodeIO):
    pass


@dataclass
class NodeOp(NodeBase):

    attributes: T.Dict[str, T.Any]

    attributeNames: T.Optional[T.List[str]]
    inputs_tensor_size: T.List[T.Union[None, T.List[int]]]
    outputs: T.List[str]
    outputs_tensor_size: T.List[T.Union[None, T.List[int]]]
    outputsSize: int

    dtype: str
    subtype: str

    hasMultipleOutputs: bool
    hasUses: bool
    module_path: str

    @property
    def tracing_data(self):
        if "values" in self.attributes:
            return self.attributes["values"]
        if self.outputs_tensor_size is None:
            raise ValueError(self)
        shape = [321 if x is None else x for x in self.outputs_tensor_size]

        return torch.rand(tuple(shape))

    @classmethod
    def parse_args(cls, node_cpp, valid_methods):
        node_args = super().parse_args(node_cpp, valid_methods)
        node_args["attributes"] = {
            k: node_cpp[k] for k in node_cpp.attributeNames()
        }
        kind = node_cpp.kind()
        node_args["kind"] = kind
        dtype = node_cpp.output().type().annotation_str
        node_args["dtype"] = dtype

        node_args["subtype"] = (
            node_cpp.output().type().scalarType() if dtype == "Tensor" else ""
        ) or ""
        node_args["module_path"] = ""
        return node_args

    @classmethod
    def parse(cls, node_cpp):
        node_args = cls.parse_args(node_cpp, methods_OP)

        if node_cpp.kind() == "prim::Constant":
            node_args["value"] = node_cpp.output().toIValue()
            return NodeConstant(**node_args)
        return cls(**node_args)


@dataclass
class NodeConstant(NodeOp):
    value: T.Any

    @property
    def tracing_data(self):
        return self.value


@dataclass
class NodeClassType(NodeOp):
    className: str

    @classmethod
    def parse(cls, node_cpp):
        node_args = cls.parse_args(node_cpp, methods_OP)
        node_args["className"] = node_cpp.output().type().annotation_str
        return cls(**node_args)


@dataclass
class NodeTensorSized(NodeBase):

    attributes: T.Dict[str, T.Any]
    tensor_size: T.Optional[T.List[int]]
    dtype: str
    subtype: str
    module_path: str

    @property
    def tracing_data(self):
        if "values" in self.attributes:
            return self.attributes["values"]
        return torch.rand(
            tuple(self.tensor_size),
            # TODO apply stric mapping between dtype/subtype and torch proper dtype
            # dtype=realised_node[input_name].dtype,
        )

    @classmethod
    def from_nodeOP(
        cls,
        debugName: str,
        tensor_size: T.List[int],
        node: NodeOp,
    ):
        if isinstance(node, NodeConstant):
            return NodeConstantTensorSized(
                inputs=node.inputs,
                scope=node.scope,
                tensor_size=tensor_size,
                kind=node.kind,
                attributes=node.attributes,
                debugName=debugName,
                dtype=node.dtype,
                subtype=node.subtype,
                value=node.value,
                module_path=node.module_path,
            )
        return cls(
            inputs=node.inputs,
            scope=node.scope,
            tensor_size=tensor_size,
            kind=node.kind,
            attributes=node.attributes,
            debugName=debugName,
            dtype=node.dtype,
            subtype=node.subtype,
            module_path=node.module_path,
        )


@dataclass
class NodeState(NodeTensorSized):
    data: torch.Tensor

    @classmethod
    def parse(cls, state, getattr_node: NodeTensorSized):
        node_args = getattr_node.__dict__
        node_args['data'] = state
        node_args['tensor_size'] = tuple(state.shape)
        return cls(**node_args)


@dataclass
class NodeConstantTensorSized(NodeTensorSized):
    value: T.Any

    @property
    def tracing_data(self):
        return self.value


class InternalPytorchGraphHelper:
    SEP = "/"

    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = "default"
        self.scope_name_appeared = []
        self.state_nodes = []

    def append(self, x):
        if isinstance(x, NodeIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodeOp):
            self.nodes_op.append(x)

    @property
    def inputs_nodes(self):
        return [_ for _ in self.nodes_io.values() if isinstance(_, NodeInput)]

    @property
    def outputs_nodes(self):
        return [_ for _ in self.nodes_io.values() if isinstance(_, NodeOutput)]

    @property
    def constant_nodes(self):
        return [
            _
            for _ in self.nodes_io.values()
            if isinstance(_, NodeConstantTensorSized)
        ]

    @property
    def dag_nodes(self):
        return [
            _
            for _ in self.nodes_io.values()
            if isinstance(_, NodeTensorSized) and _.kind != "prim::Constant"
        ]

    @property
    def operators_nodes(self):
        return [
            _
            for _ in self.nodes_io.values()
            if _.kind.startswith("aten::")
            or (
                _.kind.startswith("prim::")
                and not _.kind.startswith("prim::Constant")
                and not _.kind.startswith("prim::GetAttr")
            )
        ]

    def get_node_by_export_name(self, name: str):
        try:
            return next(_ for _ in self.state_nodes if _.export_name == name)
        except StopIteration:
            pass
        try:
            return next(_ for _ in self.dag_nodes if _.export_name == name)
        except StopIteration:
            pass
        return next(_ for _ in self.constant_nodes if _.export_name == name)

    def get_node_by_debug_name(self, name: str):
        return next(_ for _ in self.nodes_io.values() if _.debugName == name)

    def printall(self):
        console = Console(
            theme={
                "type": "blue",
                "var": "grey82",
                "kind": "yellow",
                "subsection": "red",  # dim bold
            }
        )
        print = console.print
        print(
            "\n\n[type]"
            + "_" * 35
            + "[Pytorch JIT Graph]"
            + "_" * 35
            + "[/type]"
        )
        inputs_str = ", ".join(_.print_slug for _ in self.inputs_nodes)
        print(f"inputs: ({inputs_str})")
        print("")
        print(f"\t[subsection]Static Constants:[/subsection]")
        for _ in self.constant_nodes:

            print(
                f"\t\t[type]{_.dtype}{_.subtype}[/type] "
                f"[var]{_.export_name}[/var] := {_.value}"
            )

        print()
        print(f"\t[subsection]Static States:[/subsection]")
        for _ in self.state_nodes:
            print(
                f"\t\t[type]{_.dtype}{_.subtype}[/type] "
                f"[var]{_.export_name}[/var] := shape{tuple(_.tensor_size)}"
            )

        print("")
        print(f"\t[subsection]Directed Acyclic Graph:[/subsection]")
        for _ in self.dag_nodes:
            inputs_str = ""
            if _.inputs:
                inputs_str = ", ".join(
                    f"[var]{i}[/var]" for i in _.export_inputs
                )
                inputs_str = f"( {inputs_str} )"
            cls_name = ""
            if isinstance(_, NodeClassType):
                cls_name = " cls='{_.className}'"
            print(
                f"\t\t[type]{_.dtype}{_.subtype}[/type] "
                f"[var]{_.export_name}[/var] := "
                f"[kind]{_.kind}[/kind]{inputs_str}{cls_name}"
            )

        outputs_str = ", ".join(_.print_slug for _ in self.outputs_nodes)
        print("")
        print(f"outputs: ({outputs_str})")
        print("[type]" + "_" * 100 + "[/type]")

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split(self.SEP)[0]

    def _populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for node_output, outputSize in zip(
                node.outputs, node.outputs_tensor_size
            ):
                self.scope_name_appeared.append(node.scope)
                self.nodes_io[node_output] = NodeTensorSized.from_nodeOP(
                    debugName=node_output, tensor_size=outputSize, node=node
                )

        self.find_common_root()

        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = (
                    node.scope + self.SEP + input_node_id
                )

        for key, node in self.nodes_io.items():
            if isinstance(node, NodeTensorSized):
                self.unique_name_to_scoped_name[key] = (
                    node.scope + self.SEP + node.debugName
                )
            if isinstance(node, NodeIO):
                self.unique_name_to_scoped_name[key] = (
                    node.input_or_output + self.SEP + node.debugName
                )
            if hasattr(node, "scope") and node.scope is not None:
                self.unique_name_to_scoped_name[key] = (
                    node.scope + self.SEP + node.debugName
                )
                if node.scope == "" and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = (
                        self.shallowest_scope_name + self.SEP + node.debugName
                    )

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name[node_input_id]
                for node_input_id in node.inputs
                if isinstance(node, NodeBase)
            ]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[
                    node.debugName
                ]

    def find_io_by_debug_name(self, name: str):
        return next(dn for dn in self.nodes_io.values() if dn.debugName == name)

    def _infer_undefined_tensor_size_when_possible(self, model):
        realised_node = {
            node.debugName: node
            for node in chain(self.inputs_nodes, self.constant_nodes)
        }
        realised_node.update(
            {self.SEP + node.debugName: node for node in self.state_nodes}
        )

        # need ListConstruct to become realised_node with value {
        for node in self.dag_nodes:
            if node.kind == "prim::ListConstruct":
                values = []
                for inp_name in node.inputs:
                    in_node = self.get_node_by_debug_name(inp_name)
                    if isinstance(in_node, NodeConstantTensorSized):
                        values.append(in_node.value)
                    else:
                        break
                if len(values) == len(node.inputs):
                    # is realised
                    node.attributes['values'] = values
                    node.tensor_size = len(values)
                    realised_node[node.debugName] = node
                else:
                    raise NotImplementedError(
                        "Not sure what to do in such condition"
                    )
        # }

        remaining_nodes = [
            node
            for node in self.dag_nodes
            if node not in realised_node.values()
        ]

        while remaining_nodes:
            nodes_to_del = []
            for node in remaining_nodes:
                # TODO handle special case when realised node is only partially
                # concrete with some dimenssion unknown
                inputs = node.inputs
                if not all(
                    input_name in realised_node for input_name in inputs
                ):
                    continue

                if not node.kind.startswith("aten::"):
                    raise NotImplementedError(node)

                if node.kind == "aten::elu":
                    # difference between aten and python API
                    inputs = inputs[:2]

                input_args = [
                    realised_node[input_name].tracing_data
                    for input_name in inputs
                ]
                results = aten_name_to_torch_fn(node.kind)(*input_args)
                node.tensor_size = tuple(results.shape)
                realised_node[node.debugName] = node
                nodes_to_del += [node]

            remaining_nodes = [
                _ for _ in remaining_nodes if _ not in nodes_to_del
            ]
            if len(nodes_to_del) == 0:
                if len(remaining_nodes) > 0:
                    LOGGER.debug(
                        "following nodes doesn't have shape concretized:"
                    )
                    for _ in remaining_nodes:
                        LOGGER.debug("\t", _)
                break

    def check_is_valid(self):
        for node in self.dag_nodes:
            if node.kind == "prim::CallMethod":
                raise ValueError(f"unwanted parsed node {node}")

    def _transform_prim_get_attr_in_state_nodes(self, module):
        keys_to_del = []
        for io_key, node in self.nodes_io.items():
            if (
                isinstance(node, NodeTensorSized)
                and node.kind != "prim::Constant"
                and node.is_getattr
                and node.dtype == "Tensor"
            ):
                keys_to_del.append(io_key)
                data_state = getattr(module, node.module_path).data
                self.state_nodes.append(NodeState.parse(data_state, node))
        for key in keys_to_del:
            del self.nodes_io[key]

    def merge_subraph(self, submodule_graph, prefix: str, callmethod_node):
        # Re-Wire input and output naming => {
        wire_inputs = callmethod_node.inputs[1:]
        wire_output = callmethod_node.debugName

        to_del_names = []
        for node, new_name in zip(submodule_graph.inputs_nodes, wire_inputs):
            to_del_names.append(node.debugName.split(self.SEP)[1])
            for vnode_op in submodule_graph.dag_nodes:
                vnode_op.inputs = [
                    new_name if _ == node.debugName else _
                    for _ in vnode_op.inputs
                ]

        for to_del_name in to_del_names:
            del submodule_graph.nodes_io[to_del_name]

        submodule_graph.find_io_by_debug_name(
            submodule_graph.outputs_nodes[0].inputs[0]
        ).debugName = wire_output
        del submodule_graph.nodes_io["output.1"]

        assert len(submodule_graph.inputs_nodes) == 0
        assert len(submodule_graph.outputs_nodes) == 0

        # }

        for _ in submodule_graph.nodes_op:
            _.apply_prefix(prefix, skip_names=wire_inputs + [wire_output])
            # .inputs .outputs
        # self.nodes_io = OrderedDict()
        for _ in submodule_graph.nodes_io.values():
            _.apply_prefix(prefix, skip_names=wire_inputs + [wire_output])

        for _ in submodule_graph.state_nodes:
            _.apply_prefix(prefix, skip_names=wire_inputs + [wire_output])

        self.nodes_op += submodule_graph.nodes_op
        self.nodes_io.update(
            {f"{prefix}{k}": v for k, v in submodule_graph.nodes_io.items()}
        )

        self.state_nodes += submodule_graph.state_nodes

        self.nodes_io = {
            k: v
            for k, v in self.nodes_io.items()
            if v
            not in [
                callmethod_node,
                self.get_node_by_debug_name(callmethod_node.inputs[0]),
            ]
        }

    def recursive_call_method(self, module, args, omit_useless_nodes):
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
        for dag_node in self.dag_nodes:
            if dag_node.is_callmethod:
                ref_getter_node = self.get_node_by_debug_name(
                    dag_node.inputs[0]
                )
                # prep recursion
                submodule = getattr(module, ref_getter_node.module_path)
                submodule_args = tuple(
                    [
                        self.get_node_by_debug_name(
                            submodule_in_nodename
                        ).tracing_data
                        for submodule_in_nodename in dag_node.inputs[1:]
                    ]
                )
                try:
                    submodule_graph = InternalPytorchGraphHelper().parse_module(
                        submodule, submodule_args, omit_useless_nodes
                    )
                    self.merge_subraph(
                        submodule_graph,
                        prefix=ref_getter_node.module_path,
                        callmethod_node=dag_node,
                    )
                except RuntimeError as exp:
                    print(exp)
                    import ipdb

                    ipdb.set_trace()

                    pass

    def parse_module(self, module, args, omit_useless_nodes):
        # while it is recommended to not use this func it expand correctly
        # the graph which is essential in or usecase
        origin_module = module
        trace = jit.trace(module, args)
        graph = trace.graph
        for idx, node in enumerate(graph.inputs()):
            if omit_useless_nodes:
                if (
                    len(node.uses()) == 0
                ):  # number of user of the node (= number of outputs/ fanout)
                    continue

            if node.type().kind() != CLASSTYPE_KIND:
                self.append(NodeInput.parse(node))

        attr_to_scope: T.Dict[T.Any, str] = dict()
        for node in graph.nodes():
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
                if node.output().type().kind() == CLASSTYPE_KIND:
                    node_py = NodeClassType.parse(node)
                else:
                    node_py = NodeOp.parse(node)

                node_py.scope = attr_to_scope[attr_name]  # type: ignore[attr-defined]
                self.append(node_py)
            else:
                self.append(NodeOp.parse(node))

        for i, node in enumerate(
            graph.outputs()
        ):  # Create sink nodes for output ops
            node_pyio = NodeOutput.parse(node)
            node_pyio.debugName = f"output.{i + 1}"
            node_pyio.inputs = [node.debugName()]
            self.append(node_pyio)

        def parse_traced_name(module):
            if isinstance(module, jit.TracedModule):
                module_name = module._name
            else:
                module_name = getattr(module, "original_name", "Module")
            return module_name

        alias_to_name = dict()
        base_name = parse_traced_name(trace)
        for name, module in trace.named_modules(prefix="__module"):
            mod_name = parse_traced_name(module)
            attr_name = name.split(".")[-1]
            alias_to_name[name] = f"{mod_name}[{attr_name}]"

        for node in self.nodes_op:
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

        self._populate_namespace_from_OP_to_IO()
        self._transform_prim_get_attr_in_state_nodes(module)

        self._infer_undefined_tensor_size_when_possible(module)

        # self.check_is_valid()
        self.recursive_call_method(origin_module, args, omit_useless_nodes)

        return self

    @classmethod
    def parse_model(cls, module, args, omit_useless_nodes=True):
        """This method parses an optimized PyTorch model graph and produces
        a list of nodes and node stats for eventual conversion to NNEF format.
        """

        graph_helper = cls()
        graph_helper.parse_module(
            module,
            args,
            omit_useless_nodes=omit_useless_nodes,
        )
        return graph_helper
