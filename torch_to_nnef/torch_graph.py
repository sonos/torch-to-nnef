import typing as T
from collections import OrderedDict
from itertools import chain
from dataclasses import dataclass

import torch
from torch import jit, nn
import torch.jit._trace

from .console import Console

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


def clean_dtype_name(dtype_str: str) -> str:
    old_slug_parts = dtype_str.split(".")
    new_slug_parts = []
    for old_slug_part in old_slug_parts:
        if not old_slug_part.startswith("___torch_mangle_"):
            new_slug_parts.append(old_slug_part)
    return ".".join(new_slug_parts)


def _replacement_to_relative_module_path(replacements: T.List[str]):
    return ".".join([rep.split("[")[1][:-1] for rep in replacements])


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
    def slug(self):
        if self.tensor_size:
            shape_str = f"({','.join(str(_) for _ in self.tensor_size)})"
        else:
            shape_str = UNKNOWN_SHAPE
        return f"[var]{self.debugName}[/var]={shape_str}"

    @property
    def input_or_output(self):
        return "input" if isinstance(self, NodeInput) else "output"


@dataclass
class NodeState(NodeIO):
    data: torch.Tensor

    @classmethod
    def parse(cls, state, node):
        node_args = super().parse_args(node)
        node_args['data'] = state
        return cls(**node_args)


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
class NodeConstantTensorSized(NodeTensorSized):
    value: T.Any


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

    def get_node_by_name(self, name: str):
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
        inputs_str = ", ".join(_.slug for _ in self.inputs_nodes)
        print(f"inputs: ({inputs_str})")
        print("")
        print(f"\t[subsection]Static Constants:[/subsection]")
        for _ in self.constant_nodes:
            var_name = _.debugName.split(self.SEP, maxsplit=1)[1]
            print(
                f"\t\t[type]{_.dtype}{_.subtype}[/type] "
                f"[var]{var_name}[/var] := {_.value}"
            )

        print(f"\t[subsection]Static States:[/subsection]")
        for _ in self.state_nodes:
            print(
                f"\t\t[type]{_.dtype}{_.subtype}[/type] "
                f"[var]{_.debugName}[/var] := shape{tuple(_.tensor_size)}"
            )

        print("")
        print(f"\t[subsection]Directed Acyclic Graph:[/subsection]")
        for _ in self.dag_nodes:
            inputs_str = ""
            if _.inputs:
                inputs_str = ", ".join(
                    f"[var]{i.split(self.SEP, maxsplit=1)[1]}[/var]"
                    for i in _.inputs
                )
                inputs_str = f"( {inputs_str} )"
            cls_name = ""
            if isinstance(_, NodeClassType):
                cls_name = " cls='{_.className}'"
            varname = _.debugName.split(self.SEP, maxsplit=1)[1]
            print(
                f"\t\t[type]{_.dtype}{_.subtype}[/type] "
                f"[var]{varname}[/var] := "
                f"[kind]{_.kind}[/kind]{inputs_str}{cls_name}"
            )

        outputs_str = ", ".join(_.slug for _ in self.outputs_nodes)
        print("")
        print(f"outputs: ({outputs_str})")
        print("[type]" + "_" * 100 + "[/type]")

    def get_outputs_of_node(self, node):
        nodes = []
        for output_name in node.outputs:
            nodes.append(
                self.nodes_io[output_name.split(self.SEP, maxsplit=1)[1]]
            )
        return nodes

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
                    in_node = self.get_node_by_name(inp_name)
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

                input_args = []
                for input_name in inputs:
                    rnode = realised_node[input_name]
                    if isinstance(rnode, NodeConstantTensorSized):
                        val = rnode.value
                    elif rnode.kind == "prim::ListConstruct":
                        val = rnode.attributes['values']
                    else:
                        val = torch.rand(
                            tuple(rnode.tensor_size),
                            # TODO need proper dtype
                            # dtype=realised_node[input_name].dtype,
                        )
                    input_args.append(val)
                results = getattr(torch, node.kind.replace("aten::", ""))(
                    *input_args
                )
                node.tensor_size = tuple(results.shape)
                realised_node[node.debugName] = node
                nodes_to_del += [node]

            remaining_nodes = [
                _ for _ in remaining_nodes if _ not in nodes_to_del
            ]
            if len(nodes_to_del) == 0:
                if len(remaining_nodes) > 0:
                    print("following nodes doesn't have shape concretized:")
                    for _ in remaining_nodes:
                        print("\t", _)
                break

    def check_is_valid(self):
        for node in self.dag_nodes:
            if node.kind == "prim::CallMethod":
                raise ValueError(f"unwanted parsed node {node}")

    @classmethod
    def parse_module(
        cls, graph_helper, original_model, args, omit_useless_nodes
    ):
        # while it is recommended to not use this func it expand correctly
        # the graph which is essential in or usecase
        qte_args = 1
        if isinstance(args, tuple):
            qte_args = len(args)
        traced = jit.trace(original_model, args)
        graph = traced.graph
        for idx, node in enumerate(graph.inputs()):
            if omit_useless_nodes:
                if (
                    len(node.uses()) == 0
                ):  # number of user of the node (= number of outputs/ fanout)
                    continue

            if node.type().kind() != CLASSTYPE_KIND:
                if idx < qte_args:
                    graph_helper.append(NodeInput.parse(node))
                else:
                    # TODO address states
                    pass

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
                graph_helper.append(node_py)
            else:
                graph_helper.append(NodeOp.parse(node))

        for i, node in enumerate(
            graph.outputs()
        ):  # Create sink nodes for output ops
            node_pyio = NodeOutput.parse(node)
            node_pyio.debugName = f"output.{i + 1}"
            node_pyio.inputs = [node.debugName()]
            graph_helper.append(node_pyio)

        graph_helper._populate_namespace_from_OP_to_IO()
        graph_helper._infer_undefined_tensor_size_when_possible(original_model)

        # graph_helper.check_is_valid()
        import ipdb

        ipdb.set_trace()

        return graph_helper

    @classmethod
    def parse_model(cls, original_model, args, omit_useless_nodes=True):
        """This method parses an optimized PyTorch model graph and produces
        a list of nodes and node stats for eventual conversion to NNEF format.
        """

        graph_helper = InternalPytorchGraphHelper()
        states = list(
            jit._trace._unique_state_dict(
                original_model, keep_vars=True
            ).values()
        )
        cls.parse_module(
            graph_helper,
            original_model,
            args,
            omit_useless_nodes=omit_useless_nodes,
        )
        import ipdb

        ipdb.set_trace()
        pass
