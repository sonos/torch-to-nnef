import logging
import re
import string
import typing as T
from collections import defaultdict
from collections.abc import MutableMapping

from torch_to_nnef.console import Console
from torch_to_nnef.exceptions import (
    NotFoundModuleExtractor,
    TorchCheckError,
    TorchNotFoundOp,
    TorchOpTranslatedDifferently,
    TorchToNNEFNotImplementedError,
)
from torch_to_nnef.op.custom_extractors import ModuleInfoExtractor
from torch_to_nnef.torch_graph.ir_data import (
    BlobTorchScriptObject,
    Data,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
    TtupleOrVar,
    TupleTensors,
)
from torch_to_nnef.torch_graph.ir_helpers import (
    _add_prefix_if_start_with_digit,
    _expand_containers_if_exists,
    _find_common_root,
    _find_data_node,
    _is_container,
    _parse_traced_name,
    _replacement_to_relative_module_path,
)
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.ir_op import TorchOp
from torch_to_nnef.torch_graph.torch_const import CLASSTYPE_KIND, GETATTR_KIND
from torch_to_nnef.utils import NamedItemOrderedSet

LOGGER = logging.getLogger(__name__)


def module_tracer_into_ir_graph(
    module_tracer,
    inputs: T.Optional[T.List[Data]] = None,
    outputs: T.Optional[T.List[TtupleOrVar]] = None,
    renaming_scheme: str = "numeric",
    **kwargs,
):
    ir_graph = TorchModuleIRGraph(torch_module_tracer=module_tracer, **kwargs)
    ir_graph.parse(
        provided_inputs=inputs,
        provided_outputs=outputs,
        renaming_scheme=renaming_scheme,
    )
    return ir_graph


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
        torch_module_tracer: TorchModuleTracer,
        omit_useless_nodes: bool = True,
        is_root_module: bool = False,
    ):
        self.op_nodes: T.List[TorchOp] = []
        self.inputs: T.List[Data] = []
        self.outputs: T.List[TtupleOrVar] = []

        self._data_nodes: NamedItemOrderedSet = NamedItemOrderedSet()
        self._omit_useless_nodes = omit_useless_nodes
        self.provided_inputs_picked_indexes: T.List[int] = []
        self._tracer = torch_module_tracer
        self._is_root_module = is_root_module

    @property
    def tracer(self):
        return self._tracer

    @property
    def data_nodes(self):
        return self._data_nodes

    @data_nodes.setter
    def data_nodes(self, other):
        self._data_nodes = (
            other
            if isinstance(other, NamedItemOrderedSet)
            else NamedItemOrderedSet.from_list(other)
        )

    def _check_container_items_rely_on_data_nodes(self):
        """container items reference must exists in `data_nodes`"""
        for dnode in self.data_nodes:
            if _is_container(dnode):
                for subdnode in dnode.data:
                    assert self.data_nodes.contains(
                        subdnode, strict=True
                    ), f"not referenced correctly sub item: {subdnode}"

    def _check_io_rely_on_data_nodes(self):
        """`inputs` or `outputs` reference items must exists in `data_nodes`"""
        for inode in self.inputs:
            if not self.data_nodes.contains(inode, strict=True):
                raise TorchCheckError(
                    f"not referenced correctly input: {inode}"
                )

        for onode in self.outputs:
            if not self.data_nodes.contains(onode, strict=True):
                raise TorchCheckError(
                    f"not referenced correctly output: {onode}"
                )

    def find_node(self, node_name: str) -> T.Optional[Data]:
        return self.data_nodes.get_by_name(node_name)

    def remap_node(self, from_node, to_node):
        """remap a data_node to another."""
        assert isinstance(from_node, Data)
        assert isinstance(to_node, Data)
        self.inputs = [to_node if _ is from_node else _ for _ in self.inputs]
        self.outputs = [to_node if _ is from_node else _ for _ in self.outputs]
        for op in self.op_nodes:
            op.inputs = [to_node if _ is from_node else _ for _ in op.inputs]
            op.outputs = [to_node if _ is from_node else _ for _ in op.outputs]
        self.data_nodes.remove(from_node, raise_exception_if_not_found=False)

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
        if not self.data_nodes.contains(to_node):
            self.data_nodes.append(to_node)  # this is very slow...

    def _parse_inputs(
        self, provided_inputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph inputs"""
        graph_inputs = list(self._tracer.torch_graph.inputs())[1:]
        if provided_inputs is None:
            provided_inputs = [None] * len(graph_inputs)  # type: ignore

        assert len(graph_inputs) == len(provided_inputs)

        for idx, (node_c_value, original_input) in enumerate(
            zip(graph_inputs, provided_inputs)
        ):
            if self._omit_useless_nodes:
                if (
                    len(node_c_value.uses()) == 0
                ):  # number of user of the node_c_value (= number of outputs/ fanout)
                    continue

            if node_c_value.type().kind() != CLASSTYPE_KIND:
                tv = TensorVariable.parse(node_c_value)
                if original_input is not None:
                    tv.shape = original_input.shape
                    tv.dtype = original_input.dtype
                    tv.quant = original_input.quant
                self.inputs.append(tv)
                self.data_nodes.append(tv)
                # used at _merge_subraph
                self.provided_inputs_picked_indexes.append(idx)

    def _parse_core(self):
        """Parse all Operations and collect the scope infos"""
        attr_to_scope: T.Dict[T.Any, str] = {}

        to_remap = []

        def maybe_gather_remap(exp_args):
            if len(exp_args) == 3 and exp_args[0] == "Int to be remaped":
                _, tensor_out, ancestor_node = exp_args
                to_remap.append((tensor_out, ancestor_node))

        for node in self._tracer.torch_graph.nodes():
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
                            self._tracer.mod,
                            node,
                            scope=attr_to_scope[attr_name],
                            data_nodes=self.data_nodes,
                            traced_module=self._tracer.traced_module,
                        )
                        self.op_nodes.append(op)
                    except TorchOpTranslatedDifferently as exp:
                        maybe_gather_remap(exp.args)

            else:
                try:
                    op = TorchOp.parse(
                        self._tracer.mod,
                        node,
                        scope="",
                        data_nodes=self.data_nodes,
                        traced_module=self._tracer.traced_module,
                    )
                    self.op_nodes.append(op)
                except TorchOpTranslatedDifferently as exp:
                    maybe_gather_remap(exp.args)

        # remap if needed
        for from_node, to_node in to_remap:
            self.remap_node(from_node, to_node)

    def _parse_outputs(
        self, provided_outputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph outputs"""
        torch_graph_outputs = self._tracer.torch_graph.outputs()
        outputs = [
            _find_data_node(self.data_nodes, _.debugName())
            for _ in torch_graph_outputs
        ]
        outputs = self._expand_fixed_tensor_list_in(outputs)

        if provided_outputs is not None:
            original_outputs = self._expand_fixed_tensor_list_in(
                provided_outputs
            )
            if len(outputs) != len(original_outputs):
                raise TorchCheckError(
                    f"{len(outputs)} == {len(original_outputs)}"
                )
            for original_output, output in zip(original_outputs, outputs):
                if _is_container(original_output) and _is_container(output):
                    # can be safely explored
                    continue
                if isinstance(output, TensorVariable):
                    output.shape = original_output.shape
                    output.dtype = original_output.dtype
                    output.quant = original_output.quant
                else:
                    raise TorchToNNEFNotImplementedError(
                        f"output={output}\ncompared to:\n"
                        f"original_output={original_output}"
                    )
        self.outputs = outputs

    def _update_scope_reference(self):
        """Update scope in op_nodes with additional infos"""
        alias_to_name = {}
        base_name = _parse_traced_name(self._tracer.traced_module)
        for name, module in self._tracer.traced_module.named_modules(
            prefix="__module"
        ):
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

        for node in _expand_containers_if_exists(self.data_nodes[:]):
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
                raise TorchToNNEFNotImplementedError(
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
            if datas_attr == "inputs":
                node_graph_to_wire = [
                    node_graph_to_wire[idx]
                    for idx in submodule_graph.provided_inputs_picked_indexes
                ]
            ref_nodes = list(
                _expand_containers_if_exists(
                    node_graph_to_wire, filter_container=True
                )
            )
            nodes = list(
                _expand_containers_if_exists(
                    node_subgraph_to_wire, filter_container=True
                )
            )
            for node, ref_node in zip(nodes, ref_nodes):
                submodule_graph.remap_node(from_node=node, to_node=ref_node)

        search_and_replace_data_nodes(
            submodule_graph.inputs, callmethod_node.inputs, "inputs"
        )
        search_and_replace_data_nodes(
            submodule_graph.outputs, callmethod_node.outputs, "outputs"
        )

        # }

        for _ in submodule_graph.op_nodes:
            res = _.scope.split(self.SEP, maxsplit=1)
            if len(res) >= 2 and isinstance(res, list):
                _.scope = f"{res[0]}[{prefix}]{self.SEP}{res[1]}"
            else:
                _.scope = f"{res}[{prefix}]"
            _.module_path = f"{module_prefix}.{_.module_path}"

        for _ in submodule_graph.data_nodes[:]:
            _.name = f"{prefix}.{_.name}"

        self.op_nodes = [op for op in self.op_nodes if op != callmethod_node]
        self.op_nodes += submodule_graph.op_nodes
        self.data_nodes += [
            dn
            for dn in submodule_graph.data_nodes
            if not self.data_nodes.contains(dn, strict=True)
        ]

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
                assert isinstance(op.op_ref, TorchModuleTracer), op.op_ref
                op.op_ref.args = op.args
                submodule_graph = module_tracer_into_ir_graph(
                    op.op_ref,
                    omit_useless_nodes=self._omit_useless_nodes,
                    inputs=op.inputs,
                    outputs=op.outputs,
                    renaming_scheme=renaming_scheme,
                )
                prefix = ""
                if cname is not None:
                    prefix = _add_prefix_if_start_with_digit(cname, "s")
                if ref_count[cname] == 1:
                    prefix += "_"
                elif ref_count[cname] == 2:
                    prefix += "_2nd_call"
                elif ref_count[cname] == 3:
                    prefix += "_3rd_call"
                else:
                    prefix += f"_{ref_count[cname]}th_call"
                self._merge_subraph(
                    submodule_graph,
                    prefix=prefix,
                    module_prefix=op.module_path,
                    callmethod_node=op,
                )

    def _rename_compact_numeric(self) -> None:
        count_ref: T.Dict[str, int] = defaultdict(int)
        mapping: T.Dict[str, str] = {}
        prefix_map = {
            TensorVariable: "v",
            PythonConstant: "c",
            BlobTorchScriptObject: "b",
            FixedTensorList: "l",
            TupleTensors: "tt",
            Data: "d",  # not used, avoid static analysis complain
        }
        for dnode in self.data_nodes[:]:
            prefix = prefix_map[dnode.__class__]
            if dnode.name in mapping:
                dnode.name = mapping[dnode.name]
                continue
            suffix = count_ref[prefix]
            count_ref[prefix] += 1
            mapping[dnode.name] = prefix + str(suffix)
            dnode.name = mapping[dnode.name]

    def _rename_natural_verbose(self) -> None:
        for dn in self.data_nodes[:]:
            dn.name = dn.name.split("/")[-1]
            if all(c in string.digits for c in dn.name):
                try:
                    self.find_data_node_producer(dn)
                except TorchNotFoundOp as exp:
                    if isinstance(dn, TensorVariable):
                        dn.name = f"v{dn.name}"
                    elif isinstance(dn, PythonConstant):
                        dn.name = f"c{dn.name}"
                    elif isinstance(dn, FixedTensorList):
                        dn.name = f"l{dn.name}"
                    else:
                        raise NotImplementedError(dn) from exp

            if dn.export_name[-1] in string.digits:
                try:
                    producer = self.find_data_node_producer(dn)
                    kind = producer.kind.split("::")[-1]
                    replace_data_node_name_with_suffix_auto_inc(
                        self, dn, kind, suffix_only_on_underscore=True
                    )
                except TorchNotFoundOp:
                    replace_data_node_name_with_suffix_auto_inc(
                        self, dn, suffix=""
                    )
        if self._is_root_module:
            remove_useless_digits_from_module_names(self)

    def apply_renaming_scheme(self, scheme="natural_verbose"):
        """Rename availlable data node following a scheme

        by default the natural_verbose pattern built is as close as possible
        to Pytorch graph context info. This pattern might come as too verbose.

        we propose a more concise numeric pattern that allow easier debug
        when looking at NNEF export correctness.

        """
        if scheme == "raw":
            return None
        if scheme == "natural_verbose":
            return self._rename_natural_verbose()
        if scheme == "numeric":
            return self._rename_compact_numeric()

        raise TorchToNNEFNotImplementedError(f"renaming scheme: {scheme}")

    def _filter_tuple_tensor_from_data_nodes(self):
        for dnode in self.data_nodes[:]:
            if isinstance(dnode, TupleTensors):
                self.data_nodes.remove(dnode)

    def _expand_tuple_in(self, iterable):
        expanded_data_nodes = []
        for dnode in iterable:
            if isinstance(dnode, TupleTensors):
                for sdnode in dnode.data:
                    expanded_data_nodes.append(sdnode)
            else:
                expanded_data_nodes.append(dnode)
        return expanded_data_nodes

    def _expand_fixed_tensor_list_in(self, iterable):
        expanded_data_nodes = []
        for dnode in iterable:
            if isinstance(dnode, FixedTensorList):
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
        assert isinstance(self.data_nodes, NamedItemOrderedSet)
        used_data_nodes = set(self.outputs)
        # Ensure we do not dish Module inputs
        used_data_nodes.update(self.inputs)

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
        self.data_nodes = NamedItemOrderedSet(
            sorted(
                list(used_data_nodes),
                key=lambda _: ordered_data_nodes_hashs.index(hash(_))
                if _ in ordered_data_nodes_hashs
                else -1,
            )
        )

    def _cleanup_unused_nodes_in_graph(self):
        pass

    def parse(
        self,
        renaming_scheme: str = "numeric",
        provided_inputs=None,
        provided_outputs=None,
    ):
        try:
            extractor = ModuleInfoExtractor.get_by_module(self._tracer.mod)
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

        # Cleanup all unused nodes nodes in graph
        self._cleanup_unused_nodes_in_graph()

        if renaming_scheme:
            self.apply_renaming_scheme(renaming_scheme)
        return self

    def find_data_node_producer(self, data_node: Data) -> TorchOp:
        assert isinstance(data_node, Data), data_node
        for op in self.op_nodes:
            for op_out_dnode in _expand_containers_if_exists(op.outputs):
                if op_out_dnode is data_node:
                    return op
        raise TorchNotFoundOp("Did not find operation node")

    def find_ops_nodes_by_input_node(self, data_node: Data) -> T.List[TorchOp]:
        assert isinstance(data_node, Data), data_node
        collected_ops = []
        for op in self.op_nodes:
            for op_out_dnode in _expand_containers_if_exists(op.inputs):
                if op_out_dnode is data_node:
                    collected_ops.append(op)
        if not collected_ops:
            raise TorchNotFoundOp("Did not find operation node")
        return collected_ops

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
                    f"[type]{o.dtype if hasattr(o, 'dtype') else type(o.data)}"
                    f"[/type] [var]{o.export_name}[/var]"
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


def replace_last_number(
    name: str,
    suffix: str,
    new_idx: int,
    suffix_only_on_underscore: bool = False,
):
    idx = -1
    while name[idx] in string.digits:
        idx -= 1
        if abs(idx) > len(name):
            assert len(suffix) > 0
            return f"{suffix}{new_idx}"

    if idx == -1:
        trunced_name = name
    else:
        trunced_name = name[: idx + 1]
    if suffix and trunced_name.endswith(suffix):
        trunced_name = trunced_name[: -len(suffix)]
    if suffix and trunced_name[:-1].endswith(suffix):
        trunced_name = trunced_name[: -len(suffix) - 1]
    if (
        suffix_only_on_underscore
        and len(trunced_name)
        and trunced_name[-1] != "_"
        and (len(trunced_name) < 2 or trunced_name[-2] != "_")
    ):
        suffix = ""
    return f"{trunced_name}{suffix}{new_idx}"


def replace_data_node_name_with_suffix_auto_inc(
    torch_mod_ir_graph: TorchModuleIRGraph,
    dn: Data,
    suffix="",
    suffix_only_on_underscore: bool = False,
):
    idx = 0
    new_name = replace_last_number(
        dn.name, suffix, idx, suffix_only_on_underscore
    )
    if dn.name == new_name:
        return
    while True:
        colliding_dn = torch_mod_ir_graph.find_node(new_name)
        if colliding_dn is None:
            break
        idx += 1
        new_name = replace_last_number(
            dn.name, suffix, idx, suffix_only_on_underscore
        )

    dn.name = new_name


def _flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    items: T.List[T.Tuple[str, T.Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def remove_useless_digits_from_module_names(
    torch_mod_ir_graph: TorchModuleIRGraph,
):
    module_separator = "_."
    # pylint: disable-next=protected-access
    data_node_names = list(torch_mod_ir_graph.data_nodes._map)
    assert len(data_node_names) == len(set(data_node_names))
    name_tree: T.Dict[str, T.Any] = {}
    for data_node_name in data_node_names:
        current_sub_tree = name_tree
        chunks = data_node_name.split(module_separator)
        for idx, c in enumerate(chunks):
            if c not in current_sub_tree:
                current_sub_tree[c] = (
                    data_node_name if len(chunks) - 1 == idx else {}
                )
            current_sub_tree = current_sub_tree[c]

    to_explore = [name_tree]
    while len(to_explore) > 0:
        current_sub_tree = to_explore.pop()
        keys = list(current_sub_tree.keys())
        stacked_keys = defaultdict(list)
        for key in keys:
            stacked_keys[re.sub(r"(\.[0-9]+)?$", "", key)].append(key)
        for simplified_key, original_keys in stacked_keys.items():
            if len(original_keys) == 1 and original_keys[0] != simplified_key:
                current_sub_tree[simplified_key] = current_sub_tree[
                    original_keys[0]
                ]
                del current_sub_tree[original_keys[0]]
        for next_sub_tree in current_sub_tree.values():
            if isinstance(next_sub_tree, dict):
                to_explore.append(next_sub_tree)
    remapping_table = _flatten_dict(name_tree, sep=module_separator)
    for new_name, original in remapping_table.items():
        if new_name == original:
            continue
        orignal_data_node = torch_mod_ir_graph.data_nodes.get_by_name(original)
        orignal_data_node.name = new_name
