import logging
import typing as T
from collections import defaultdict

import torch

from torch_to_nnef.console import Console
from torch_to_nnef.dtypes import dtype_is_whole_number
from torch_to_nnef.exceptions import (
    T2NError,
    T2NErrorDataNodeValue,
    T2NErrorNotFoundModuleExtractor,
    T2NErrorNotImplemented,
    T2NErrorTorchCheck,
    T2NErrorTorchNotFoundOp,
    T2NErrorTorchOpTranslatedDifferently,
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
    _expand_node_containers_if_exists,
    _find_common_root,
    _find_data_node,
    _parse_traced_name,
    _replacement_to_relative_module_path,
)
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.ir_naming import (
    DEFAULT_VARNAME_SCHEME,
    VariableNamingScheme,
    apply_nnef_variable_naming_scheme,
    rename_variable_by_incr,
)
from torch_to_nnef.torch_graph.ir_op import TorchOp
from torch_to_nnef.torch_graph.torch_const import (
    CALL_KIND,
    CLASSTYPE_KIND,
    GETATTR_KIND,
)
from torch_to_nnef.utils import ReactiveNamedItemDict

LOGGER = logging.getLogger(__name__)


def module_tracer_into_ir_graph(
    module_tracer,
    inputs: T.Optional[T.List[Data]] = None,
    outputs: T.Optional[T.List[TtupleOrVar]] = None,
    forced_inputs_names: T.Optional[T.List[str]] = None,
    forced_outputs_names: T.Optional[T.List[str]] = None,
    nnef_variable_naming_scheme: VariableNamingScheme = DEFAULT_VARNAME_SCHEME,
    **kwargs,
):
    ir_graph = TorchModuleIRGraph(torch_module_tracer=module_tracer, **kwargs)
    ir_graph.parse(
        provided_inputs=inputs,
        provided_outputs=outputs,
        forced_inputs_names=forced_inputs_names,
        forced_outputs_names=forced_outputs_names,
        nnef_variable_naming_scheme=nnef_variable_naming_scheme,
    )
    return ir_graph


class TorchModuleIRGraph:
    """Torch Graph intermediate representation from: jit.trace with recursion.

    This is not direct torch._C.Graph but simpler abstraction, with:

    A list of data nodes in `self.data_nodes`
    A list of operations nodes in `self.op_nodes`

    `self.inputs` is a list of reference of some `self.data_nodes`
    `self.outputs` is a list of reference of some `self.data_nodes`

    This abstraction of the vanilla Torch Graph allow to manipulate graph
    in order to check/complete missing data informations and ignore
    useless operations for our transcription needs.

    It's also allows to be less reliant on base graph in case
    of modification of PyTorch Internals (think Adapter Pattern).

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

        self._data_nodes: ReactiveNamedItemDict = ReactiveNamedItemDict()
        self._omit_useless_nodes = omit_useless_nodes
        self.provided_inputs_picked_indexes: T.List[int] = []
        self._tracer = torch_module_tracer
        self.is_root_module = is_root_module

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
            if isinstance(other, ReactiveNamedItemDict)
            else ReactiveNamedItemDict.from_list(other)
        )

    def _check_container_items_rely_on_data_nodes(self):
        """Container items reference must exists in `data_nodes`."""
        for dnode in self.data_nodes:
            if dnode.is_container:
                for subdnode in dnode.iter():
                    assert self.data_nodes.contains(subdnode, strict=True), (
                        f"not referenced correctly sub item: {subdnode}"
                    )

    def _check_io_rely_on_data_nodes(self):
        """`inputs` or `outputs` reference items must exists in `data_nodes`."""
        for inode in self.inputs:
            if not self.data_nodes.contains(inode, strict=True):
                raise T2NErrorTorchCheck(
                    f"not referenced correctly input: {inode}"
                )

        for onode in self.outputs:
            if not self.data_nodes.contains(onode, strict=True):
                raise T2NErrorTorchCheck(
                    f"not referenced correctly output: {onode}"
                )

    def find_node(self, node_name: str) -> T.Optional[Data]:
        return self.data_nodes.get_by_name(node_name)

    def remap_node(self, from_node, to_node):
        """Remap a data_node to another."""
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
            if dnode.is_container:
                new_data = []
                for subdnode in dnode.iter():
                    value = to_node if subdnode is from_node else subdnode
                    new_data.append(value)
                dnode.set_data(new_data)

        # add if not exists in graph
        if not self.data_nodes.contains(to_node):
            self.data_nodes.append(to_node)  # this is very slow...

    def _parse_inputs(
        self, provided_inputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph inputs."""
        graph_inputs = []
        is_start_cls = True
        for torch_ir_inp in self._tracer.torch_graph.inputs():
            if torch_ir_inp.type().kind() == CLASSTYPE_KIND and is_start_cls:
                continue
            is_start_cls = False
            graph_inputs.append(torch_ir_inp)

        if provided_inputs is None:
            provided_inputs = [None] * len(graph_inputs)  # type: ignore

        assert len(graph_inputs) == len(provided_inputs)

        for idx, (node_c_value, original_input, arg) in enumerate(
            zip(graph_inputs, provided_inputs, self._tracer.args)
        ):
            if self._omit_useless_nodes and len(node_c_value.uses()) == 0:
                # number of user of the node_c_value
                # (= number of outputs/ fanout)
                continue

            if node_c_value.type().kind() != CLASSTYPE_KIND:
                tv = TensorVariable.parse(node_c_value)
                if original_input is None:
                    if isinstance(arg, torch.Tensor):
                        tv.shape = list(arg.shape)
                        tv.dtype = arg.dtype
                        if dtype_is_whole_number(arg.dtype):
                            tv._traced_data = arg
                    else:
                        raise T2NErrorNotImplemented(type(arg))
                else:
                    tv.shape = original_input.shape
                    tv.dtype = original_input.dtype
                    tv.quant = original_input.quant
                    if (
                        original_input._traced_data is None
                        and dtype_is_whole_number(arg.dtype)
                    ):
                        tv._traced_data = arg
                    else:
                        tv._traced_data = original_input._traced_data
                self.inputs.append(tv)
                self.data_nodes.append(tv)
                # used at _merge_subraph
                self.provided_inputs_picked_indexes.append(idx)

    def _parse_core(self):
        """Parse all Operations and collect the scope infos."""
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
                    attr_to_scope[attr_name] = (
                        f"{parent_scope}/{attr_scope}.{attr_name}"
                    )
                else:
                    attr_to_scope[attr_name] = f"__module.{attr_name}"
                # We don't need classtype nodes; scope will provide
                # this information
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
                    except T2NErrorTorchOpTranslatedDifferently as exp:
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
                except T2NErrorTorchOpTranslatedDifferently as exp:
                    maybe_gather_remap(exp.args)

        # remap if needed
        for from_node, to_node in to_remap:
            self.remap_node(from_node, to_node)

    def _parse_outputs(
        self, provided_outputs: T.Optional[T.List[TensorVariable]] = None
    ):
        """Parse traced graph outputs."""
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
                raise T2NErrorTorchCheck(
                    f"{len(outputs)} == {len(original_outputs)}"
                )
            for original_output, output in zip(original_outputs, outputs):
                if original_output.is_container and output.is_container:
                    # can be safely explored
                    continue
                if isinstance(output, TensorVariable):
                    output.shape = original_output.shape
                    output.dtype = original_output.dtype
                    output.quant = original_output.quant
                else:
                    raise T2NErrorNotImplemented(
                        f"output={output}\ncompared to:\n"
                        f"original_output={original_output}"
                    )
        self.outputs = outputs

    def _update_scope_reference(self):
        """Update scope in op_nodes with additional infos."""
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
            for input_node in _expand_node_containers_if_exists(node.inputs):
                unique_name_to_scoped_name[input_node.name] = (
                    node.scope + self.SEP + input_node.name
                )

        for node in _expand_node_containers_if_exists(self.data_nodes[:]):
            if not node.name.startswith(selected_scope_name + self.SEP):
                node.name = selected_scope_name + self.SEP + node.name

        for node in _expand_node_containers_if_exists(self.inputs):
            if not node.name.startswith(selected_scope_name + self.SEP):
                node.name = selected_scope_name + self.SEP + node.name

    def _infer_missing_shapes_from_ops_outputs(self, raise_error: bool = False):
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
                    for _ in _expand_node_containers_if_exists(op_node.outputs):
                        if _.name in unshaped_data:
                            del unshaped_data[_.name]
            remaining_ops = [op for op in remaining_ops if op not in ops_to_rm]
            end_len = len(unshaped_data)
            if start_len == end_len:
                msg = f"missing unshaped_data: {unshaped_data}"
                if raise_error:
                    raise T2NErrorNotImplemented(msg)
                LOGGER.debug(msg)
                break

    def _merge_subraph(  # noqa: MC0001
        self, submodule_graph, callmethod_node, prefix: str, module_prefix: str
    ):
        # Re-Wire input and output naming => {
        def search_and_replace_data_nodes(
            node_subgraph_to_wire: T.List[Data],
            node_graph_to_wire: T.List[Data],
            datas_attr: str,
        ):
            assert datas_attr in ["inputs", "outputs"]
            if datas_attr == "inputs":
                node_graph_to_wire = [
                    node_graph_to_wire[idx]
                    for idx in submodule_graph.provided_inputs_picked_indexes
                ]
            ref_nodes = list(
                _expand_node_containers_if_exists(
                    node_graph_to_wire, filter_container=True
                )
            )
            subgraph_nodes = list(
                _expand_node_containers_if_exists(
                    node_subgraph_to_wire, filter_container=True
                )
            )
            if datas_attr == "inputs":
                for snode, ref_node in zip(subgraph_nodes, ref_nodes):
                    submodule_graph.remap_node(
                        from_node=snode, to_node=ref_node
                    )
            elif datas_attr == "outputs":
                for snode, ref_node in zip(subgraph_nodes, ref_nodes):
                    self.remap_node(from_node=ref_node, to_node=snode)

        search_and_replace_data_nodes(
            submodule_graph.inputs, callmethod_node.inputs, "inputs"
        )
        # }

        for _ in submodule_graph.op_nodes:
            res = _.scope.split(self.SEP, maxsplit=1)
            if len(res) >= 2 and isinstance(res, list):
                _.scope = f"{res[0]}[{prefix}]{self.SEP}{res[1]}"
            else:
                _.scope = f"{res}[{prefix}]"
            _.module_path = f"{module_prefix}.{_.module_path}"

        protected_from_rename_node = set(submodule_graph.inputs)
        for dn in submodule_graph.data_nodes[:]:
            if dn in protected_from_rename_node:
                continue
            dn.name = f"{prefix}.{dn.name}"

        for dn in submodule_graph.data_nodes[:]:
            if not self.data_nodes.contains(dn, strict=True):
                # A failure mode is not fully understood here:
                # in some edge-cases (only observed in CI)
                # there is a collision that is detected between
                # names and append do not work
                # (which is supposed to not happen thanks to
                # `rename_variable_by_incr`)
                # hence this retry logic
                for start_index in range(1, 4):  # give 3 try
                    new_name = dn.name
                    if self.data_nodes.get_by_name(dn.name):
                        new_name = rename_variable_by_incr(
                            dn.name,
                            [self.data_nodes, submodule_graph.data_nodes],
                            start_index=start_index,
                        )
                        LOGGER.info(
                            "potential name collision detected rename"
                            "new '%s' into '%s'",
                            dn.name,
                            new_name,
                        )
                        dn.name = new_name
                    try:
                        self.data_nodes.append(dn)
                        break
                    except T2NErrorDataNodeValue as exp:
                        LOGGER.warning(
                            "tried to append '%s' in data_nodes but failed: %s",
                            new_name,
                            exp,
                        )
        search_and_replace_data_nodes(
            submodule_graph.outputs, callmethod_node.outputs, "outputs"
        )
        self.op_nodes = [op for op in self.op_nodes if op != callmethod_node]
        self.op_nodes += submodule_graph.op_nodes

    def _recursive_call_method(
        self, nnef_variable_naming_scheme: VariableNamingScheme
    ):
        """In case prim::CallMethod is encountered it tries to trace it.

        It does this by recursive call to parse_module on linked submodule.

        Some part of the submodule may not be serializable to JIT
        this is for this very API limitation that we do not use directly
        the method torch.jit._get_trace_graph that is used in
        ONNX builtin pytorch serialization and instead build on recursive
        jit.parse.

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

                if set(self.tracer.torch_graph.outputs()) == set(
                    op.op_ref.torch_graph.outputs()
                ):
                    raise T2NError(
                        "Bug: Recursive call detected ! "
                        "Trying to parse same Pytorch IR sub-module twice: "
                        f"{op}"
                    )
                submodule_graph = module_tracer_into_ir_graph(
                    op.op_ref,
                    omit_useless_nodes=self._omit_useless_nodes,
                    inputs=op.inputs,
                    outputs=op.outputs,
                    nnef_variable_naming_scheme=nnef_variable_naming_scheme,
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
                self._infer_missing_shapes_from_ops_outputs()

    def _filter_tuple_tensor_from_data_nodes(self):
        for dnode in self.data_nodes[:]:  # pylint: disable=not-an-iterable
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
        """Remove all references to tuple by using only unpacked variables."""
        self._filter_tuple_tensor_from_data_nodes()
        self.inputs = self._expand_tuple_in(self.inputs)
        self.outputs = self._expand_tuple_in(self.outputs)
        for op in self.op_nodes:
            op.inputs = self._expand_tuple_in(op.inputs)
            op.outputs = self._expand_tuple_in(op.outputs)

    def _filter_nodes_not_in_trace_between_inputs_and_outputs(self):
        """Remove all unused graph nodes.

        Backward propagation from graph output to input to select kept nodes

        """
        assert isinstance(self.data_nodes, ReactiveNamedItemDict)
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
                    if used_data_node.is_container:
                        additional_data_node_from_list.update(
                            used_data_node.iter()
                        )
                remaining_data_nodes.difference_update(
                    additional_data_node_from_list
                )
                used_data_nodes.update(additional_data_node_from_list)

        self.op_nodes = [op for op in self.op_nodes[:] if op in used_op_nodes]

        # filtered bug with original order

        ordered_data_nodes_hashs = {
            hash(_): idx for idx, _ in enumerate(self.data_nodes)
        }
        self.data_nodes = ReactiveNamedItemDict(
            sorted(
                list(used_data_nodes),
                key=lambda _: ordered_data_nodes_hashs[hash(_)]
                if _ in ordered_data_nodes_hashs
                else -1,
            )
        )

    def parse(
        self,
        nnef_variable_naming_scheme: VariableNamingScheme = DEFAULT_VARNAME_SCHEME,  # noqa: E501
        provided_inputs=None,
        provided_outputs=None,
        forced_inputs_names=None,
        forced_outputs_names=None,
    ):
        """Core parsing transforming nn.Module into torch_to_nnef IR."""
        LOGGER.debug(
            "start parse to IR: %s", self._tracer.mod.__class__.__name__
        )
        try:
            extractor = ModuleInfoExtractor.get_by_module(self._tracer.mod)
            extractor.generate_in_torch_graph(
                self, provided_inputs, provided_outputs
            )
            return self
        except T2NErrorNotFoundModuleExtractor:
            pass
        self._parse_inputs(provided_inputs)
        self._parse_core()
        self._parse_outputs(provided_outputs)
        self._update_scope_reference()
        self._update_data_node_name_with_base_context()
        self._infer_missing_shapes_from_ops_outputs()
        self._recursive_call_method(
            nnef_variable_naming_scheme=nnef_variable_naming_scheme
        )
        self._avoid_reference_to_tuples()
        self._filter_nodes_not_in_trace_between_inputs_and_outputs()

        self._check_container_items_rely_on_data_nodes()
        self._check_io_rely_on_data_nodes()

        self._cleanup_dangling_data_node_hooks()

        if self.is_root_module:
            if forced_inputs_names:
                for inode, new_name in zip(self.inputs, forced_inputs_names):
                    inode.name = new_name
                    assert inode.name == inode.export_name
            if forced_outputs_names:
                for onode, new_name in zip(self.outputs, forced_outputs_names):
                    onode.name = new_name
                    assert onode.name == onode.export_name
            # need to repeat the if's:
            # in case of input paramater directly in outputs
            # (ie. torchaudio.Conformer)
            if forced_inputs_names:
                self.data_nodes.protect_item_names(forced_inputs_names)
            if forced_outputs_names:
                self.data_nodes.protect_item_names(forced_outputs_names)
        elif forced_inputs_names or forced_outputs_names:
            raise T2NErrorNotImplemented(
                "forced names are only for root module"
            )

        if nnef_variable_naming_scheme:
            apply_nnef_variable_naming_scheme(self, nnef_variable_naming_scheme)

        LOGGER.debug("parsed to IR: %s", self._tracer.mod.__class__.__name__)
        return self

    def _cleanup_dangling_data_node_hooks(self):
        for dn in self.data_nodes:
            # pylint: disable-next=protected-access
            dn._name_hooks = {self.data_nodes._change_name_hook}

    def find_data_node_producer(self, data_node: Data) -> TorchOp:
        assert isinstance(data_node, Data), data_node
        for op in self.op_nodes:
            for op_out_dnode in _expand_node_containers_if_exists(op.outputs):
                if op_out_dnode is data_node:
                    return op
        raise T2NErrorTorchNotFoundOp("Did not find operation node")

    def printall(self):
        """Display Helper Graph infos in stdout of your tty."""
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
            + f"[PyTorch JIT Graph '{self.tracer.mod.__class__}']"
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
                    "\t\t[type]List[/type] "
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
            if _.kind == CALL_KIND:
                mod_name = _.op_ref.mod.__class__.__name__
                mod_fn = _.op_ref.fn_name
                inputs_str = f"<{mod_name}.{mod_fn}>{inputs_str}"
            cls_name = ""
            outputs_str = ", ".join(
                [
                    f"[type]{o.dtype if hasattr(o, 'dtype') else type(o.data)}"
                    f"[/type] [var]{o.export_name}[/var]"
                    for o in _.outputs
                ]
            )
            cprint(
                f"\t\t {outputs_str} := "
                f"[kind]{_.kind}[/kind]{inputs_str}{cls_name}"
            )

        outputs_str = ", ".join(_.slug for _ in self.outputs)
        cprint("")
        cprint(f"outputs: ({outputs_str})")
        cprint("[type]" + "_" * 100 + "[/type]")
