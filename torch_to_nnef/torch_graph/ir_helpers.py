import contextlib
import importlib
import logging
import typing as T

import torch
from torch import jit, nn

from torch_to_nnef.dtypes import str_to_torch_dtype
from torch_to_nnef.exceptions import (
    T2NError,
    T2NErrorNotImplemented,
    T2NErrorTorchNotFoundDataNode,
    T2NErrorTorchOpTranslatedDifferently,
)
from torch_to_nnef.tensor.opaque import IR_OPAQUE_NAME, find_opaque_ref_by_py_id
from torch_to_nnef.torch_graph.ir_data import (
    BlobTorchScriptObject,
    Data,
    DictTensors,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
    TupleTensors,
    cleanup_data_name,
)
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_ARANGE,
    ATEN_CONTIGUOUS_KIND,
    ATEN_INT,
    ATEN_RELU,
    ATEN_SILU,
    ATEN_STARTID,
    CALL_KIND,
    CONSTANT_KIND,
    DICTCONSTRUCT_KIND,
    GETATTR_KIND,
    LISTCONSTRUCT_KIND,
    LISTTYPE_KIND,
    LISTUNPACK_KIND,
    MAP_TO_NOP,
    MODULE_PATH_ATEN,
    MODULE_PATH_QUANTIZED,
    NUMTOTENSOR_KIND,
    PRIM_STARTID,
    TUPLECONSTRUCT_KIND,
    TUPLEUNPACK_KIND,
)
from torch_to_nnef.utils import ReactiveNamedItemDict

LOGGER = logging.getLogger(__name__)


def nop(x, *args, **kwargs):
    return x


def _node_get(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


# pylint: disable-next=protected-access
torch._C.Node.__getitem__ = _node_get


def aten_name_to_torch_fn(
    aten_name: str,
):
    """Get aten cpp torch operator raw python binding."""
    name = aten_name.replace(ATEN_STARTID, "")
    return getattr(torch.ops.aten, name)


def quantized_name_to_torch_fn(aten_name):
    name = aten_name.replace("quantized::", "")
    return getattr(torch.ops.quantized, name)


def _add_prefix_if_start_with_digit(text: str, prefix: str) -> str:
    """Ensure we do not start with integer a text."""
    _prefix = ""
    if text[0] in "0123456789":
        _prefix = prefix
    return _prefix + text


def _parse_traced_name(module):
    if isinstance(module, jit.TracedModule):
        # pylint: disable-next=protected-access
        module_name = module._name
    else:
        module_name = getattr(module, "original_name", "Module")
    return module_name


def _expand_node_containers_if_exists(
    data_items, filter_container: bool = False
):
    for data_item in data_items:
        if hasattr(data_item, "is_container") and data_item.is_container:
            yield from data_item.iter()
            if filter_container:
                continue
        yield data_item


def _expand_containers_if_exists(data_items, filter_container: bool = False):
    for data_item in data_items:
        if isinstance(data_item, (tuple, list)):
            yield from iter(data_item)
            if filter_container:
                continue
        yield data_item


def _unfold_graph_getattr_by_node(
    module: T.Union[nn.Module, torch.jit.TracedModule],
    getattr_node: torch._C.Node,
) -> T.Tuple[str, T.Union[nn.Module, torch.jit.TracedModule]]:
    """Unfold  nn.Module python code reference to sub...sub modules in graph."""
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
    """Reconstruct shapes of whished view.

    For example:
        x_reshape = x.contiguous().view((-1,) + x.shape[2:])

        Provide incomplete graph information about shapes filing info with None
        as such:

        view((-1, None, None))

        but we know input for example:

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


def dynamic_tensor_list_parse(node_c_value: torch._C.Value):
    """Hold outputs of aten::chunk and other pytorch graph Tensor[] ."""
    node_type = node_c_value.type()
    assert node_type.kind() == LISTTYPE_KIND
    LOGGER.debug(
        "ListType can be of arbitrary length "
        "but we can not handle this dynamism at inference "
        " so 'split and other ops' will generate array "
        "of tensor with fixed size"
    )
    used_in = node_c_value.uses()
    if len(used_in) != 1:
        raise T2NErrorNotImplemented()
    use_op = used_in[0].user
    if use_op.kind() != LISTUNPACK_KIND:
        raise T2NErrorNotImplemented()

    return FixedTensorList(
        name=node_c_value.debugName(),
        data=[TensorVariable.parse(_) for _ in use_op.outputs()],
    )


def _find_data_node(data_nodes: ReactiveNamedItemDict, name: str):
    data_node = data_nodes.get_by_name(cleanup_data_name(name))
    if data_node is None:
        raise T2NErrorTorchNotFoundDataNode(
            f"'{name}' not found in {data_nodes}"
        )
    return data_node


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
    # pylint: disable-next=protected-access
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
        raise T2NErrorNotImplemented(dtype)
    data_nodes.append(PythonConstant(name=name, data=data))
    return data_nodes[-1]


def _fetch_backward(data_nodes, c_node: torch._C.Node):
    """Backward search of final resolution argument from list_construct."""
    if c_node.kind() in [ATEN_INT, NUMTOTENSOR_KIND]:
        return _fetch_backward(data_nodes, c_node.input().node())
    try:
        return _find_data_node(data_nodes, c_node.output().debugName())
    except T2NErrorTorchNotFoundDataNode as exp:
        raise T2NErrorNotImplemented("_fetch_backward c_node:", c_node) from exp


def _parse_list_construct_values(node, data_nodes: ReactiveNamedItemDict):
    values = []
    contains_tensors = False
    for cvalue in node.inputs():
        if cvalue.node().kind() == CONSTANT_KIND:
            value = _parse_constant(
                cvalue.node(),
                [],  # data_nodes empty as added later
            )
            if isinstance(value, TensorVariable):
                contains_tensors = True
        else:
            contains_tensors = True
            if cvalue.node().kind() == ATEN_INT:
                value = _fetch_backward(data_nodes, cvalue.node())
            elif str(cvalue.type()) in ["Tensor", "int"]:
                try:
                    value = _find_data_node(data_nodes, cvalue.debugName())
                except T2NErrorTorchNotFoundDataNode:
                    value = TensorVariable.parse(cvalue)
            else:
                raise T2NErrorNotImplemented(
                    "parse list construct argument", cvalue
                )
        values.append(value)
    return contains_tensors, values


def _parse_list_construct(node, data_nodes: ReactiveNamedItemDict):
    # should build a Data
    contains_tensors, values = _parse_list_construct_values(node, data_nodes)

    if contains_tensors:
        tensor_values = []
        for value in values:
            if isinstance(value, (TensorVariable, PythonConstant)):
                if not data_nodes.get_by_name(value.name):
                    data_nodes.append(value)
            else:
                raise T2NErrorNotImplemented()
            tensor_values.append(value)

        data_nodes.append(
            FixedTensorList(name=node.output().debugName(), data=tensor_values)
        )
    else:
        data_nodes.append(
            PythonConstant(
                name=node.output().debugName(), data=[v.data for v in values]
            )
        )

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

        # skip non interesting for export
        abstracted_inputs = abstracted_inputs[:-3]

        # cast to torch dtype
        # abstracted_inputs[-1].set_data(SCALAR_TYPE_TO_PYTORCH_TYPE[
        # abstracted_inputs[-1].data
        # ])

        n_inputs = len(abstracted_inputs)
        if n_inputs < 2 or n_inputs > 4:
            raise T2NErrorNotImplemented(n_inputs, abstracted_inputs)
        if n_inputs == 2:
            dnode = PythonConstant(
                f"ar_{abstracted_inputs[0].name}_stub_start", data=0
            )
            data_nodes.append(dnode)
            abstracted_inputs.insert(0, dnode)
        if n_inputs != 4:
            dnode = PythonConstant(
                f"ar_{abstracted_inputs[0].name}_stub_step", data=1
            )
            data_nodes.append(dnode)
            abstracted_inputs.insert(2, dnode)

    return abstracted_inputs


def _aten_inputs_and_op_ref(kind, inputs, data_nodes):
    abstracted_inputs = _prepare_arguments(kind, inputs, data_nodes)
    op_ref = None
    with contextlib.suppress(AttributeError):
        op_ref = aten_name_to_torch_fn(kind)
    return op_ref, abstracted_inputs


# pylint: disable-next=too-many-branches
def _rerouted_parsing(
    node: torch._C.Node, data_nodes: ReactiveNamedItemDict, module
):
    """Specific torch kind operation are changed to improve T2N IR.

    If specific kind matched it raise T2NErrorTorchOpTranslatedDifferently
    meaning it is handled differently than vanilla torch graph

    """
    kind: str = node.kind()
    if kind.startswith("t2n::"):
        if kind == IR_OPAQUE_NAME:
            py_id = int(node.input().toIValue())
            opaque_tensor = find_opaque_ref_by_py_id(module, py_id)
            tv = TensorVariable(
                name=node.output().debugName(),
                shape=list(opaque_tensor.shape),
                dtype=opaque_tensor.dtype,
                data=opaque_tensor,
            )
            assert tv.shaped_and_typed
            data_nodes.append(tv)
            raise T2NErrorTorchOpTranslatedDifferently(
                "geattr handled as TensorVariable"
            )
        raise T2NErrorNotImplemented(
            f"node: {node} dispatch is not implemented"
        )
    if kind == GETATTR_KIND:
        _parse_getattr_tensor(node, module, data_nodes)
        raise T2NErrorTorchOpTranslatedDifferently(
            "geattr handled as TensorVariable"
        )
    if kind == CONSTANT_KIND:
        _parse_constant(node, data_nodes)
        raise T2NErrorTorchOpTranslatedDifferently(
            "constant become PythonConstant"
        )
    if kind == LISTCONSTRUCT_KIND:
        _parse_list_construct(node, data_nodes)
        raise T2NErrorTorchOpTranslatedDifferently(
            "List Construct handled as PythonConstant"
        )
    if kind == ATEN_INT:
        ancestor_node = _fetch_backward(data_nodes, node)
        if ancestor_node not in data_nodes:
            data_nodes.append(ancestor_node)
            raise T2NErrorTorchOpTranslatedDifferently("Int added tensor")
        # will be remaped to
        tensor_out = TensorVariable.parse(node.output())
        data_nodes.append(tensor_out)
        raise T2NErrorTorchOpTranslatedDifferently(
            "Int to be remaped", tensor_out, ancestor_node
        )

    if kind.startswith(PRIM_STARTID):
        if kind == TUPLEUNPACK_KIND:
            dnodes = _find_data_node(data_nodes, node.input().debugName()).data
            for dnode, o_node_c_value in zip(dnodes, node.outputs()):
                o_type = o_node_c_value.type()
                if o_type.kind() == "TensorType":
                    stype = o_type.scalarType()
                    shape = o_type.sizes()
                elif o_type.kind() == "IntType":
                    stype = "int"
                    shape = [1]
                else:
                    raise T2NErrorNotImplemented(o_type.kind())

                dtype = str_to_torch_dtype(stype) if stype else None
                dnode.name = o_node_c_value.debugName()
                dnode.shape = shape
                dnode.dtype = dtype
                dnode.set_data(o_node_c_value.toIValue())
            raise T2NErrorTorchOpTranslatedDifferently("Tuple unpacked")
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
            raise T2NErrorTorchOpTranslatedDifferently("Tuple Construct")
        if kind == NUMTOTENSOR_KIND:
            return
        if kind == LISTUNPACK_KIND:
            # note: maybe should be replace dataNode to a FixedTensorList
            raise T2NErrorTorchOpTranslatedDifferently("List unpacked")
        if kind == DICTCONSTRUCT_KIND:
            data_nodes.append(
                DictTensors.parse_from_dic_node_c_value(
                    node.output(), data_nodes
                )
            )
            raise T2NErrorTorchOpTranslatedDifferently("Dict Construct")
        if kind != CALL_KIND:
            raise T2NErrorNotImplemented(node)
    if kind == CALL_KIND and not any(
        bool(use) for onode in node.outputs() for use in onode.uses()
    ):
        raise T2NErrorTorchOpTranslatedDifferently(
            "This method outputs are not used anywhere in graph"
        )


def _extract_op_infos_call_kind(module, traced_module, node, inputs):
    value_call_ref = inputs[0]
    module_getter_ref, op_ref = _unfold_graph_getattr_by_node(
        module, value_call_ref.node()
    )
    qualname = value_call_ref.type().qualified_name()
    kind = CALL_KIND
    if qualname.startswith("__torch__.") and not module_getter_ref:
        # NOTE: this part of the code is
        # non generic: the problem arise from:
        # the call to a function that is passed as
        # input of the PyTorch IR 'graph'
        # in this case finding the Python object reference based on
        # PyTorch IR python API is hard.
        ref_cls_path = ".".join(
            [
                _
                for _ in qualname.replace("__torch__.", "").split(".")
                if "___torch_mangle_" not in _
            ]
        )
        ref_mod_path, ref_cls_name = ref_cls_path.rsplit(".", 1)
        ref_cls = getattr(importlib.import_module(ref_mod_path), ref_cls_name)
        module_getter_ref, op_ref = next(
            (name, mod)
            for name, mod in module.named_modules()
            if isinstance(mod, ref_cls)
        )
        inputs = inputs[1:]
        if isinstance(op_ref, torch.nn.ReLU):
            kind = ATEN_RELU
        elif isinstance(op_ref, torch.nn.SiLU):
            kind = ATEN_SILU
        else:
            raise T2NErrorNotImplemented(op_ref)
    else:
        inputs = inputs[1:]
        # use appropriate graph
        _, op_ref_traced = _unfold_graph_getattr_by_node(
            traced_module, value_call_ref.node()
        )
        op_ref = TorchModuleTracer(
            op_ref, op_ref_traced, fn_name=node.s("name")
        )
    call_name = value_call_ref.debugName()
    if op_ref == module:
        raise T2NError(
            "Bug: Recursive call detected ! "
            f"Trying to parse same Pytorch IR sub-module twice: {op_ref}"
        )
    return kind, call_name, op_ref, inputs, module_getter_ref


def _extract_op_infos(
    module,
    data_nodes: ReactiveNamedItemDict,
    node: torch._C.Node,
    traced_module: torch.jit.TracedModule,
) -> T.Tuple[
    str, T.Optional[str], str, T.Callable[[T.Any], T.Any], T.List[Data]
]:
    """Extract informations from module or torch operation."""
    call_name = None
    kind: str = node.kind()
    inputs = list(node.inputs())

    if kind == CALL_KIND:
        kind, call_name, op_ref, inputs, module_getter_ref = (
            _extract_op_infos_call_kind(module, traced_module, node, inputs)
        )

    elif kind.startswith("quantized::"):
        module_getter_ref = MODULE_PATH_QUANTIZED
        op_ref = quantized_name_to_torch_fn(kind)
        for inp in inputs:
            in_name = inp.debugName()
            try:
                _find_data_node(data_nodes, in_name)
            except T2NErrorTorchNotFoundDataNode:
                _parse_getattr_script_obj(inp.node(), module, data_nodes)
    else:
        module_getter_ref = MODULE_PATH_ATEN
        if kind in MAP_TO_NOP:
            op_ref = nop  # type: ignore
        elif kind.startswith(ATEN_STARTID):
            op_ref, inputs = _aten_inputs_and_op_ref(kind, inputs, data_nodes)
        else:
            raise T2NErrorNotImplemented(
                f"Unable to extract operation from {kind}"
            )

    abstracted_inputs: T.List[Data] = []
    for inp in inputs:
        if isinstance(inp, Data):
            abstracted_inputs.append(inp)
        else:
            try:
                dn = _find_data_node(data_nodes, inp.debugName())
                abstracted_inputs.append(dn)
            except T2NErrorTorchNotFoundDataNode:
                logging.debug(
                    "Data node %s not found, trying to parse node_c_value",
                    inp.debugName(),
                )

    return (kind, call_name, module_getter_ref, op_ref, abstracted_inputs)
