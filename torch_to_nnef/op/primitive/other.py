import logging
import typing as T

import numpy as np
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import (
    TORCH_DTYPE_TO_TRACT_STR,
    numpy_dtype_to_tract_str,
)
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    SimpleOpChainer,
    add_nnef_operation,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
)
from torch_to_nnef.torch_graph import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()

_EXTERNAL_MAP_UNPRECISE_NNEF_TO_PRECISE_TRACT = {np.float32, np.int64}


@OP_REGISTRY.register()
def external(
    g: NGraph,
    node: TensorVariable,
    name_to_tensor: T.Dict[str, NTensor],
    nnef_spec_strict: bool,
):
    """Add External NNEF Operation in graph"""
    nnef_tensor_ref = add_tensor_variable_node_as_nnef_tensor(
        g, node, name_to_tensor, prevent_variable=True
    )
    custom_fragments = []
    if nnef_tensor_ref.dtype in _EXTERNAL_MAP_UNPRECISE_NNEF_TO_PRECISE_TRACT:
        add_nnef_operation(
            graph=g,
            type="external",
            inputs=None,
            outputs=nnef_tensor_ref,
            attribs={
                "shape": list(nnef_tensor_ref.shape),
                "dtype": nnef_tensor_ref.dtype,
            },
        )
    else:
        if nnef_spec_strict:
            raise ValueError(
                "NNEF Spec is not precise enough "
                f"to ensure correct mapping of numpy type {nnef_tensor_ref.dtype}"
            )
        add_nnef_operation(
            graph=g,
            type="tract_core_external",
            inputs=None,
            outputs=nnef_tensor_ref,
            attribs={
                "shape": list(nnef_tensor_ref.shape),
                "datum_type": numpy_dtype_to_tract_str(nnef_tensor_ref.dtype),
            },
        )
        custom_fragments.append("tract_core")
    return nnef_tensor_ref, custom_fragments


@OP_REGISTRY.register()
def dropout(node, torch_graph, **kwargs):
    (
        input_node,
        _,  # probability
        is_active_node,
    ) = node.inputs
    # should wire directly input_node to output without intermediate
    if is_active_node.data:
        raise TorchToNNEFNotImplementedError("dropout active at inference")

    # this replace order is important for graph of single nodes or starting with
    torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


@OP_REGISTRY.register()
def detach(node, torch_graph, **kwargs):
    """This does not translate to any operation"""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


@OP_REGISTRY.register()
def contiguous(node, torch_graph, **kwargs):
    """This does not translate to any operation"""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


@OP_REGISTRY.register()
def to(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        input_node,
        *_,  # dtype_name, non_blocking_name, copy_name, memory_format_name
    ) = node.inputs

    onode = node.outputs[0]
    LOGGER.debug(
        "convert .to() with tract custom operator since it can express "
        "all torch type (contrary to vanilla cast NNEF operator)"
    )
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError("`to` with nnef_spec_strict ?")
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "to": TORCH_DTYPE_TO_TRACT_STR[onode.dtype],
            # "shape": list(onode.shape),
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def type_as(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        input_node,
        _,  # ref_node
    ) = node.inputs

    onode = node.outputs[0]
    LOGGER.debug(
        "convert .to() with tract custom operator since it can express "
        "all torch type (contrary to vanilla cast NNEF operator)"
    )
    if nnef_spec_strict:
        raise TorchToNNEFNotImplementedError("`to` with nnef_spec_strict ?")
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "to": TORCH_DTYPE_TO_TRACT_STR[onode.dtype],
            # "shape": list(onode.shape),
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def size(
    g,
    node,
    name_to_tensor,
    nnef_spec_strict,
    has_dynamic_axes,
    torch_graph,
    op_helper,
    **kwargs,
):
    """
    We can not use NNEF shape_of that have been deprecated since 1.0.1 version:

    ```
    The shape_of function is deprecated and is discouraged from use.
    The reason is that it provides syntactic means to access a
    property of tensors that is not defined via the syntax itself.

    Furthermore, its definition is problematic in cases where the shape
    of a tensor is not known in graph compilation time.

    These result in problems with custom operations and operations with results
    of dynamic shape for a consumer of an NNEF document.

    By removing support for the shape_of function from NNEF syntax,
    it becomes possible to de-couple parsing
    from shape propagation in a consumer of an NNEF document.
    ```

    Since it is a core component to express some dynamic network that may use
    tract symbolic dimensions:
    by example using stream size to apply an averaging:
    We map it to `tract_core_shape_of`

    """
    input_node, axis_node = node.inputs
    if nnef_spec_strict or not has_dynamic_axes:
        original_vec_node, axis_node = node.inputs
        original_variable_output = node.outputs[0]
        if original_variable_output.data is None:
            dim = original_vec_node.shape[axis_node.data]
        else:
            dim = original_variable_output.data.numpy().tolist()
        new_node = PythonConstant(
            name=original_variable_output.name,
            data=dim,
        )
        torch_graph.remap_node(original_variable_output, new_node)

        for data_node in torch_graph.data_nodes[:]:
            if (
                isinstance(data_node, FixedTensorList)
                and any(_ is new_node for _ in data_node.data)
                and all(isinstance(_, PythonConstant) for _ in data_node.data)
            ):
                # recompute fixed data based on new infos
                torch_graph.remap_node(
                    data_node,
                    PythonConstant(
                        name=data_node.name,
                        data=[_.data for _ in data_node.data],
                    ),
                )
        torch_graph.op_nodes = [
            _ for _ in torch_graph.op_nodes if _ is not node
        ]

        LOGGER.warning(
            "aten::size replaced by constant traced value (follows NNEF spec)."
            "Keeping dynamism would require dynamic_axes specified."
        )
        return []
    # original_variable_output = node.outputs[0]

    # ensure consistant name to avoid strangeness
    input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    shape_tensor_name = f"{input_tensor.name}_shape"
    soc = SimpleOpChainer(op_helper=op_helper, input_data_nodes=[input_node])
    soc = soc.chain(
        "tract_core_shape_of",
        force_full_output_tensor_name=shape_tensor_name,
    )

    begin = pick_rank(input_node, axis_node.data)

    index_tensor_name = f"{shape_tensor_name}_{begin}"
    if index_tensor_name not in name_to_tensor:
        soc = soc.chain(
            "slice",
            attrs={
                "axes": [0],
                "begin": [begin],
                "end": [begin + 1],
                "stride": [1],
            },
            output_tensor_name_suffix="sliced",
        ).chain(
            "squeeze",
            attrs={
                "axes": [0],
            },
            force_full_output_tensor_name=index_tensor_name,
        )
    outnode = node.outputs[0]
    new_outnode = torch_graph.find_node(index_tensor_name)
    if not new_outnode:
        new_outnode = TensorVariable(
            name=index_tensor_name,
            data=outnode.data,
            shape=outnode.shape,
            dtype=outnode.dtype,
        )
    torch_graph.remap_node(
        from_node=outnode,
        to_node=new_outnode,
    )

    return ["tract_core"]
