import logging

import torch

from torch_to_nnef.dtypes import SCALAR_TYPE_TO_PYTORCH_TYPE
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_tensor_variable_node_as_nnef_tensor,
    get_list_of_int,
    unary_output_op_without_params,
)
from torch_to_nnef.torch_graph import MAP_TO_NOP

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def arange(g, node, name_to_tensor, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (start_node, end_node, step_node) = node.inputs
    LOGGER.warning(
        "aten::arange replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    node.outputs[0].data = torch.arange(
        start_node.data, end_node.data, step=step_node.data
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


@OP_REGISTRY.register()
def ones(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    LOGGER.warning(
        "the aten::ones replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = torch.float32
    if len(_) > 0:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[_[0].data]
    dim_data = get_list_of_int(
        input_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
    )
    node.outputs[0].data = torch.ones(dim_data, dtype=dtype)
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


@OP_REGISTRY.register()
def zeros_like(g, node, name_to_tensor, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    LOGGER.warning(
        "the aten::zeros_like replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = torch.float32
    if len(_) > 0:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[_[0].data]

    node.outputs[0].data = torch.zeros(
        input_node.shape,
        dtype=dtype,
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


@OP_REGISTRY.register()
def new_zeros(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        _,  # input_node,
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs
    LOGGER.warning(
        "the aten::new_zeros replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]

    assert shape_node.data
    assert all(isinstance(v, int) for v in shape_node.data), shape_node.data

    node.outputs[0].data = torch.zeros(
        *shape_node.data,
        dtype=dtype,
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


@OP_REGISTRY.register()
def zeros(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs
    LOGGER.warning(
        "the aten::zeros replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]

    assert shape_node.data
    assert all(isinstance(v, int) for v in shape_node.data), shape_node.data

    node.outputs[0].data = torch.zeros(
        *shape_node.data,
        dtype=dtype,
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


@OP_REGISTRY.register()
def full(g, node, name_to_tensor, nnef_spec_strict, **kwargs):
    (shape_node, val_node, _, _, _, _) = node.inputs  # device_node,  # False
    LOGGER.warning(
        "the aten::zeros replaced by constant traced values (follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    assert shape_node.data
    assert all(isinstance(v, int) for v in shape_node.data), shape_node.data

    node.outputs[0].data = (
        torch.ones(*shape_node.data, dtype=val_node.dtype) * val_node.data
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


@OP_REGISTRY.register(torch_op_ids=["copy", "clone"])
def copy(
    g, node, name_to_tensor, nnef_spec_strict, torch_graph, null_ref, **kwargs
):
    if nnef_spec_strict:
        # nnef spec include copy fragment
        return unary_output_op_without_params(
            nnef_op_type="copy",
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    torch_graph.remap_node(node.outputs[0], node.inputs[0])


@OP_REGISTRY.register(
    torch_op_ids=[_.replace("aten::", "") for _ in MAP_TO_NOP]
)
def _post_graph_creation_remap(
    g, node, name_to_tensor, nnef_spec_strict, torch_graph, null_ref, **kwargs
):
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
