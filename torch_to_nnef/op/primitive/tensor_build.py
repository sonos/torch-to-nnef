import logging

import torch

from torch_to_nnef.dtypes import SCALAR_TYPE_TO_PYTORCH_TYPE
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    unary_output_op_without_params,
)
from torch_to_nnef.torch_graph import (
    MAP_TO_NOP,
    FixedTensorList,
    TensorVariable,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant
from torch_to_nnef.tract import tract_version_lower_than

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def arange(
    g, node, name_to_tensor, nnef_spec_strict, has_dynamic_axes: bool, **kwargs
):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    if len(node.inputs) == 3:
        (start_node, end_node, dtype_node) = node.inputs
        step_node = PythonConstant(name=f"step_node_{node.export_name}", data=1)
    elif len(node.inputs) == 4:
        (start_node, end_node, dtype_node, step_node) = node.inputs
    else:
        raise NotImplementedError(f"arange with {len(node.inputs)} inputs")

    if dtype_node.data != 1:
        raise NotImplementedError(
            f"dtype {dtype_node} not implemented for arange"
        )

    if not nnef_spec_strict or has_dynamic_axes:
        if tract_version_lower_than("0.20.0"):
            raise NotImplementedError(
                "please update to latest tract to use 'tract_core_range'"
            )

        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_range",
            inputs=[
                get_or_add_tensor_variable_in_nnef(
                    g, start_node, name_to_tensor
                ),
                get_or_add_tensor_variable_in_nnef(g, end_node, name_to_tensor),
            ]
            + (
                []
                if isinstance(step_node, PythonConstant)
                else [
                    get_or_add_tensor_variable_in_nnef(
                        g, step_node, name_to_tensor
                    ),
                ]
            ),
            attrs={"step": step_node.data}
            if isinstance(step_node, PythonConstant)
            else {},
        )
        return ["tract_core"]
    if start_node.data is None or end_node.data is None:
        raise NotImplementedError(
            "Dynamic arange not handled in strict NNEF For now"
        )

    node.outputs[0].data = torch.arange(
        start_node.data, end_node.data, step=step_node.data
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )


def _generic_auto_tensor_expansion(
    shape_node,
    node,
    g,
    torch_graph,
    name_to_tensor,
    has_dynamic_axes,
    dtype=torch.float32,
    tensor_build_fn=torch.ones,
):
    """In case the tensor need to be dependant on shape of another"""
    if isinstance(shape_node, (list, tuple)) and all(
        isinstance(d, int) for d in shape_node
    ):
        dim_data = shape_node
    else:
        dim_data = get_list_of_int(
            shape_node,
            torch_graph,
            name_to_tensor=name_to_tensor,
            has_dynamic_axes=has_dynamic_axes,
        )
    fixed_dim = []
    to_expand_dim = {}
    for dim_idx, dim_any in enumerate(dim_data):
        if isinstance(dim_any, str):
            fixed_dim.append(1)
            to_expand_dim[dim_idx] = dim_any
        else:
            assert isinstance(dim_any, int), dim_any
            fixed_dim.append(dim_any)

    base_tensor_node = node.outputs[0]
    if to_expand_dim and has_dynamic_axes:
        base_tensor_node.name += "_to_be_expanded"
    node.outputs[0].data = tensor_build_fn(fixed_dim, dtype=dtype)
    add_tensor_variable_node_as_nnef_tensor(
        g,
        base_tensor_node,
        name_to_tensor,
    )
    if to_expand_dim and has_dynamic_axes:
        LOGGER.debug(
            "the aten::ones replaced by constant traced values"
            " with additional expansion (follows NNEF spec)."
        )
        repeats = [1 for _ in range(len(fixed_dim))]
        for k, v in to_expand_dim.items():
            repeats[k] = v
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tile",
            inputs=get_or_add_tensor_variable_in_nnef(
                g, base_tensor_node, name_to_tensor
            ),
            attrs={"repeats": repeats},
        )


@OP_REGISTRY.register()
def ones(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    dtype = torch.float32
    if len(_) > 0:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[_[0].data]
    return _generic_auto_tensor_expansion(
        input_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.ones,
    )


@OP_REGISTRY.register()
def zeros_like(
    g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs
):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    dtype = torch.float32
    if len(_) > 0:
        dtype_node = _[0]
        if dtype_node.data:
            dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
        else:
            dtype = input_node.dtype

    shape_node = input_node.shape
    if has_dynamic_axes:
        # in this case we need to get full expansion of input_node shape
        input_tensor = get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        )
        shape_tensor_name = f"{input_tensor.name}_shape"
        shape_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_shape_of",
            inputs=input_tensor,
            force_full_output_tensor_name=shape_tensor_name,
        )
        shape_node = FixedTensorList(data=[])
        for dim in range(
            input_node.rank
        ):  # assume always same rank at each graph run
            index_tensor_name = f"{shape_tensor_name}_{dim}"
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=shape_tensor,
                attrs={
                    "axes": [0],
                    "begin": [dim],
                    "end": [dim + 1],
                    "stride": [1],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
            shape_node.data.append(
                TensorVariable(
                    name=out.name,
                    shape=[1],
                    dtype=input_node.dtype,
                )
            )

    return _generic_auto_tensor_expansion(
        shape_node,  # not dynamic for now
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def new_zeros(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    (
        input_node,  # input_node,
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs

    if dtype_node.data:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
    else:
        dtype = input_node.dtype

    assert shape_node.data

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def zeros(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
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
    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def full(g, node, name_to_tensor, torch_graph, has_dynamic_axes, **kwargs):
    (shape_node, val_node, _, _, _, _) = node.inputs  # device_node,  # False

    def full_fn(*args, **kwargs):
        return torch.ones(*args, **kwargs) * val_node.data

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=has_dynamic_axes,
        dtype=torch.float32,
        tensor_build_fn=full_fn,
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
    return []


@OP_REGISTRY.register(
    torch_op_ids=[_.replace("aten::", "") for _ in MAP_TO_NOP]
)
def _post_graph_creation_remap(
    g, node, name_to_tensor, nnef_spec_strict, torch_graph, null_ref, **kwargs
):
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
