import logging

import nnef
import numpy as np
import torch

from torch_to_nnef.dtypes import NUMPY_DTYPE_TO_STR, SCALAR_TYPE_TO_PYTORCH_TYPE
from torch_to_nnef.exceptions import T2NErrorNotImplemented, T2NErrorTract
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
    unary_output_op_without_attr,
)
from torch_to_nnef.torch_graph import (
    MAP_TO_NOP,
    FixedTensorList,
    TensorVariable,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def arange(g, node, name_to_tensor, inference_target, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    if len(node.inputs) == 4:
        # for now should never happen since dtype info is
        (start_node, end_node, step_node, dtype_node) = node.inputs
    else:
        raise T2NErrorNotImplemented(
            f"arange with {len(node.inputs)} inputs (see `ir_helpers` module)"
        )

    if dtype_node.data not in [6, None, 4, 3]:  # accept float, int64, int
        # see SCALAR_TYPE_TO_PYTORCH_TYPE for reference index
        raise T2NErrorNotImplemented(
            f"dtype {dtype_node} not implemented for arange"
        )

    if inference_target.has_dynamic_axes or isinstance(
        inference_target, TractNNEF
    ):
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(inference_target)
        if inference_target.version < "0.20.0":
            raise T2NErrorTract(
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
        raise T2NErrorNotImplemented(
            "Dynamic arange not handled in strict NNEF For now"
        )

    node.outputs[0].data.set_data(
        torch.arange(start_node.data, end_node.data, step=step_node.data)
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )
    return []


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
    """In case the tensor need to be dependant on shape of another."""
    if isinstance(shape_node, (list, tuple)) and all(
        isinstance(d, (int, str)) for d in shape_node
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
    new_output_tensor = tensor_build_fn(fixed_dim, dtype=dtype)
    node.outputs[0].set_data(
        new_output_tensor, force_dtype=True, force_shape=True
    )
    if to_expand_dim and has_dynamic_axes:
        LOGGER.debug(
            "the aten::ones replaced by constant traced values"
            " with additional expansion (follows NNEF spec)."
        )
        cached_input = get_or_add_tensor_variable_in_nnef(
            g, base_tensor_node, name_to_tensor, name_suffix="to_be_expanded"
        )
        repeats = [1 for _ in range(len(fixed_dim))]
        for k, v in to_expand_dim.items():
            repeats[k] = v
        if cached_input.dtype == np.int64:
            cached_input = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_cast",
                inputs=cached_input,
                attrs={"to": NUMPY_DTYPE_TO_STR[np.int64]},
                output_tensor_name_suffix=f"{cached_input.name}_casted",
            )

        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tile",
            inputs=cached_input,
            attrs={"repeats": repeats},
        )
    else:
        # late bug catching
        if base_tensor_node.data.dtype != base_tensor_node.dtype:
            LOGGER.warning(
                "late 'dtype' miss-alignment catched in "
                "_generic_auto_tensor_expansion"
            )
            base_tensor_node.set_data(
                base_tensor_node.data.to(base_tensor_node.dtype),
                force_dtype=True,
            )
        add_tensor_variable_node_as_nnef_tensor(
            g,
            base_tensor_node,
            name_to_tensor,
        )


@OP_REGISTRY.register()
def ones(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
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
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.ones,
    )


def _x_like(
    g,
    torch_graph,
    name_to_tensor,
    node,
    inference_target,
    tensor_build_fn,
    **kwargs,
):
    (input_node, *_) = node.inputs
    dtype = input_node.dtype
    if len(_) > 0:
        dtype_node = _[0]
        if dtype_node.data:
            dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]

    shape_node = input_node.shape
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
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
            inputs=(input_tensor,),
            force_full_output_tensor_name=shape_tensor_name,
        )
        shape_node = FixedTensorList(name="recomposed_shape_node", data=[])
        for dim in range(
            input_node.rank
        ):  # assume always same rank at each graph run
            index_tensor_name = f"{shape_tensor_name}_{dim}"
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=(shape_tensor,),
                attrs={
                    "axes": [0],
                    "begin": [dim],
                    "end": [dim + 1],
                    "stride": [1],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
            index_tensor_name = f"{shape_tensor_name}_{dim}_scalar"
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "squeeze",
                inputs=(out,),
                attrs={
                    "axes": [0],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
            shape_node.data.append(
                TensorVariable(
                    name=str(out.name),
                    data=None,
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
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=tensor_build_fn,
    )


@OP_REGISTRY.register()
def zeros_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use exapnsion

    """
    return _x_like(tensor_build_fn=torch.zeros, **kwargs)


@OP_REGISTRY.register()
def empty_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use expansion

    """
    return _x_like(tensor_build_fn=torch.zeros, **kwargs)


@OP_REGISTRY.register()
def ones_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use expansion

    """
    return _x_like(tensor_build_fn=torch.ones, **kwargs)


@OP_REGISTRY.register()
def full_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use expansion

    """
    fill_value = kwargs["node"].inputs.pop(1).data
    return _x_like(
        tensor_build_fn=lambda sh, dtype: torch.full(
            sh, fill_value, dtype=dtype
        ),
        **kwargs,
    )


@OP_REGISTRY.register()
def new_zeros(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:new_zeros' to NNEF."""
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
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def zeros(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:zeros' to NNEF."""
    (
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs
    LOGGER.warning(
        "the aten::zeros replaced by constant traced values "
        "(follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = (
        SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
        if dtype_node.data
        else torch.float32
    )
    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def full(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:full' to NNEF."""
    (shape_node, val_node, _, _, _, _) = node.inputs  # device_node,  # False

    def full_fn(*args, **kwargs):
        return torch.ones(*args, **kwargs) * val_node.data

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=torch.float32,
        tensor_build_fn=full_fn,
    )


@OP_REGISTRY.register(["fill", "fill_"])
def fill(
    g, node, name_to_tensor, torch_graph, inference_target, op_helper, **kwargs
):
    """Map PyTorch: 'aten:fill', 'aten:fill_' to NNEF."""
    (input_node, val_node, *_) = node.inputs  # device_node,  # False

    def full_fn(*args, **kwargs):
        return torch.ones(*args, **kwargs) * val_node.data

    if inference_target.has_dynamic_axes:
        dims_nnef = []
        for ix, _ in enumerate(input_node.shape[:]):
            soc = get_tract_dyn_axis_size_soc(op_helper, input_node, ix)
            dims_nnef.append(soc.output_name)
    else:
        dims_nnef = input_node.shape[:]
    shape_node = dims_nnef

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=input_node.dtype,
        tensor_build_fn=full_fn,
    )


@OP_REGISTRY.register(torch_op_ids=["copy", "clone"])
def copy(
    g, node, name_to_tensor, inference_target, torch_graph, null_ref, **kwargs
):
    """Map PyTorch: 'aten:copy', 'aten:clone' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        # nnef spec include copy fragment
        return unary_output_op_without_attr(
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
def _post_graph_creation_remap(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: no-ops to NNEF.

    List of no-ops:
        'aten:prim::NumToTensor',
        'aten:prim::ListConstruct',
        'aten:ScalarImplicit',
        'aten:alias'
    """
    torch_graph.remap_node(node.outputs[0], node.inputs[0])


def _trilu(g, name_to_tensor, node, inference_target, is_upper: bool = True):
    (input_node, diag_node) = node.inputs
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("trilu need `tract_core_trilu`")

    if inference_target.version < "0.21.3":
        raise T2NErrorNotImplemented(
            "triu need `tract_core_trilu` from tract >= 0.21.4 "
            "(prior nnef deserialization was failing)"
        )

    # k = 0
    # upper =true
    if isinstance(diag_node, PythonConstant):
        k_diag = diag_node.data
    else:
        k_diag_tensor = get_or_add_tensor_variable_in_nnef(
            g, diag_node, name_to_tensor
        )
        k_diag = nnef.Identifier(k_diag_tensor.name)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_trilu",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
        ],
        attrs={"upper": is_upper, "k": k_diag},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def triu(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:triu' to NNEF."""
    return _trilu(g, name_to_tensor, node, inference_target, is_upper=True)


@OP_REGISTRY.register()
def tril(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:tril' to NNEF."""
    return _trilu(g, name_to_tensor, node, inference_target, is_upper=False)
