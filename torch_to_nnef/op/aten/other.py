import logging
import platform
import typing as T

import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import (
    SCALAR_TYPE_TO_PYTORCH_TYPE,
    TORCH_DTYPE_TO_TRACT_STR,
    numpy_dtype_to_tract_str,
)
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import (
    InferenceTarget,
    KhronosNNEF,
    TractNNEF,
)
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    SimpleOpChainer,
    add_nnef_operation,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
)
from torch_to_nnef.torch_graph import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.utils import warn_once

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()

_EXTERNAL_DTYPE_PRECISE_ENOUGHT = {np.float32, np.int64}


@OP_REGISTRY.register()
def external(
    g: NGraph,
    node: TensorVariable,
    name_to_tensor: T.Dict[str, NTensor],
    inference_target: InferenceTarget,
):
    """Add External NNEF Operation in graph."""
    nnef_tensor_ref = add_tensor_variable_node_as_nnef_tensor(
        g, node, name_to_tensor, prevent_variable=True
    )
    custom_fragments = []
    if isinstance(inference_target, KhronosNNEF):
        if node.dtype not in _EXTERNAL_DTYPE_PRECISE_ENOUGHT:
            LOGGER.warning(
                "NNEF Spec is not precise enough "
                "to ensure correct mapping of numpy type %s",
                nnef_tensor_ref.dtype,
            )
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
    elif isinstance(inference_target, TractNNEF):
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
    else:
        raise T2NErrorNotImplemented(f"inference_target: {inference_target}")
    return nnef_tensor_ref, custom_fragments


@OP_REGISTRY.register(["dropout", "native_dropout"])
def dropout(node, torch_graph, **kwargs):
    """Map PyTorch: 'aten:dropout' to NNEF."""
    (
        input_node,
        _,  # probability
        is_active_node,
    ) = node.inputs
    # should wire directly input_node to output without intermediate
    if is_active_node.data:
        raise T2NErrorNotImplemented("dropout active at inference")

    # this replace order is important for graph of single nodes or starting with
    torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


@OP_REGISTRY.register()
def detach(node, torch_graph, **kwargs):
    """This does not translate to any operation."""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


@OP_REGISTRY.register()
def contiguous(node, torch_graph, **kwargs):
    """This does not translate to any operation."""
    torch_graph.remap_node(from_node=node.outputs[0], to_node=node.inputs[0])
    torch_graph.op_nodes = [_ for _ in torch_graph.op_nodes if _ is not node]


@OP_REGISTRY.register()
def to(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:to' to NNEF."""
    (
        input_node,
        *_,  # dtype_name, non_blocking_name, copy_name, memory_format_name
    ) = node.inputs

    onode = node.outputs[0]
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(f"`to` with {inference_target} ?")
    LOGGER.debug(
        "convert .to() with tract custom operator since it can express "
        "all torch type (contrary to vanilla cast NNEF operator)"
    )
    input_nnef = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if (
        node.inputs[0].dtype == torch.float32
        and not onode.dtype.is_signed
        and not platform.machine().startswith("arm")
    ):
        LOGGER.warning(
            "reinterpret cast to unsigned, if negative number is cpu "
            "device dependant (arm trunk bits while intel circular buffer left)"
        )
        # simulate a reinterpret_cast as implicitly done in PyTorch
        input_nnef = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=input_nnef,
            attrs={
                "to": TORCH_DTYPE_TO_TRACT_STR[torch.int64],
            },
        )

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=input_nnef,
        attrs={
            "to": TORCH_DTYPE_TO_TRACT_STR[onode.dtype],
            # "shape": list(onode.shape),
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def type_as(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:type_as' to NNEF."""
    (
        input_node,
        _,  # ref_node
    ) = node.inputs

    onode = node.outputs[0]
    LOGGER.debug(
        "convert .to() with tract custom operator since it can express "
        "all torch type (contrary to vanilla cast NNEF operator)"
    )
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(f"`to` with {inference_target} ?")
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
# pylint: disable-next=too-many-positional-arguments
def size(
    g,
    node,
    name_to_tensor,
    inference_target,
    torch_graph,
    op_helper,
    **kwargs,
):
    """Map PyTorch `aten::size` as NNEF.

    We can not use NNEF shape_of that have been deprecated since 1.0.1 version:.

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
    for example using stream size to apply an averaging:
    We map it to `tract_core_shape_of`

    """
    input_node, axis_node = node.inputs
    if not inference_target.has_dynamic_axes:
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

        warn_once(
            LOGGER,
            "aten::size replaced by constant traced value (follows NNEF spec)."
            "Keeping dynamism would require dynamic_axes specified.",
        )
        return []
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)

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

    begin = pick_axis(input_node, axis_node.data)

    index_tensor_name = f"{input_tensor.name}_dim{begin}"
    if index_tensor_name not in name_to_tensor:
        soc = soc.chain(
            "slice",
            attrs={
                "axes": [0],
                "begin": [begin],
                "end": [begin + 1],
                "stride": [1],
            },
            output_tensor_name_suffix=f"sliced{begin}",
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


@OP_REGISTRY.register()
def numel(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:numel' to NNEF."""
    assert len(node.inputs) == 1
    input_node = node.inputs[0]
    soc = SimpleOpChainer(op_helper=op_helper, input_data_nodes=[input_node])
    soc = (
        soc.chain(
            "tract_core_shape_of",
            force_full_output_tensor_name=f"{input_node.export_name}_shape",
        )
        .chain(
            "tract_core_cast",
            force_full_output_tensor_name=f"{input_node.export_name}_shape_i64",
            attrs={
                "to": TORCH_DTYPE_TO_TRACT_STR[torch.int64],
            },
        )
        .chain(
            "tract_core_product_reduce",
            force_full_output_tensor_name=f"{node.outputs[0].export_name}_reduced",
            attrs={"axes": [0]},
        )
        .chain(
            "squeeze",
            force_full_output_tensor_name=node.outputs[0].export_name,
            attrs={"axes": [0]},
        )
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def scalar_tensor(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:scalar_tensor' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("need casting")
    val_node, dtype_node, *_ = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_cast",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(val_node),
        attrs={
            "to": TORCH_DTYPE_TO_TRACT_STR[
                SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
            ],
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def _to_copy(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:_to_copy' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("need casting")
    val_node, dtype_node, *_ = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_cast",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(val_node),
        attrs={
            "to": TORCH_DTYPE_TO_TRACT_STR[
                SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
            ],
        },
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def isnan(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:isnan' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("need special NNEF operator")
    assert len(node.inputs) == 1
    val_node = node.inputs[0]
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_is_nan",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(val_node),
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def isinf(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:isinf' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("need special NNEF operator")
    assert len(node.inputs) == 1
    val_node = node.inputs[0]
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_is_inf",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(val_node),
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def isposinf(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:isposinf' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("need special NNEF operator")
    assert len(node.inputs) == 1
    val_node = node.inputs[0]
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_is_inf",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(val_node),
        attrs={"detect_negative": False},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def isneginf(node, inference_target, op_helper, **kwargs):
    """Map PyTorch: 'aten:isneginf' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("need special NNEF operator")
    assert len(node.inputs) == 1
    val_node = node.inputs[0]
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_is_inf",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(val_node),
        attrs={"detect_positive": False},
    )
    return ["tract_core"]
