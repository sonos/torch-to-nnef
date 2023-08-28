import logging
import typing as T

import nnef
import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import NUMPY_TO_TORCH_DTYPE, numpy_dtype_to_tract_str
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.torch_graph import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)

LOGGER = logging.getLogger(__name__)


class OpRegistry:
    def __init__(self, torch_mod_id: str):
        self.torch_mod_id = torch_mod_id
        self._registry: T.Dict[
            str, T.Callable[..., T.Optional[T.List[str]]]
        ] = {}

    def register(self, torch_op_ids: T.Optional[T.List[str]] = None):
        """by default we take the name of the function if not specified"""

        def wrapper(decorated):
            nonlocal torch_op_ids
            if torch_op_ids is None:
                torch_op_ids = [decorated.__name__]
            for torch_id in torch_op_ids:
                assert (
                    torch_id not in self._registry
                ), f"'{torch_id}' already in registry"
                self._registry[torch_id] = decorated
            return decorated

        return wrapper

    def get(self, name: str):
        try:
            return self._registry[name]
        except KeyError as exp:
            raise TorchToNNEFNotImplementedError(
                f"'{name}' operator as not yet been translated "
                "to NNEF or registred"
            ) from exp

    def __add__(self, other: "OpRegistry"):
        new = OpRegistry(self.torch_mod_id)
        if self.torch_mod_id != other.torch_mod_id:
            raise ValueError(
                "try to group different torch_mod:"
                f"{self.torch_mod_id} != {other.torch_mod_id}"
            )
        new._registry = self._registry.copy()
        common_keys = set(new._registry.keys()).intersection(
            other._registry.keys()
        )
        assert len(common_keys) == 0, common_keys
        new._registry.update(other._registry.copy())
        return new


class AtenOpRegistry(OpRegistry):
    def __init__(self):
        super().__init__(torch_mod_id="aten")


class QuantizedOpRegistry(OpRegistry):
    def __init__(self):
        super().__init__(torch_mod_id="quantized")


def add_nnef_operation(
    graph, inputs, *args, force_consistent_inputs_shapes: bool = True, **kwargs
):
    if (
        isinstance(inputs, (list, tuple))
        and len(inputs) >= 2
        and force_consistent_inputs_shapes
    ):
        inputs = maybe_unsqueeze_to_consistent_inputs_ranks(graph, inputs)
    kwargs["graph"] = graph
    kwargs["inputs"] = inputs
    return NOperation(*args, **kwargs)


def add_tensor_variable_node_as_nnef_tensor(
    g: NGraph,
    node: TensorVariable,
    name_to_tensor: T.Dict[str, NTensor],
    name_suffix: str = "",
    prevent_variable: bool = False,
    force_full_output_tensor_name: T.Optional[str] = None,
):
    """Create NNEF tensor and register in graph from torch_graph.Data node

    It automatically adds variable if node is a torch tensor is associated
    (it avoids bloating nnef graph file with matrix values)

    """
    if force_full_output_tensor_name:
        name = force_full_output_tensor_name
    else:
        name = node.export_name
        if name_suffix:
            name += f"_{name_suffix}"

    nnef_tensor_ref = NTensor(
        g,
        name,
        dtype=node.np_dtype,
        shape=node.shape,
    )
    if node.data is not None:
        nnef_tensor_ref.data = node.data.detach().numpy()
        nnef_tensor_ref.shape = tuple(node.data.shape)
        if not prevent_variable and (
            len(node.data.size()) > 0 or "e" in str(nnef_tensor_ref.data)
        ):
            add_nnef_operation(
                graph=g,
                type="variable",
                inputs=None,
                outputs=nnef_tensor_ref,
                attribs={
                    "label": nnef_tensor_ref.name,
                    "shape": list(nnef_tensor_ref.shape),
                    "dtype": nnef_tensor_ref.dtype,
                },
            )

    name_to_tensor[name] = nnef_tensor_ref
    return nnef_tensor_ref


def maybe_unsqueeze_to_consistent_inputs_ranks(g, nnef_tensors):
    """May unsqueeze at 0 rank n time to ensure consistent rank between inputs

    This is done at export time and not inference time because:
    inference implementation may use 1 dim expansion from left to right
    like Tract or Tensorflow
    instead of Pytorch expansion which happen in opposite direction.

    """
    tensors_ranks = [len(_.shape) for _ in nnef_tensors]
    if len(set(tensors_ranks)) > 1:
        reference_rank = max(tensors_ranks)
        new_nnef_tensors = []
        for nnef_tensor in nnef_tensors:
            original_rank = len(nnef_tensor.shape)
            missing_dims = reference_rank - original_rank
            if missing_dims > 0 and (
                nnef_tensor.data is None or nnef_tensor.data.size != 1
            ):
                new_shape = list(nnef_tensor.shape)
                new_shape = ([0] * missing_dims) + new_shape
                unsqueeze_axes = [0] * missing_dims

                output_nnef_tensor = NTensor(
                    g,
                    name=f"{nnef_tensor.name}_expanded",
                    dtype=nnef_tensor.dtype,
                    shape=tuple(new_shape),
                )
                NOperation(
                    g,
                    type="unsqueeze",
                    attribs={"axes": unsqueeze_axes},
                    inputs=nnef_tensor,
                    outputs=output_nnef_tensor,
                )
                nnef_tensor = output_nnef_tensor
            new_nnef_tensors.append(nnef_tensor)

        if isinstance(nnef_tensors, list):
            nnef_tensors = new_nnef_tensors
        else:
            nnef_tensors = tuple(new_nnef_tensors)
    return nnef_tensors


def get_or_add_tensor_variable_in_nnef(
    g, node, name_to_tensor, name_suffix: str = ""
) -> NTensor:
    name = node.export_name
    if name_suffix:
        name += f"_{name_suffix}"

    if name not in name_to_tensor:
        if isinstance(node, PythonConstant):
            node = node.into_tensor_variable()
        add_tensor_variable_node_as_nnef_tensor(
            g, node, name_to_tensor, name_suffix
        )
    return name_to_tensor[name]


def add_single_output_op(
    g,
    node,
    name_to_tensor,
    nnef_op_type,
    inputs,
    attrs=None,
    ensure_tuple=True,
    output_tensor_name_suffix: str = "",
    pass_quantization_params: bool = False,
    force_full_output_tensor_name: T.Optional[str] = None,
    force_consistent_inputs_shapes: bool = True,
) -> NTensor:
    assert len(node.outputs) == 1
    out = add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
        name_suffix=output_tensor_name_suffix,
        prevent_variable=True,
        force_full_output_tensor_name=force_full_output_tensor_name,
    )
    if isinstance(inputs, list) and ensure_tuple:
        inputs = tuple(inputs)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=nnef_op_type,
        inputs=inputs,
        outputs=tuple([out]),
        attribs=attrs or {},
        force_consistent_inputs_shapes=force_consistent_inputs_shapes,
    )
    if pass_quantization_params:
        input_quants = (
            inputs if isinstance(inputs, NTensor) else inputs[0]
        ).quant
        if input_quants:
            out.quant = input_quants
    return out


def pick_rank(input_node, rank: int) -> int:
    """Enforce that axis, axes ect does contains only positive values"""
    if rank >= 0:
        return rank
    if isinstance(input_node, FixedTensorList):
        base_rank = len(input_node.data)
    else:
        base_rank = input_node.rank
    return base_rank + rank


def pick_value_in_rank(input_node, rank: int, index: int) -> int:
    """Enforce that index in axis does contains only positive values"""
    if not isinstance(index, int):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        else:
            raise TorchToNNEFNotImplementedError(type(index))
    if index >= 0:
        return index
    return input_node.shape[rank] + index


def unary_output_op_without_params(
    nnef_op_type: str, g, node, name_to_tensor, null_ref, **kwargs
):
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=nnef_op_type,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            if _ and not (isinstance(_.data, str) and _.data == "none")
            else null_ref
            for _ in node.inputs
        ],
    )


def unary_input_output_op_with_constant(nnef_op_type, **kwargs):
    # avoid unpacking then repacking {
    g = kwargs["g"]
    node = kwargs["node"]
    name_to_tensor = kwargs["name_to_tensor"]
    # }
    for const in node.inputs[1:]:
        if isinstance(const, PythonConstant):
            data = np.array(const.data)
        else:
            data = const.data.numpy()
        nptype = data.dtype.type

        name_to_tensor[const.export_name] = NTensor(
            g,
            const.export_name,
            data=data,
            dtype=nptype,
            shape=data.shape,
        )
    return unary_output_op_without_params(nnef_op_type, **kwargs)


def _prevent_raw_number_with_e_notation(g, name_to_tensor, value):
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        if "e" in str(value):
            # create a tensor to avoid issue with number formatting
            # containing exp notation 'e'
            nvalue = torch.from_numpy(np.array(value))
            var_name = f"var_{str(value).replace('.', '_').replace('+', 'p')}"
            return nnef.Identifier(
                get_or_add_tensor_variable_in_nnef(
                    g,
                    TensorVariable(
                        name=var_name,
                        data=nvalue,
                        shape=list(nvalue.shape),
                        dtype=nvalue.dtype,
                    ),
                    name_to_tensor,
                ).name
            )
    return value


def cast_inputs_and_attrs(inputs, attrs, g, name_to_tensor):
    """Catch all input or attr that would still be torch_graph values into NNEF"""
    casted_inputs = []
    casted_attrs = {}

    def cast(value):
        if isinstance(value, (int, str, float, NTensor)):
            return _prevent_raw_number_with_e_notation(g, name_to_tensor, value)
        if isinstance(value, TensorVariable):
            return nnef.Identifier(
                get_or_add_tensor_variable_in_nnef(
                    g, value, name_to_tensor
                ).name
            )
        if isinstance(value, PythonConstant):
            return value.data
        if isinstance(value, list):
            return [cast(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cast(v) for v in value)
        if value in list(NUMPY_TO_TORCH_DTYPE.keys()):
            return value
        if isinstance(value, torch.Tensor):
            nvalue = value.numpy()
            if nvalue.shape == ():
                nvalue = nvalue.tolist()
            return _prevent_raw_number_with_e_notation(
                g, name_to_tensor, nvalue
            )
        raise TorchToNNEFNotImplementedError(
            f"Wrong {value} value of type: {type(value)}"
        )

    if isinstance(inputs, (tuple, list)):
        for inp in inputs:
            casted_inputs.append(cast(inp))
        casted_inputs = tuple(casted_inputs)

        if isinstance(inputs, list):  # some case need list of args as 1st arg
            casted_inputs = list(inputs)
    else:
        casted_inputs = cast(inputs)

    if attrs:
        for attr_name, attr_value in attrs.items():
            casted_attrs[attr_name] = cast(attr_value)

    return casted_inputs, casted_attrs


def cast_and_add_nnef_operation(name_to_tensor, **kwargs):
    """ensure to cast parameters before adding operation to NNEF graph"""
    kwargs["inputs"], kwargs["attribs"] = cast_inputs_and_attrs(
        kwargs["inputs"],
        kwargs["attribs"],
        kwargs["graph"],
        name_to_tensor,
    )
    return add_nnef_operation(**kwargs)


def add_multi_output_op(
    g,
    node,
    name_to_tensor,
    nnef_op_type,
    inputs,
    attrs=None,
    ensure_tuple=True,
    output_tensor_name_suffix: str = "",
):
    if len(node.outputs) == 1:
        LOGGER.debug(
            "Obverved multi to be single output "
            "(which may be normal depending on graph)"
        )
    output_tensors = []
    for out_node in node.outputs:
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            name_suffix=output_tensor_name_suffix,
            prevent_variable=True,
        )
        output_tensors.append(out)

    if isinstance(inputs, list) and ensure_tuple:
        inputs = tuple(inputs)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=nnef_op_type,
        inputs=inputs,
        outputs=tuple(output_tensors),
        attribs=attrs or {},
    )
    return output_tensors


def weight_bias_and_output_tensor(
    g,
    node,
    weight_node,
    bias_node,
    name_to_tensor,
    null_ref,
):
    weight_ref = get_or_add_tensor_variable_in_nnef(
        node=weight_node,
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix="weight" if weight_node.data is not None else "",
    )

    bias_ref = null_ref
    if bias_node.data is not None:
        bias_ref = get_or_add_tensor_variable_in_nnef(
            node=bias_node,
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="bias" if bias_node.data is not None else "",
        )

    out_node = node.outputs[0]
    out_tensor_name = out_node.export_name
    output_tensor = NTensor(
        graph=g,
        name=out_tensor_name,
        dtype=weight_ref.dtype,
        shape=tuple(out_node.shape) if out_node.shape else None,
    )
    name_to_tensor[out_tensor_name] = output_tensor
    return weight_ref, bias_ref, output_tensor


def get_list_of_int(
    data_node,
    torch_graph,
    name_to_tensor,
    accept_none: int = 0,
    has_dynamic_axes: bool = True,
    force_none_as_tensor_ref: bool = False,
):
    assert accept_none >= 0
    accepted_none = 0

    def cast_element(node, accepted_none):
        nonlocal has_dynamic_axes
        tensor = name_to_tensor.get(node.export_name)
        if tensor is not None and (
            force_none_as_tensor_ref or has_dynamic_axes
        ):
            return nnef.Identifier(tensor.name)
        val = node.data
        if val is None and accept_none > 0 and accepted_none < accept_none:
            accepted_none += 1
            return val
        return int(val)

    if isinstance(data_node, PythonConstant):
        int_list = [int(_) for _ in data_node.data]
    elif isinstance(data_node, FixedTensorList):
        int_list = [cast_element(_, accepted_none) for _ in data_node.data]
        if any(_ is None for _ in int_list):
            for ax_data in data_node.data:
                if ax_data.data is None:
                    producer = torch_graph.find_data_node_producer(ax_data)
                    producer.realise_output_type_and_size()
                    if ax_data.data is not None:
                        ax_data.data = ax_data.data.tolist()
            int_list = [cast_element(_, accepted_none) for _ in data_node.data]
            if len([_ for _ in int_list if _ is None]) > 1:
                raise TorchToNNEFNotImplementedError(
                    f"too much unknown dimensions for view {int_list}"
                )
    else:
        raise TorchToNNEFNotImplementedError(
            "Extracting int list from ", data_node
        )

    assert all(
        isinstance(_, (nnef.Identifier, int)) for _ in int_list
    ), int_list
    return int_list


def cast_to_if_not_dtype_and_variable(
    g,
    name_to_tensor,
    node,
    nnef_tensor: NTensor,
    cast_to: np.dtype,
    suffix: str = "",
):
    """Force casting not expressed in IR graph in case of div by example.

    This is neccessary since tract and maybe other inference engine may not cast
    implicitly to float during div operation by example leading to rounding
    issues.

    """
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=nnef_tensor,
        attrs={
            "to": numpy_dtype_to_tract_str(cast_to),
        },
        output_tensor_name_suffix=suffix,
    )
    return out, ["tract_core"]
