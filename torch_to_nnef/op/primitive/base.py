import logging
import typing as T

import nnef
import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import (
    NUMPY_TO_TORCH_DTYPE,
    TORCH_TO_NUMPY_DTYPE,
    numpy_dtype_to_tract_str,
)
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.torch_graph import (
    Data,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.torch_graph.ir_op import TorchOp

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
    graph: NGraph,
    inputs: T.Optional[T.Tuple[NTensor]],
    *args,
    force_consistent_inputs_shapes: bool = True,
    **kwargs,
):
    if (
        isinstance(inputs, (list, tuple))
        and len(inputs) >= 2
        and force_consistent_inputs_shapes
    ):
        outputs = kwargs["outputs"]
        op_type = kwargs["type"]
        inputs = maybe_align_inputs_ranks(
            graph, inputs, outputs, op_type
        )  # type: ignore
    kwargs["graph"] = graph
    kwargs["inputs"] = inputs
    return NOperation(*args, **kwargs)


def nnef_tensor_from_tv(g: NGraph, name: str, node: TensorVariable):
    quant = None
    if node.quant and "shape" not in name:
        np_dtype = TORCH_TO_NUMPY_DTYPE[node.dtype]
        quant = {
            "scale": node.quant["scale"],
            "zero_point": node.quant["zero_point"],
            "bits": np_dtype().nbytes * 8,
            "signed": np.issubdtype(np_dtype, np.signedinteger),
            "symmetric": False,
            "op-name": "zero_point_linear_quantize",
        }
    return NTensor(g, name, dtype=node.np_dtype, shape=node.shape, quant=quant)


def add_tensor_variable_node_as_nnef_tensor(
    g: NGraph,
    node: TensorVariable,
    name_to_tensor: T.Dict[str, NTensor],
    name_suffix: str = "",
    prevent_variable: bool = False,
    force_full_output_tensor_name: T.Optional[str] = None,
) -> NTensor:
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

    nnef_tensor_ref = nnef_tensor_from_tv(g, name, node=node)
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


def maybe_align_inputs_ranks(
    g: NGraph,
    inputs: T.Sequence[NTensor],
    outputs: T.Sequence[NTensor],
    op_type: str,
) -> T.Sequence[NTensor]:
    """ensure consistent rank between inputs and outputs with regard to spec

    - May unsqueeze at 0 rank n time to align inputs

    This is done at export time and not inference time because:
    - inference implementation may use 1 dim expansion from left to right
    like Tract or Tensorflow
    instead of Pytorch expansion which happen in opposite direction.

    """
    tensors_ranks = [len(_.shape) for _ in inputs]
    if len(set(tensors_ranks)) > 1:
        all_inputs_shapes_are_scalar_like = all(
            rank == 0 or all(d == 1 for d in inputs[idx].shape)
            for idx, rank in enumerate(tensors_ranks)
        )
        if all_inputs_shapes_are_scalar_like and all(
            len(_.shape) == 0 for _ in outputs
        ):
            # auto squeeze useless dims
            new_inputs = []
            for nnef_tensor in inputs:
                if len(nnef_tensor.shape) > 0:
                    squeeze_axes = [0] * len(nnef_tensor.shape)
                    output_nnef_tensor = NTensor(
                        g,
                        name=f"{nnef_tensor.name}_aligned_rank_reduced",
                        dtype=nnef_tensor.dtype,
                        shape=tuple([]),
                    )
                    NOperation(
                        g,
                        type="squeeze",
                        attribs={"axes": squeeze_axes},
                        inputs=nnef_tensor,
                        outputs=output_nnef_tensor,
                    )
                    nnef_tensor = output_nnef_tensor
                new_inputs.append(nnef_tensor)
        else:
            # auto unsqueeze missing dims
            reference_rank = max(tensors_ranks)
            new_inputs = []
            for nnef_tensor in inputs:
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
                        name=f"{nnef_tensor.name}_aligned_rank_expanded",
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
                new_inputs.append(nnef_tensor)
        if isinstance(inputs, list):
            inputs = new_inputs
        else:
            inputs = tuple(new_inputs)
    return inputs


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
    g: NGraph,
    node,
    name_to_tensor,
    nnef_op_type: str,
    inputs: T.Union[NTensor, T.Sequence[NTensor]],
    attrs: T.Optional[T.Dict[str, T.Any]] = None,
    ensure_tuple: bool = True,
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
        if value in list(NUMPY_TO_TORCH_DTYPE):
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


def cast_and_add_nnef_operation(name_to_tensor: str, **kwargs):
    """ensure to cast parameters before adding operation to NNEF graph"""
    kwargs["inputs"], kwargs["attribs"] = cast_inputs_and_attrs(
        kwargs["inputs"],
        kwargs["attribs"],
        kwargs["graph"],
        name_to_tensor,
    )
    return add_nnef_operation(**kwargs)


def add_multi_output_op(
    g: NGraph,
    node,
    name_to_tensor,
    nnef_op_type,
    inputs: T.Sequence[NTensor],
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
    weight_suffix = ""
    if weight_node.data is not None and not weight_node.export_name.endswith(
        "__weight"
    ):
        weight_suffix = "weight"

    weight_ref = get_or_add_tensor_variable_in_nnef(
        node=weight_node,
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix=weight_suffix,
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


class OpHelper:
    def __init__(self, g, node, name_to_tensor, null_ref):
        self.g = g
        self.node = node
        self.name_to_tensor = name_to_tensor
        self.null_ref = null_ref

    def clone_with_new_node(self, node):
        return self.__class__(
            g=self.g,
            node=node,
            name_to_tensor=self.name_to_tensor,
            null_ref=self.null_ref,
        )

    def data_nodes_to_nnef_tensors(self, data_nodes):
        return [
            get_or_add_tensor_variable_in_nnef(self.g, dn, self.name_to_tensor)
            for dn in data_nodes
        ]

    def _guess_output_dtype_and_shape(
        self, nnef_op_type: str, input_nodes, attrs
    ):
        if nnef_op_type == "tract_core_shape_of":
            return torch.int64, (len(input_nodes[0].shape),)

        if nnef_op_type == "slice":
            if len(attrs["begin"]) != 1:
                raise NotImplementedError()
            if (
                isinstance(attrs["begin"][0], int)
                and isinstance(attrs["end"][0], int)
                and attrs.get("stride", [1])[0] == 1
            ):
                size = attrs["end"][0] - attrs["begin"][0]
                sh = list(input_nodes[0].shape)
                sh[0] = size
                return input_nodes[0].dtype, sh
        if nnef_op_type == "squeeze":
            sh = []
            for dim_idx, dim_value in enumerate(input_nodes[0].shape):
                if dim_idx in attrs["axes"]:
                    continue
                sh.append(dim_value)
            return input_nodes[0].dtype, tuple(sh)

        raise NotImplementedError(nnef_op_type)

    def new_single_output_op(
        self,
        nnef_op_type: str,
        *args,
        input_nodes: T.List[Data],
        output_tensor_name_suffix: str = "",
        force_full_output_tensor_name: str = "",
        **kwargs,
    ):
        dtype, shape = self._guess_output_dtype_and_shape(
            nnef_op_type, input_nodes, kwargs.get("attrs", {})
        )
        if force_full_output_tensor_name:
            output_name = force_full_output_tensor_name
        else:
            output_name = self.node.outputs[0].name
            if output_tensor_name_suffix:
                output_name += f"_{output_tensor_name_suffix}"
        new_output_data = TensorVariable(
            name=output_name, data=None, dtype=dtype, shape=shape
        )
        # assert output_name not in self.name_to_tensor, output_name
        new_node = TorchOp(
            kind=nnef_op_type,
            module_path=self.node.module_path,
            inputs=input_nodes,
            outputs=[new_output_data],
            scope=self.node.scope,
            op_ref=None,
            call_name=None,
        )
        kwargs["nnef_op_type"] = nnef_op_type
        assert "inputs" not in kwargs
        kwargs["inputs"] = self.data_nodes_to_nnef_tensors(input_nodes)
        _ = add_single_output_op(
            self.g, new_node, self.name_to_tensor, *args, **kwargs
        )
        return new_node


class SimpleOpChainer:
    def __init__(self, op_helper: OpHelper, input_data_nodes):
        self.op_helper = op_helper
        self.input_data_nodes = input_data_nodes

    @property
    def output_name(self):
        return self.input_data_nodes[0].export_name

    def chain(self, *args, **kwargs):
        new_node = self.op_helper.new_single_output_op(
            input_nodes=self.input_data_nodes, *args, **kwargs
        )
        return SimpleOpChainer(
            self.op_helper.clone_with_new_node(new_node), new_node.outputs
        )
