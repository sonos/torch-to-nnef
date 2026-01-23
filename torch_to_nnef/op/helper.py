import logging
import string
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
    str_to_torch_dtype,
)
from torch_to_nnef.exceptions import T2NErrorConsistency, T2NErrorNotImplemented
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.tensor import OpaqueTensorRef, QTensor
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.torch_graph import (
    Data,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.torch_graph.ir_data import cleanup_data_name
from torch_to_nnef.torch_graph.ir_op import TorchOp
from torch_to_nnef.utils import torch_version

LOGGER = logging.getLogger(__name__)

# unused see `_implicits_input_casting` doc (
DTYPES_EXPECTED_IMPLICIT_CAST_ORDER = [
    torch.float64,
    torch.float16,
    torch.float32,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]
NP_DTYPES_EXPECTED_IMPLICIT_CAST_ORDER = [
    TORCH_TO_NUMPY_DTYPE[_] for _ in DTYPES_EXPECTED_IMPLICIT_CAST_ORDER
]
# )
OPS_IMPLICIT_CAST_BY_OUTPUT_DTYPE = [
    "mul",
    "div",
    "add",
    "sub",
    "rsub",
    "pow",
]
OPS_IMPLICIT_CAST_CONSISTENT_INPS = [
    "ne",
    "ge",
    "le",
    "gt",
    "eq",
    "lt",
    "max",
    "min",
]

OPS_IMPLICIT_CAST_BINARY = [
    "and",
    "or",
    "xor",
]


class OpRegistry:
    def __init__(self, torch_mod_id: str):
        self.torch_mod_id = torch_mod_id
        self._registry: T.Dict[
            str, T.Callable[..., T.Optional[T.List[str]]]
        ] = {}

    def register(self, torch_op_ids: T.Optional[T.List[str]] = None):
        """By default we take the name of the function if not specified."""

        def wrapper(decorated):
            nonlocal torch_op_ids
            if torch_op_ids is None:
                torch_op_ids = [decorated.__name__]
            for torch_id in torch_op_ids:
                assert torch_id not in self._registry, (
                    f"'{torch_id}' already in registry"
                )
                self._registry[torch_id] = decorated
            if not decorated.__doc__:
                ops_str = ", ".join(
                    f"'{self.torch_mod_id}:{o}'" for o in torch_op_ids
                )
                decorated.__doc__ = f"Map PyTorch: {ops_str} to NNEF"
                decorated._auto_gen_doc = True
            else:
                decorated._auto_gen_doc = False
            return decorated

        return wrapper

    def get(self, name: str):
        try:
            return self._registry[name]
        except KeyError as exp:
            raise T2NErrorNotImplemented(
                f"'{name}' operator as not yet "
                "been translated to NNEF or registred"
            ) from exp

    def __add__(self, other: "OpRegistry"):
        new = OpRegistry(self.torch_mod_id)
        if self.torch_mod_id != other.torch_mod_id:
            raise T2NErrorConsistency(
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
        inputs = maybe_align_inputs_ranks(graph, inputs, outputs, op_type)  # type: ignore
    kwargs["graph"] = graph
    kwargs["inputs"] = inputs
    return NOperation(*args, **kwargs)


def nnef_tensor_from_tv(g: NGraph, name: str, node: TensorVariable):
    assert isinstance(node, TensorVariable)
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
    """Create NNEF tensor and register in graph from torch_graph.Data node.

    It automatically adds variable if node is a torch tensor is associated
    (it avoids bloating nnef graph file with matrix values)

    """
    if force_full_output_tensor_name:
        name = force_full_output_tensor_name
    else:
        name = node.export_name
        if name_suffix:
            name += f"_{name_suffix}"

    if isinstance(node, PythonConstant):
        node = node.into_tensor_variable()
    nnef_tensor_ref = nnef_tensor_from_tv(g, name, node=node)
    if node.data is not None:
        if isinstance(node.data, OpaqueTensorRef):
            node.set_data(node.data.opaque_tensor)

        if isinstance(node.data, QTensor) or (
            isinstance(node.data, OffloadedTensor)
            and issubclass(node.data.offloaded_tensor_type, QTensor)
        ):
            # main assign to allow corect dump
            nnef_tensor_ref.qtensor = node.data
            add_nnef_operation(
                graph=g,
                type="variable",
                inputs=None,
                outputs=nnef_tensor_ref,
                attribs={
                    "custom_datatype": "quant_tensor",
                    "label": node.data.nnef_name or nnef_tensor_ref.name,
                    "shape": list(nnef_tensor_ref.shape),
                },
            )
        else:
            nnef_tensor_ref.data = node.data
            nnef_tensor_ref.shape = tuple(node.data.shape)

            # pylint: disable-next=too-many-boolean-expressions
            if not prevent_variable and (
                len(node.data.shape) > 1
                or node.data.numel() == 0
                or node.data.numel() > 1
                or (
                    nnef_tensor_ref.data.numel() == 1
                    and "e" in str(nnef_tensor_ref.data.item())
                )
                or (  # special dtype need to avoid dtype undefined
                    # in NNEF graph
                    node.dtype
                    in [
                        torch.int16,
                        # torch.int64, # avoid for tract that create
                        # confusion with TDim
                        torch.float16,
                        torch.float64,
                    ]
                )
            ):
                add_nnef_operation(
                    graph=g,
                    type="variable",
                    inputs=None,
                    outputs=nnef_tensor_ref,
                    attribs={
                        "label": getattr(node.data, "nnef_name", None)
                        or nnef_tensor_ref.name,
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
    """Ensure consistent rank between inputs and outputs with regard to spec.

    - May unsqueeze at 0 rank n time to align inputs

    This is done at export time and not inference time because:
    - inference implementation may use 1 dim expansion from left to right
    like Tract or Tensorflow
    instead of PyTorch expansion which happen in opposite direction.

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
                    new_shape = ([1] * missing_dims) + new_shape
                    unsqueeze_axes = [0] * missing_dims

                    # print(nnef_tensor.name, nnef_tensor.dtype)
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
        inputs = new_inputs if isinstance(inputs, list) else tuple(new_inputs)
    return inputs


def get_or_add_tensor_variable_in_nnef(
    g, node, name_to_tensor, name_suffix: str = "", **kwargs
) -> NTensor:
    name = node.export_name
    if name_suffix:
        name += f"_{name_suffix}"

    kwargs["name_suffix"] = name_suffix
    if hasattr(node.data, "nnef_name") and not kwargs.get(
        "force_full_output_tensor_name", ""
    ):
        name = cleanup_data_name(node.data.nnef_name)
        if name[0] in string.digits:  # avoid var name issue
            name = f"_{name}"
        del kwargs["name_suffix"]
        node.name = name

    if name not in name_to_tensor:
        if isinstance(node, PythonConstant):
            node = node.into_tensor_variable()
        add_tensor_variable_node_as_nnef_tensor(
            g, node, name_to_tensor, **kwargs
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


def pick_axis(input_node, rank: int) -> int:
    """Enforce that axis, axes ect does contains only positive values."""
    if rank >= 0:
        return rank
    if isinstance(input_node, FixedTensorList):
        base_rank = len(input_node.data)
    else:
        base_rank = input_node.rank
    return base_rank + rank


def pick_index_in_axis(
    input_node, rank: int, index: int, check_is_positive: bool = True
) -> int:
    """Enforce that index in axis does contains only values within bounds.

    Because in case of tract out of bound is not supported !

    """
    if not isinstance(index, int) and not (
        isinstance(index, float) and index.is_integer()
    ):
        if not isinstance(index, torch.Tensor):
            raise T2NErrorNotImplemented(type(index))
        index = index.tolist()
    if index >= 0:
        return index
    new_index = input_node.shape[rank] + index
    if check_is_positive:
        assert new_index >= 0, new_index
    return int(new_index)


def unary_output_op_without_attr(
    nnef_op_type: str, g, node, name_to_tensor, null_ref, **kwargs
):
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=nnef_op_type,
        inputs=[
            (
                get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
                if _ and not (isinstance(_.data, str) and _.data == "none")
                else null_ref
            )
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
    return unary_output_op_without_attr(nnef_op_type, **kwargs)


def _prevent_raw_number_with_e_notation(g, name_to_tensor, value):
    if (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and "e" in str(value)
    ):
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
    """Catch input or attr that would still be torch_graph values into NNEF."""
    casted_inputs = []
    casted_attrs = {}

    def cast(value):  # pylint: disable=too-many-return-statements
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
        raise T2NErrorNotImplemented(
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
    """Ensure to cast parameters before adding operation to NNEF graph."""
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
    suffix_weight_name="",
    suffix_bias_name="",
    suffix_out_name="",
):
    if suffix_weight_name == "" and (
        weight_node.data is not None
        and not weight_node.export_name.endswith("__weight")
    ):
        suffix_weight_name = "weight"

    weight_ref = get_or_add_tensor_variable_in_nnef(
        node=weight_node,
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix=suffix_weight_name,
    )

    bias_ref = null_ref
    if isinstance(bias_node, TensorVariable) and bias_node.shape:
        if suffix_bias_name == "":
            suffix_bias_name = "bias" if bias_node.data is not None else ""
        bias_ref = get_or_add_tensor_variable_in_nnef(
            node=bias_node,
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix=suffix_bias_name,
        )

    out_node = node.outputs[0]
    out_tensor_name = out_node.export_name
    if suffix_out_name:
        out_tensor_name += suffix_out_name
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
                        ax_data.data.set_data(ax_data.data.tolist())
            int_list = [cast_element(_, accepted_none) for _ in data_node.data]
            if len([_ for _ in int_list if _ is None]) > 1:
                raise T2NErrorNotImplemented(
                    f"too much unknown dimensions for view {int_list}"
                )
    else:
        raise T2NErrorNotImplemented("Extracting int list from ", data_node)

    assert all(isinstance(_, (nnef.Identifier, int)) for _ in int_list), (
        int_list
    )
    return int_list


def cast_to_if_not_dtype_and_variable(
    g,
    name_to_tensor,
    node,
    nnef_tensor: NTensor,
    cast_to: np.dtype,
    suffix: str = "",
):
    """Force casting not expressed in IR graph in case of div for example.

    This is neccessary since tract and maybe other inference engine may not cast
    implicitly to float during div operation for example leading to rounding
    issues.

    """
    if torch_version() < "1.13.0" and cast_to == np.uint64:
        logging.warning(
            "discarded force casting to dtype=%s "
            "since obverved bug prior 1.13.0",
            cast_to,
        )
        cast_to = nnef_tensor.dtype
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


def _guess_out_dtype_and_shape_tc_shape_of(input_nodes, attrs):
    return torch.int64, (len(input_nodes[0].shape),)


def _guess_out_dtype_and_shape_slice(input_nodes, attrs):
    if len(attrs["begin"]) != 1:
        raise T2NErrorNotImplemented("len(begin) != 1 in slicing")
    if (
        isinstance(attrs["begin"][0], int)
        and isinstance(attrs["end"][0], int)
        and attrs.get("stride", [1])[0] == 1
    ):
        size = attrs["end"][0] - attrs["begin"][0]
        sh = list(input_nodes[0].shape)
        sh[0] = size
        return input_nodes[0].dtype, sh
    raise T2NErrorNotImplemented("complex slicing")


def _guess_out_dtype_and_shape_squeeze(input_nodes, attrs):
    sh = []
    for dim_idx, dim_value in enumerate(input_nodes[0].shape):
        if dim_idx in attrs["axes"]:
            continue
        sh.append(dim_value)
    return input_nodes[0].dtype, tuple(sh)


def _guess_out_dtype_and_shape_transpose(input_nodes, attrs):
    sh = list(input_nodes[0].shape)
    sh = [sh[ax] for ax in attrs["axes"]]
    return (input_nodes[0].dtype, sh)


def _guess_out_dtype_and_shape_unsqueeze(input_nodes, attrs):
    sh = list(input_nodes[0].shape)
    for ax in sorted(attrs["axes"]):
        sh.insert(ax, 1)
    return (input_nodes[0].dtype, sh)


def _guess_out_dtype_and_shape_n_elm_math(input_nodes, attrs):
    # keep biggest volume input
    sh = []
    max_vol = 0
    for inode in input_nodes:
        if (
            isinstance(inode, TensorVariable)
            and inode.shape
            and inode.volume
            and inode.volume > max_vol
        ):
            max_vol = inode.volume
            sh = list(inode.shape)
    return (input_nodes[0].dtype, sh)


def _guess_out_dtype_and_shape_tc_cast(input_nodes, attrs):
    return (
        str_to_torch_dtype(attrs["to"]),
        list(input_nodes[0].shape),
    )


def _guess_out_dtype_and_shape_tc_prod_reduce(input_nodes, attrs):
    return (torch.int64, tuple())


def _guess_out_dtype_and_shape(nnef_op_type: str, input_nodes, attrs):
    """Guess output dtype and shape from input nodes and attrs."""
    return {
        "tract_core_shape_of": _guess_out_dtype_and_shape_tc_shape_of,
        "slice": _guess_out_dtype_and_shape_slice,
        "squeeze": _guess_out_dtype_and_shape_squeeze,
        "tract_core_product_reduce": _guess_out_dtype_and_shape_tc_prod_reduce,
        "tract_core_cast": _guess_out_dtype_and_shape_tc_cast,
        "transopose": _guess_out_dtype_and_shape_transpose,
        "unsqueeze": _guess_out_dtype_and_shape_unsqueeze,
        "min": _guess_out_dtype_and_shape_n_elm_math,
        "max": _guess_out_dtype_and_shape_n_elm_math,
        "sub": _guess_out_dtype_and_shape_n_elm_math,
        "add": _guess_out_dtype_and_shape_n_elm_math,
        "div": _guess_out_dtype_and_shape_n_elm_math,
        "mul": _guess_out_dtype_and_shape_n_elm_math,
    }[nnef_op_type](input_nodes, attrs)


class OpHelper:
    def __init__(self, g, node, name_to_tensor, null_ref, inference_target):
        self.g = g
        self.node = node
        self.name_to_tensor = name_to_tensor
        self.null_ref = null_ref
        self.inference_target = inference_target

    def clone_with_new_node(self, node):
        return self.__class__(
            g=self.g,
            node=node,
            name_to_tensor=self.name_to_tensor,
            null_ref=self.null_ref,
            inference_target=self.inference_target,
        )

    def data_nodes_to_nnef_tensors(self, data_nodes):
        return [
            (
                self.get_or_add_tensor_variable_in_nnef(node=dn)
                if dn and not (isinstance(dn.data, str) and dn.data == "none")
                else self.null_ref
            )
            for dn in data_nodes
        ]

    def get_or_add_tensor_variable_in_nnef(self, node, **kwargs):
        return get_or_add_tensor_variable_in_nnef(
            g=self.g, node=node, name_to_tensor=self.name_to_tensor, **kwargs
        )

    def unary_output_op_without_attr(self, nnef_op_type, node):
        self.add_single_output_op_from_nnef_tensors(
            node,
            nnef_op_type=nnef_op_type,
            inputs=self.data_nodes_to_nnef_tensors(node.inputs),
        )

    def _implicits_input_casting(self, node, nnef_op_type, inputs):
        """Express implicit torch inputs casting with different dtype in NNEF.

        Those known implicit casting rules have been observed in math operators
        and tested empirically on PyTorch 2.2:

        - if Python constant then should align with
          the other inputs torch tensors dtype whathever it is
        - if inputs contains >=1 tensor but all with only 1 element
          => take first tensor dtype as ref dtype
        - if inputs contains >= 1 tensor with size > 1
          => take among tensor of size > 1 following implicit casting order
            (specific ranked dtype list DTYPES_EXPECTED_IMPLICIT_CAST_ORDER)


        For example:

        >>> torch.mul(1, torch.rand(10)).dtype
        torch.float32
        >>> torch.mul(torch.arange(10).to(torch.float64), torch.rand(10)).dtype
        torch.float64
        >>> torch.mul(torch.arange(10).to(torch.bool), torch.rand(10)).dtype
        torch.float32
        >>> torch.mul(torch.arange(10).to(torch.int16), torch.rand(10)).dtype
        torch.float32

        ...

        We need to express those implicit casting in graph to keep
        deterministic downstream inference engine
        (likely, most do not support such implicit casting, or use variations)

        Implementation: Instead of ensuring all the rules are good,
        we leverage the traced outputs dtype for some selected operators

        NOTE: For operators such as remainder rules implementation will
        be needed
        (but we do not yet support implicit dtype cast for such unusual op)

        """
        if (nnef_op_type in OPS_IMPLICIT_CAST_BINARY) and len(inputs) == 2:
            dtype_target = np.bool_
            for idx, inp in enumerate(inputs):
                if inp.dtype != dtype_target:
                    to_str = numpy_dtype_to_tract_str(dtype_target)
                    out = self.add_single_output_op_from_nnef_tensors(
                        node=node,
                        nnef_op_type="tract_core_cast",
                        inputs=inp,
                        attrs={"to": to_str},
                        force_full_output_tensor_name=f"{inp.name}_as_{to_str}",
                    )
                    inputs[idx] = out
            return tuple(inputs)

        if (
            nnef_op_type in OPS_IMPLICIT_CAST_CONSISTENT_INPS
            and len(inputs) > 1
            and len({_.dtype for _ in inputs}) > 1
        ):
            lowest_idx = np.inf
            for _ in inputs:
                idx = NP_DTYPES_EXPECTED_IMPLICIT_CAST_ORDER.index(_.dtype)
                lowest_idx = min(lowest_idx, idx)
            dtype_target = NP_DTYPES_EXPECTED_IMPLICIT_CAST_ORDER[lowest_idx]
            for idx, inp in enumerate(inputs):
                to_str = numpy_dtype_to_tract_str(dtype_target)
                out = self.add_single_output_op_from_nnef_tensors(
                    node=node,
                    nnef_op_type="tract_core_cast",
                    inputs=inp,
                    attrs={"to": to_str},
                    force_full_output_tensor_name=f"{inp.name}_as_{to_str}",
                )
                inputs[idx] = out
        if nnef_op_type not in OPS_IMPLICIT_CAST_BY_OUTPUT_DTYPE:
            return inputs
        final_dtype = TORCH_TO_NUMPY_DTYPE[node.outputs[0].dtype]
        inputs = list(inputs)
        for idx, inp in enumerate(inputs):
            if inp.dtype != final_dtype:
                to_str = numpy_dtype_to_tract_str(final_dtype)
                out = self.add_single_output_op_from_nnef_tensors(
                    node=node,
                    nnef_op_type="tract_core_cast",
                    inputs=inp,
                    attrs={"to": to_str},
                    force_full_output_tensor_name=f"{inp.name}_as_{to_str}",
                )
                inputs[idx] = out
        return tuple(inputs)

    def add_single_output_op_from_nnef_tensors(
        self,
        node,
        nnef_op_type: str,
        inputs,
        maybe_cast_align_tract: bool = True,
        **kwargs,
    ):
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        if (
            isinstance(self.inference_target, TractNNEF)
            and maybe_cast_align_tract
        ):
            inputs = self._implicits_input_casting(node, nnef_op_type, inputs)
        return add_single_output_op(
            node=node,
            nnef_op_type=nnef_op_type,
            inputs=inputs,
            g=self.g,
            name_to_tensor=self.name_to_tensor,
            **kwargs,
        )

    def cast_to_if_not_dtype_and_variable(
        self,
        node,
        nnef_tensor: NTensor,
        cast_to: np.dtype,
        suffix: str = "",
    ):
        return cast_to_if_not_dtype_and_variable(
            self.g,
            self.name_to_tensor,
            node,
            nnef_tensor,
            cast_to,
            suffix,
        )

    def cast_and_add_nnef_operation(self, **kwargs):
        return cast_and_add_nnef_operation(
            graph=self.g, name_to_tensor=self.name_to_tensor, **kwargs
        )

    def add_single_output_op_from_ir_datas(
        self,
        nnef_op_type: str,
        input_nodes: T.List[Data],
        output_tensor_name_suffix: str = "",
        force_full_output_tensor_name: str = "",
        reuse_if_name_exists: bool = False,
        **kwargs,
    ) -> TorchOp:
        """Use input_nodes Data instead of nnef.Tensor.

        Also nnefe
        """
        dtype, shape = _guess_out_dtype_and_shape(
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
        if (
            reuse_if_name_exists
            and self.name_to_tensor.get(new_output_data.export_name) is not None
        ):
            return new_node

        kwargs["nnef_op_type"] = nnef_op_type
        assert "inputs" not in kwargs
        kwargs["inputs"] = self.data_nodes_to_nnef_tensors(input_nodes)
        _ = self.add_single_output_op_from_nnef_tensors(node=new_node, **kwargs)
        return new_node


class SimpleOpChainer:
    def __init__(self, op_helper: OpHelper, input_data_nodes):
        self.op_helper = op_helper
        self.input_data_nodes = input_data_nodes

    def clone(self):
        return SimpleOpChainer(
            op_helper=self.op_helper,
            input_data_nodes=self.input_data_nodes[:],
        )

    @property
    def output_name(self):
        return self.input_data_nodes[0].export_name

    def chain(self, nnef_op_type, **kwargs):
        reuse_if_name_exists = (
            kwargs.pop("reuse_if_name_exists")
            if "reuse_if_name_exists" in kwargs
            else False
        )
        new_node = self.op_helper.add_single_output_op_from_ir_datas(
            nnef_op_type,
            input_nodes=self.input_data_nodes,
            reuse_if_name_exists=reuse_if_name_exists,
            **kwargs,
        )
        return SimpleOpChainer(
            self.op_helper.clone_with_new_node(new_node), new_node.outputs
        )

    def add_new_input_node(self, input_node, index=-1):
        assert isinstance(input_node, Data)
        new_data = self.input_data_nodes[:]
        new_data.insert(index, input_node)
        return SimpleOpChainer(self.op_helper, new_data)


def get_tract_dyn_axis_size_soc(
    op_helper, input_node, axis: int
) -> SimpleOpChainer:
    assert input_node.rank - np.abs(axis) >= 0, (
        f"{input_node.rank} - {np.abs(axis)}"
    )
    index_tensor_name = f"{input_node.export_name}_dim{axis}"
    soc = (
        SimpleOpChainer(
            op_helper=op_helper,
            input_data_nodes=[input_node],
        )
        .chain(
            "tract_core_shape_of",
            force_full_output_tensor_name=f"{input_node.export_name}_shape",
            reuse_if_name_exists=True,
        )
        .chain(
            "slice",
            attrs={
                "axes": [0],
                "begin": [axis],
                "end": [axis + 1],
                "stride": [1],
            },
            output_tensor_name_suffix=f"sliced{axis}",
            reuse_if_name_exists=True,
        )
        .chain(
            "squeeze",
            attrs={
                "axes": [0],
            },
            force_full_output_tensor_name=index_tensor_name,
            reuse_if_name_exists=True,
        )
    )
    return soc
