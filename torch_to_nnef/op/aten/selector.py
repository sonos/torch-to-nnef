import logging
from copy import copy

import nnef
import numpy as np
import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR, TORCH_TO_NUMPY_DTYPE
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
    pick_axis,
    pick_index_in_axis,
    resolve_attr_axis_size,
)
from torch_to_nnef.tensor import OpaqueTensorRef
from torch_to_nnef.torch_graph.ir_data import PythonConstant, TensorVariable

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


def _should_cast_for_select(inp, expected_np_dtype):
    """Return True when `select` inputs should be cast to the expected dtype."""
    return (inp.dtype != expected_np_dtype) or (
        np.prod(inp.shape) == 1
        and inp.dtype not in (np.float32, np.int64, np.bool_)
    )


def _nnef_cast(op_helper, node, tensor, to_tract_dtype: str, suffix: str = ""):
    name = f"{tensor.name}_cast_{to_tract_dtype}"
    if suffix:
        name = f"{name}_{suffix}"
    return op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type="tract_core_cast",
        inputs=tensor,
        attrs={"to": to_tract_dtype},
        force_full_output_tensor_name=name,
    )


def _resolve_slice_bound(
    bound_node, input_node, dim, default, has_dynamic_axes
):
    """Resolve a slice begin/end node to (value, has_concrete).

    `bound_node.data` is `None` when the trace stored a Python ``None``
    bound (e.g. ``x[:n]`` / ``x[k:]`` / ``x[:]``). In static-axes mode
    that means "use the default" (``0`` for begin, ``dim_size`` for
    end). In dynamic-axes mode it can refer to a runtime-computed
    value, so we keep the upstream identifier.
    """
    if bound_node.data is not None:
        if bound_node.data >= 0:
            return (
                pick_index_in_axis(
                    input_node, dim, bound_node.data, check_is_positive=False
                ),
                True,
            )
        return bound_node.data, False
    if has_dynamic_axes:
        return nnef.Identifier(bound_node.export_name), False
    return default, True


@OP_REGISTRY.register(torch_op_ids=["slice"])
def slice_(
    node,
    torch_graph,
    inference_target,
    op_helper,
    **kwargs,
):
    """Map PyTorch: 'aten:slice' to NNEF."""
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.version < "0.21.7"
    ):
        return tract_pre_0_21_7_slice(
            node,
            torch_graph,
            False,
            inference_target.has_dynamic_axes,
            op_helper,
            **kwargs,
        )
    input_node, axis_node, begin_node, end_node, stride_node = node.inputs
    # we assert for now all node except first are all constant
    dim = axis_node.data
    has_dynamic_axes = inference_target.has_dynamic_axes

    begin, begin_concrete = _resolve_slice_bound(
        begin_node,
        input_node,
        dim,
        default=0,
        has_dynamic_axes=has_dynamic_axes,
    )
    end, end_concrete = _resolve_slice_bound(
        end_node,
        input_node,
        dim,
        default=input_node.shape[dim],
        has_dynamic_axes=has_dynamic_axes,
    )
    has_concrete_values = begin_concrete and end_concrete

    fixed_dims_and_higher_end_slice = (
        isinstance(end, int)
        and end >= input_node.shape[dim]
        and not inference_target.has_dynamic_axes
    )
    if (
        begin == 0
        and stride_node.data == 1
        and (end == np.iinfo(np.int64).max or fixed_dims_and_higher_end_slice)
    ):
        LOGGER.debug("Slice is not needed since it have not effect")
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return []

    if has_concrete_values:
        assert begin < end

    if inference_target.has_dynamic_axes:
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(inference_target)
        # Cases with TractNNEF.version < 0.21.7 are handled upper
        attrs = {
            "axis": pick_axis(input_node, dim),
            "begin": begin,
            "stride": stride_node.data,
        }
        if end == np.iinfo(np.int64).max:
            # skip end value expression
            fragment_name = "dyn_slice_begin"
        else:
            fragment_name = "dyn_slice"
            attrs["end"] = end
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            fragment_name,
            inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            attrs=attrs,
            pass_quantization_params=True,
        )
        return [fragment_name, "within_bound_index"]

    # In static-axes mode `_resolve_slice_bound` always returns ints
    # (Identifiers only appear under dynamic axes, which returned above).
    dim_size = input_node.shape[dim]
    if end < 0:
        end += dim_size
    end = min(end, dim_size)
    if begin < 0:
        begin += dim_size
    begin = max(begin, 0)

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
        attrs={
            "axes": [pick_axis(input_node, dim)],
            "begin": [begin],
            "end": [end],
            "stride": [stride_node.data],
        },
        pass_quantization_params=True,
    )
    return ["tract_core"]


def tract_pre_0_21_7_slice(
    node,
    torch_graph,
    nnef_spec_strict,
    has_dynamic_axes,
    op_helper,
    **kwargs,
):
    """Old version of slice for tract version prior to 0.21.7."""
    LOGGER.debug("use legacy tract slice pre 0.21.7")
    input_node, axis_node, begin_node, end_node, stride_node = node.inputs

    # we assert for now all node except first are all constant
    dim = axis_node.data

    has_concrete_values = True
    # we use this since by default pytorch generate max int64 value for end
    if begin_node.data is not None:
        begin = pick_index_in_axis(
            input_node, dim, begin_node.data, check_is_positive=False
        )
    else:
        has_concrete_values = False
        begin = nnef.Identifier(begin_node.export_name)

    if end_node.data is not None:
        end = pick_index_in_axis(
            input_node, dim, end_node.data, check_is_positive=False
        )
    else:
        has_concrete_values = False
        end = nnef.Identifier(end_node.export_name)

    if (
        begin == 0
        and end in [input_node.shape[dim], np.iinfo(np.int64).max]
        and stride_node.data == 1
    ):
        LOGGER.debug("Slice is not needed since it have not effect")
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return []

    if has_concrete_values:
        assert begin < end

    if begin_node.data is not None and begin < 0:
        if has_dynamic_axes and not nnef_spec_strict:
            real_begin_tensor_name = (
                f"{node.outputs[0].export_name}_slice_begin"
            )
            soc = (
                get_tract_dyn_axis_size_soc(op_helper, input_node, dim)
                .add_new_input_node(begin_node)
                .chain(
                    "add",
                    force_full_output_tensor_name=f"{real_begin_tensor_name}_add",
                )
                .add_new_input_node(
                    PythonConstant(
                        name=f"{real_begin_tensor_name}_zero", data=0
                    )
                )
                .chain(
                    "max",
                    force_full_output_tensor_name=real_begin_tensor_name,
                )
                .chain(
                    "tract_core_cast",
                    attrs={"to": "TDim"},
                    force_full_output_tensor_name=f"{real_begin_tensor_name}_as_tdim",
                )
            )
            begin = nnef.Identifier(soc.output_name)
        else:
            begin = max(input_node.shape[dim] - begin, 0)

    if end_node.data is not None:
        if (
            has_dynamic_axes
            and not nnef_spec_strict
            and end >= input_node.shape[dim]
        ):
            # NOTE: since we can't ensure used dimension is not symbolic
            # we use `tract_core_shape_of`
            end = nnef.Identifier(
                get_tract_dyn_axis_size_soc(
                    op_helper, input_node, dim
                ).output_name
            )
        else:
            end = min(
                end,
                input_node.shape[dim],
            )

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
        attrs={
            "axes": [pick_axis(input_node, dim)],
            "begin": [begin],
            "end": [end],
            "stride": [stride_node.data],
        },
    )
    return ["tract_core"]


def _select_maybe_cast(op_helper, node, inputs, target_torch_dtype):
    decision = inputs[0]
    if decision.dtype != np.bool_:
        decision = _nnef_cast(op_helper, node, decision, "bool")
    casted_inputs = [decision]
    expected_dtype = TORCH_TO_NUMPY_DTYPE[target_torch_dtype]
    expected_dtype_tract = TORCH_DTYPE_TO_TRACT_STR[target_torch_dtype]
    for inp in inputs[1:]:
        if isinstance(
            op_helper.inference_target, TractNNEF
        ) and _should_cast_for_select(inp, expected_dtype):
            inp = _nnef_cast(op_helper, node, inp, expected_dtype_tract)
        casted_inputs.append(inp)
    return casted_inputs


@OP_REGISTRY.register()
def where(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:where' to NNEF."""
    (condition_node, true_value_node, false_value_node) = node.inputs

    inputs = op_helper.data_nodes_to_nnef_tensors(
        [condition_node, true_value_node, false_value_node]
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type="select",
        inputs=_select_maybe_cast(
            op_helper,
            node,
            inputs,
            target_torch_dtype=node.outputs[0].dtype,
        ),
    )


@OP_REGISTRY.register()
def narrow(node, op_helper, **kwargs):
    """Fancy slice made in PyTorch.

    torch.narrow(input, dim, start, length)

    Example:
    >>> import torch
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> torch.narrow(x, 0, 0, 2)
    tensor([[1, 2, 3],
            [4, 5, 6]])

    """
    input_node, axis_node, start_node, length_node = node.inputs

    # only ops subset implemented
    assert isinstance(axis_node.data, int)
    assert isinstance(start_node.data, int)
    assert isinstance(length_node.data, int)
    assert length_node.data > 0

    start_idx = pick_index_in_axis(input_node, axis_node.data, start_node.data)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
        attrs={
            "axes": [pick_axis(input_node, axis_node.data)],
            "begin": [start_idx],
            "end": [start_idx + length_node.data],
            "stride": [1],
        },
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def select(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:select' to NNEF."""
    input_node, axis_node, index_node = node.inputs
    begin = pick_index_in_axis(input_node, axis_node.data, index_node.data)
    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
        attrs={
            "axes": [pick_axis(input_node, axis_node.data)],
            "begin": [begin],
            "end": [
                pick_index_in_axis(
                    input_node, axis_node.data, index_node.data + 1
                )
            ],
            "stride": [1],
        },
        output_tensor_name_suffix="_select",
        pass_quantization_params=True,
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "squeeze",
        inputs=out,
        attrs={"axes": [pick_axis(input_node, axis_node.data)]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def gather(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:gather' to NNEF."""
    # gather
    input_node, dim_node, indexes_node, *_ = node.inputs
    # input_node = TensorVariable([?], shape=(169,4))
    # indexes_node = FixedTensorList (data=[TensorVariable([?], shape=(2401,))])
    if (
        indexes_node.data is not None
        and len(indexes_node.data) > 1
        and not all(
            (isinstance(idx, PythonConstant) and idx.data is None)
            for idx in indexes_node.data[:-1]
        )
    ):
        raise T2NErrorNotImplemented(
            "index dim>1 implemented only with all prior dim slice being [:]"
        )

    custom_fragments = []
    if isinstance(inference_target, TractNNEF):
        op_name = "tract_core_gather_elements"
        custom_fragments += ["tract_core"]
    else:
        raise T2NErrorNotImplemented()
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        op_name,
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            op_helper.get_or_add_tensor_variable_in_nnef(
                indexes_node,
            ),
        ],
        attrs={
            "axis": dim_node.data,
        },
        force_consistent_inputs_shapes=False,
    )
    return custom_fragments


@OP_REGISTRY.register()
def take_along_dim(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: `aten::take_along_dim(self, indices, dim?)` to NNEF.

    With an explicit `dim`, the op is `gather` along that axis. The
    `dim=None` form (flatten both tensors and do a 1-D take) needs an
    extra reshape pair; left for follow-up.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented()
    input_node, indices_node, dim_node = node.inputs
    if dim_node.data is None:
        raise T2NErrorNotImplemented(
            "take_along_dim with dim=None (flattened) not yet supported"
        )
    dim = pick_axis(input_node, dim_node.data)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather_elements",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            op_helper.get_or_add_tensor_variable_in_nnef(indices_node),
        ],
        attrs={"axis": dim},
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


@OP_REGISTRY.register(torch_op_ids=["index"])
def index_(node, op_helper, inference_target, **kwargs):
    """Translate `aten::index` to NNEF.

    Fragment gather<?>(.
        input: tensor<?>,                 # the tensor to gather from
        indices: tensor<integer>,         # the indices to gather at
        axis: integer = 0 )               # the axis to gather at
    -> ( output: tensor<?> )


    torch ir, in this case structure `indexes_node` with:
    a list of n values where n <= input_node rank
    each value is either a constant or a tensor.
    if the constant is None this means the full dimension

    """
    # gather
    input_node, indexes_node = node.inputs
    # input_node = TensorVariable([?], shape=(169,4))
    # indexes_node = FixedTensorList (data=[TensorVariable([?], shape=(2401,))])
    if len(indexes_node.data) > 1:
        # gather_elements
        len_idx_vars = len(
            [_ for _ in indexes_node.data if isinstance(_, TensorVariable)]
        )
        if len_idx_vars > 1:
            return _gather_nd(node, op_helper)

    custom_fragments = []
    attrs = {
        "axis": len(indexes_node.data) - 1,
    }
    if isinstance(inference_target, TractNNEF):
        op_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
        if isinstance(input_node.data, OpaqueTensorRef):
            attrs["datum_type"] = TORCH_DTYPE_TO_TRACT_STR[input_node.dtype]
    else:
        op_name = "gather"
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        op_name,
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            op_helper.get_or_add_tensor_variable_in_nnef(
                indexes_node.data[-1],
            ),
        ],
        attrs=attrs,
        force_consistent_inputs_shapes=False,
    )
    return custom_fragments


def _gather_nd(node, op_helper):
    input_node, indexes_node = node.inputs
    idx_nodes = list(indexes_node.data)

    # Cast each index to TDim (tract_core_gather_nd consumes TDim coords).
    tdim_refs = []
    for idx_node in idx_nodes:
        i_ref = op_helper.get_or_add_tensor_variable_in_nnef(idx_node)
        tdim_refs.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "tract_core_cast",
                inputs=[i_ref],
                attrs={"to": "TDim"},
                force_full_output_tensor_name=f"{i_ref.name}_as_tdim",
            )
        )

    # PyTorch advanced indexing broadcasts the index tensors against each
    # other before gathering, so `input[idx0, idx1, ...]` uses the common
    # broadcast shape of the idx tensors. tract_core_gather_nd instead needs
    # a single coordinate tensor whose leading shape is shared by every
    # index. When the idx tensors differ in shape (e.g. transformers 5.x
    # mask indexing with `[B,1,1,1]` and `[1,1,1,S]`), broadcast them to
    # their common shape first via an additive zero frame -- tract broadcasts
    # elementwise TDim adds natively, including symbolic/dynamic dims, so no
    # explicit repeats need resolving.
    if len({tuple(idx.shape) for idx in idx_nodes}) > 1:
        zero_frame = None
        for i, ref in enumerate(tdim_refs):
            zero = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "sub",
                inputs=[ref, ref],
                force_consistent_inputs_shapes=False,
                force_full_output_tensor_name=f"{ref.name}_gnd_zero_{i}",
            )
            if zero_frame is None:
                zero_frame = zero
            else:
                zero_frame = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "add",
                    inputs=[zero_frame, zero],
                    force_consistent_inputs_shapes=False,
                    force_full_output_tensor_name=f"{ref.name}_gnd_zframe_{i}",
                )
        tdim_refs = [
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "add",
                inputs=[ref, zero_frame],
                force_consistent_inputs_shapes=False,
                force_full_output_tensor_name=f"{ref.name}_gnd_bcast",
            )
            for ref in tdim_refs
        ]

    # Stack the (now shape-consistent) coordinates on a trailing axis so the
    # result is `[*broadcast_shape, num_indexed_dims]`, the layout expected by
    # tract_core_gather_nd (batch_dims=0).
    coord_axis = max(len(idx.shape) for idx in idx_nodes)
    coord_refs = [
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=[ref],
            attrs={"axes": [coord_axis]},
            force_full_output_tensor_name=f"{ref.name}_coord",
        )
        for ref in tdim_refs
    ]
    concat_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "concat",
        inputs=coord_refs,
        ensure_tuple=False,
        attrs={"axis": coord_axis},
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="indices_concat",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather_nd",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            concat_ref,
        ],
        attrs={
            "batch_dims": 0,
        },
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def embedding(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:embedding' to NNEF."""
    (
        weight_node,
        indices_node,
        _,  # padding_idx_node
        _,  # scale_grad_by_freq_node
        _,  # sparse_node
    ) = node.inputs

    custom_fragments = []
    attrs = {"axis": 0}
    if isinstance(inference_target, TractNNEF):
        op_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
        if isinstance(weight_node.data, OpaqueTensorRef):
            attrs["datum_type"] = TORCH_DTYPE_TO_TRACT_STR[weight_node.dtype]
    else:
        op_name = "gather"

    apply_squeeze = indices_node.rank == 1
    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        op_name,
        inputs=op_helper.data_nodes_to_nnef_tensors(
            [weight_node, indices_node]
        ),
        attrs=attrs,
        output_tensor_name_suffix="pre_squeeze" if apply_squeeze else "",
    )
    if apply_squeeze:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "squeeze",
            inputs=out,
            attrs={"axes": [0]},
        )
    return custom_fragments


_EMBEDDING_BAG_MODE_TO_REDUCE = {
    0: "sum_reduce",
    1: "mean_reduce",
    2: "max_reduce",
}


def _embedding_bag_validate(node):
    """Validate `aten::embedding_bag` inputs and pull out the args.

    Raises `T2NErrorNotImplemented` for the cases the current emitter
    can't handle (mode out of {sum, mean, max}, `per_sample_weights`,
    `include_last_offset`, explicit `padding_idx`, dynamic offsets).
    Returns `(weight_node, indices_node, offsets_list, mode)`.
    """
    if len(node.inputs) < 5:
        raise T2NErrorNotImplemented(
            f"embedding_bag with {len(node.inputs)} inputs"
        )
    weight_node = node.inputs[0]
    indices_node = node.inputs[1]
    offsets_node = node.inputs[2]
    mode_node = node.inputs[4]
    per_sample_weights_node = node.inputs[6] if len(node.inputs) > 6 else None
    include_last_offset_node = node.inputs[7] if len(node.inputs) > 7 else None
    padding_idx_node = node.inputs[8] if len(node.inputs) > 8 else None

    mode = int(mode_node.data)
    if mode not in _EMBEDDING_BAG_MODE_TO_REDUCE:
        raise T2NErrorNotImplemented(f"embedding_bag mode={mode}")
    if per_sample_weights_node is not None and not (
        isinstance(per_sample_weights_node, PythonConstant)
        and per_sample_weights_node.data is None
    ):
        raise T2NErrorNotImplemented(
            "embedding_bag with per_sample_weights not supported"
        )
    if (
        include_last_offset_node is not None
        and isinstance(include_last_offset_node, PythonConstant)
        and bool(include_last_offset_node.data)
    ):
        raise T2NErrorNotImplemented(
            "embedding_bag with include_last_offset=True not supported"
        )
    if padding_idx_node is not None and not (
        isinstance(padding_idx_node, PythonConstant)
        and padding_idx_node.data in (None, -1)
    ):
        raise T2NErrorNotImplemented(
            "embedding_bag with explicit padding_idx not supported"
        )

    offsets_data = offsets_node.data
    if offsets_data is None:
        raise T2NErrorNotImplemented(
            "embedding_bag requires statically-known offsets"
        )
    if hasattr(offsets_data, "tolist"):
        offsets_data = offsets_data.tolist()
    return weight_node, indices_node, [int(x) for x in offsets_data], mode


def _embedding_bag_uniform_path(
    g,
    name_to_tensor,
    base,
    np_dtype,
    emb,
    out_ref,
    n_bags,
    bag_size,
    d_dim,
    reduce_op,
):
    """Equal-bag-size shortcut: reshape + reduce on axis 1 + squeeze."""
    reshaped = NTensor(
        g,
        f"{base}_eb_reshape",
        dtype=np_dtype,
        shape=(n_bags, bag_size, d_dim),
    )
    name_to_tensor[reshaped.name] = reshaped
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="reshape",
        name=f"{reshaped.name}_op",
        inputs=(emb,),
        outputs=(reshaped,),
        attribs={"shape": [n_bags, bag_size, d_dim]},
    )
    kd = NTensor(g, f"{base}_eb_kd", dtype=np_dtype, shape=(n_bags, 1, d_dim))
    name_to_tensor[kd.name] = kd
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=reduce_op,
        name=f"{kd.name}_op",
        inputs=(reshaped,),
        outputs=(kd,),
        attribs={"axes": [1]},
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="squeeze",
        name=f"{base}_eb_squeeze",
        inputs=(kd,),
        outputs=(out_ref,),
        attribs={"axes": [1]},
    )


def _embedding_bag_variable_path(
    g,
    name_to_tensor,
    base,
    np_dtype,
    emb,
    out_ref,
    boundaries,
    bag_sizes,
    d_dim,
    reduce_op,
):
    """Variable-bag-size path: per-bag slice + reduce + concat."""
    if any(s <= 0 for s in bag_sizes):
        raise T2NErrorNotImplemented(
            f"embedding_bag with empty bag (sizes={bag_sizes}); torch "
            "fills empty bags with zeros which would need a different "
            "emit path"
        )
    per_bag = []
    for b, (start, size) in enumerate(
        zip(boundaries[:-1], bag_sizes, strict=True)
    ):
        slc = NTensor(
            g, f"{base}_eb_b{b}_slice", dtype=np_dtype, shape=(size, d_dim)
        )
        name_to_tensor[slc.name] = slc
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            name=f"{slc.name}_op",
            inputs=(emb,),
            outputs=(slc,),
            attribs={
                "axes": [0],
                "begin": [start],
                "end": [start + size],
                "stride": [1],
            },
        )
        reduced = NTensor(
            g, f"{base}_eb_b{b}_kd", dtype=np_dtype, shape=(1, d_dim)
        )
        name_to_tensor[reduced.name] = reduced
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type=reduce_op,
            name=f"{reduced.name}_op",
            inputs=(slc,),
            outputs=(reduced,),
            attribs={"axes": [0]},
        )
        per_bag.append(reduced)
    # `concat` takes a list-typed `values` arg in NNEF -- pass as a
    # Python list (not tuple) so the serializer writes `[...]`.
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="concat",
        name=f"{base}_eb_concat",
        inputs=list(per_bag),
        outputs=(out_ref,),
        attribs={"axis": 0},
        force_consistent_inputs_shapes=False,
    )


@OP_REGISTRY.register()
def embedding_bag(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
):
    """Map PyTorch: 'aten:embedding_bag' to NNEF.

    `embedding_bag(weight, indices, offsets, scale_grad, mode, sparse,
    per_sample_weights, include_last_offset, padding_idx)` returns a
    4-tuple in torch (`output, offset2bag, bag_size, max_indices`); we
    only emit the first output (the bag-reduced embeddings). The
    other three are gradient bookkeeping and typically aren't
    consumed in inference traces.

    Decomposition (statically-known offsets):

      emb = tract_core_gather(weight, indices, axis=0)  # (K, D)
      for each bag b: bag_out_b = reduce(emb[offsets[b]:end_b], axis=0)
      output = concat(bag_outs, axis=0)  # (B, D)

    Equal bag sizes collapse to a single `reshape + reduce + squeeze`.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    weight_node, indices_node, offsets_list, mode = _embedding_bag_validate(
        node
    )
    k = int(indices_node.shape[0])
    boundaries = [*offsets_list, k]
    bag_sizes = [
        boundaries[i + 1] - boundaries[i] for i in range(len(offsets_list))
    ]
    n_bags = len(bag_sizes)
    d_dim = int(weight_node.shape[1])

    onode = node.outputs[0]
    base = onode.export_name
    np_dtype = onode.np_dtype
    reduce_op = _EMBEDDING_BAG_MODE_TO_REDUCE[mode]

    emb = NTensor(g, f"{base}_eb_emb", dtype=np_dtype, shape=(k, d_dim))
    name_to_tensor[f"{base}_eb_emb"] = emb
    weight_ref = get_or_add_tensor_variable_in_nnef(
        g, weight_node, name_to_tensor
    )
    indices_ref = get_or_add_tensor_variable_in_nnef(
        g, indices_node, name_to_tensor
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="tract_core_gather",
        name=f"{emb.name}_op",
        inputs=(weight_ref, indices_ref),
        outputs=(emb,),
        attribs={"axis": 0},
        force_consistent_inputs_shapes=False,
    )

    out_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        onode, prevent_variable=True
    )

    if len(set(bag_sizes)) == 1 and bag_sizes[0] > 0:
        _embedding_bag_uniform_path(
            g,
            name_to_tensor,
            base,
            np_dtype,
            emb,
            out_ref,
            n_bags,
            bag_sizes[0],
            d_dim,
            reduce_op,
        )
    else:
        _embedding_bag_variable_path(
            g,
            name_to_tensor,
            base,
            np_dtype,
            emb,
            out_ref,
            boundaries,
            bag_sizes,
            d_dim,
            reduce_op,
        )
    return ["tract_core"]


@OP_REGISTRY.register()
def masked_fill(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:masked_fill' to NNEF."""
    input_node, mask_node, value_node = node.inputs

    false_value_node = input_node
    false_nnef_tensor = op_helper.get_or_add_tensor_variable_in_nnef(
        false_value_node
    )
    # value is always a float according to torch spec
    true_value_node = value_node.into_tensor_variable()
    if true_value_node.data is not None:
        node.outputs[0].dtype = false_value_node.dtype  # TODO: understand
        target_dtype = node.inputs[0].dtype
        true_value_node.set_data(
            true_value_node.data.to(target_dtype), force_dtype=True
        )
    if inference_target.has_dynamic_axes:
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(inference_target)
        # repeats on non const not working in tract<=0.21.3
        # so while correct graph notation, tract will fail
        out = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_shape_of",
            inputs=false_nnef_tensor,
            output_tensor_name_suffix="shape_of_false",
        )

        # force rank to be the same
        true_value_node.set_data(
            true_value_node.data.repeat(*([1] * false_value_node.rank)),
            force_shape=True,
        )
        inp = op_helper.get_or_add_tensor_variable_in_nnef(
            true_value_node, name_suffix="true_scalar"
        )
        if inp.dtype == np.int64 and node.outputs[0].dtype == torch.int64:
            inp = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "tract_core_cast",
                inputs=inp,
                attrs={"to": TORCH_DTYPE_TO_TRACT_STR[torch.int64]},
                output_tensor_name_suffix="true_expanded_casted",
            )

        true_nnef_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tile",
            inputs=inp,
            attrs={"repeats": nnef.Identifier(str(out.name))},
            output_tensor_name_suffix="true_expanded",
        )
    else:
        # Static expansion
        true_value_node.shape = false_value_node.shape
        true_value_node.set_data(
            true_value_node.data.repeat(false_value_node.shape),
            force_shape=True,
        )
        true_value_node.dtype = false_value_node.dtype
        true_nnef_tensor = op_helper.get_or_add_tensor_variable_in_nnef(
            true_value_node
        )

    # tract need float where ?
    # mask_node.set_data(mask_node.data.float())
    # mask_node.dtype = mask_node.data.dtype
    condition_node = mask_node

    decision = op_helper.get_or_add_tensor_variable_in_nnef(condition_node)
    assert true_nnef_tensor.dtype == false_nnef_tensor.dtype, (
        f"masked_fill true and false branch must have the same dtype, got "
        f"{true_nnef_tensor.dtype} and {false_nnef_tensor.dtype}"
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type="select",
        inputs=_select_maybe_cast(
            op_helper,
            node,
            [
                decision,
                true_nnef_tensor,
                false_nnef_tensor,
            ],
            target_torch_dtype=node.outputs[0].dtype,
        ),
    )


@OP_REGISTRY.register()
def argsort(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:argsort' to NNEF."""
    assert isinstance(inference_target, TractNNEF), (
        "not supported by Khronos spec"
    )
    input_node, dim_node, descending_node = node.inputs
    input_nnef = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    assert isinstance(descending_node.data, bool), descending_node
    assert isinstance(dim_node, PythonConstant), dim_node
    assert isinstance(dim_node.data, int), dim_node
    dim = pick_axis(input_node, dim_node.data)
    if inference_target.has_dynamic_axes:
        # Centralized dynamic axis extraction
        get_tract_dyn_axis_size_soc(op_helper, input_node, dim)
        dim_size = nnef.Identifier(f"{input_node.export_name}_dim{dim}")
    else:
        dim_size = input_nnef.shape[dim]

    output_tensors = []
    out_node = copy(node.outputs[0])
    out_node.dtype = input_node.dtype
    out = op_helper.get_or_add_tensor_variable_in_nnef(
        out_node,
        name_suffix="values",
        prevent_variable=True,
    )
    output_tensors.append(out)
    out = op_helper.get_or_add_tensor_variable_in_nnef(
        node.outputs[0],
        prevent_variable=True,
    )
    output_tensors.append(out)

    cast_and_add_nnef_operation(
        name_to_tensor=op_helper.name_to_tensor,
        graph=op_helper.g,
        type="tract_core_topk",
        inputs=(input_nnef,),
        outputs=tuple(output_tensors),
        attribs={"k": dim_size, "axis": dim, "largest": descending_node.data},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def sort(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:sort' to NNEF."""
    assert isinstance(inference_target, TractNNEF), (
        "not supported by Khronos spec"
    )
    input_node, dim_node, descending_node = node.inputs
    input_nnef = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    assert isinstance(descending_node.data, bool), descending_node
    assert isinstance(dim_node, PythonConstant), dim_node
    assert isinstance(dim_node.data, int), dim_node
    dim = pick_axis(input_node, dim_node.data)
    if inference_target.has_dynamic_axes:
        # Centralized dynamic axis extraction
        get_tract_dyn_axis_size_soc(op_helper, input_node, dim)
        dim_size = nnef.Identifier(f"{input_node.export_name}_dim{dim}")
    else:
        dim_size = input_nnef.shape[dim]

    output_tensors = [
        op_helper.get_or_add_tensor_variable_in_nnef(
            node.outputs[_],
            prevent_variable=True,
        )
        for _ in range(2)
    ]

    cast_and_add_nnef_operation(
        name_to_tensor=op_helper.name_to_tensor,
        graph=op_helper.g,
        type="tract_core_topk",
        inputs=(input_nnef,),
        outputs=tuple(output_tensors),
        attribs={"k": dim_size, "axis": dim, "largest": descending_node.data},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def topk(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:topk' to NNEF."""
    assert isinstance(inference_target, TractNNEF), (
        "not supported by Khronos spec"
    )
    input_node, k_node, dim_node, largest_node, sorted_node = node.inputs
    input_nnef = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    assert isinstance(largest_node.data, bool), largest_node
    assert isinstance(dim_node, PythonConstant), dim_node
    assert isinstance(dim_node.data, int), dim_node
    assert isinstance(k_node.data, int), k_node
    if not sorted_node.data:
        raise T2NErrorNotImplemented("non sorted topk not implemented")
    dim = pick_axis(input_node, dim_node.data)

    output_tensors = [
        op_helper.get_or_add_tensor_variable_in_nnef(
            node.outputs[_],
            prevent_variable=True,
        )
        for _ in range(2)
    ]

    cast_and_add_nnef_operation(
        name_to_tensor=op_helper.name_to_tensor,
        graph=op_helper.g,
        type="tract_core_topk",
        inputs=(input_nnef,),
        outputs=tuple(output_tensors),
        attribs={"k": k_node.data, "axis": dim, "largest": largest_node.data},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def index_select(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:index_select' to NNEF."""
    input_node, dim_node, indexes_node = node.inputs
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    attrs = {
        "axis": dim_node.data,
    }
    if isinstance(input_node.data, OpaqueTensorRef):
        attrs["datum_type"] = TORCH_DTYPE_TO_TRACT_STR[input_node.dtype]
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            op_helper.get_or_add_tensor_variable_in_nnef(
                indexes_node,
            ),
        ],
        attrs=attrs,
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


def _emit_scatter_elements(
    node, op_helper, input_node, dim, indexes_node, src_node, reduction
):
    """Common emitter for the scatter family on the TractNNEF path.

    All variants lower to `tract_core_scatter_elements`; the only thing
    that varies is the `reduction` attribute and which torch overload's
    inputs we pulled apart upstream.
    """
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_scatter_elements",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            op_helper.get_or_add_tensor_variable_in_nnef(indexes_node),
            op_helper.get_or_add_tensor_variable_in_nnef(src_node),
        ],
        attrs={"axis": dim, "reduction": reduction},
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def scatter(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:scatter' to NNEF."""
    input_node, dim_node, indexes_node, src_node = node.inputs
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    return _emit_scatter_elements(
        node,
        op_helper,
        input_node,
        dim_node.data,
        indexes_node,
        src_node,
        reduction="none",
    )


_SCATTER_REDUCTION_MIN_TRACT_VERSION = "0.23.0-dev.4"


def _check_scatter_reduction_supported(inference_target):
    """Tract gained the NNEF `reduction` attribute in 0.23.0-dev.4 (#2109).

    Earlier releases (incl. the published 0.22.1) silently ignore the
    attribute and run the default overwrite path, which silently gives
    wrong answers. Refuse to emit when targeting an older runtime.
    """
    if inference_target.version < _SCATTER_REDUCTION_MIN_TRACT_VERSION:
        raise T2NErrorNotImplemented(
            f"scatter with reduction needs tract >= "
            f"{_SCATTER_REDUCTION_MIN_TRACT_VERSION}; got "
            f"{inference_target.version}"
        )


@OP_REGISTRY.register()
def scatter_add(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:scatter_add' to NNEF.

    `scatter_add(input, dim, index, src)` accumulates `src` values into
    `input` at positions selected by `index` along `dim`. Equivalent to
    `tract_core_scatter_elements` with `reduction="add"`.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    _check_scatter_reduction_supported(inference_target)
    input_node, dim_node, indexes_node, src_node = node.inputs
    return _emit_scatter_elements(
        node,
        op_helper,
        input_node,
        dim_node.data,
        indexes_node,
        src_node,
        reduction="add",
    )


_SCATTER_REDUCE_TORCH_TO_TRACT = {
    "sum": "add",
    "prod": "mul",
    "amax": "max",
    "amin": "min",
}


@OP_REGISTRY.register()
def scatter_reduce(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:scatter_reduce' to NNEF.

    Maps torch's reduce mode to tract's `ScatterReduction`. `mean` is
    not in tract's set ({add, mul, min, max}) so we raise; the same goes
    for `include_self=False`, since tract always reduces against the
    pre-existing destination value.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    _check_scatter_reduction_supported(inference_target)
    (
        input_node,
        dim_node,
        indexes_node,
        src_node,
        reduce_node,
        include_self_node,
    ) = node.inputs
    reduce_str = reduce_node.data
    if reduce_str not in _SCATTER_REDUCE_TORCH_TO_TRACT:
        raise T2NErrorNotImplemented(
            f"scatter_reduce: reduce='{reduce_str}' not supported "
            "(tract has add/mul/min/max only; no mean)"
        )
    if include_self_node.data is False:
        raise T2NErrorNotImplemented(
            "scatter_reduce: include_self=False not supported by tract"
        )
    return _emit_scatter_elements(
        node,
        op_helper,
        input_node,
        dim_node.data,
        indexes_node,
        src_node,
        reduction=_SCATTER_REDUCE_TORCH_TO_TRACT[reduce_str],
    )


@OP_REGISTRY.register()
def select_scatter(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:select_scatter' to NNEF.

    `out = input.clone(); out.select(dim, index).copy_(src)` -- the
    functional select-write. Decomposes to `slice` + `unsqueeze` +
    `concat`: replace the (size-1) slab at position `index` along `dim`
    with `src` (which has rank `input.rank - 1`). Static-shape only.
    """
    input_node, src_node, dim_node, index_node = node.inputs
    dim = pick_axis(input_node, dim_node.data)
    dim_size = input_node.shape[dim]
    if not isinstance(dim_size, int):
        raise T2NErrorNotImplemented(
            f"select_scatter on dynamic dim {dim} not yet supported"
        )

    index = index_node.data
    if index < 0:
        index += dim_size

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    src_unsq = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(src_node),
        attrs={"axes": [dim]},
        output_tensor_name_suffix="_ss_src_unsq",
    )

    parts = []
    if index > 0:
        parts.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "slice",
                inputs=inp_ref,
                attrs={"axes": [dim], "begin": [0], "end": [index]},
                output_tensor_name_suffix="_ss_left",
            )
        )
    parts.append(src_unsq)
    if index + 1 < dim_size:
        parts.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "slice",
                inputs=inp_ref,
                attrs={
                    "axes": [dim],
                    "begin": [index + 1],
                    "end": [dim_size],
                },
                output_tensor_name_suffix="_ss_right",
            )
        )

    if len(parts) == 1:
        # Whole-axis replacement (dim_size == 1): src already covers the
        # full output. Emit a no-op reshape so the output gets named.
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "reshape",
            inputs=parts[0],
            attrs={"shape": list(input_node.shape)},
        )
    else:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "concat",
            inputs=parts,
            ensure_tuple=False,
            attrs={"axis": dim},
            force_consistent_inputs_shapes=False,
        )
    return []


@OP_REGISTRY.register()
def slice_scatter(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:slice_scatter' to NNEF.

    `out = input.clone(); out[..., start:end:step, ...] = src` -- the
    functional slice-write. Decomposes to `slice` + `concat`. `step != 1`
    is rejected (would need an interleave path). Static-shape only.
    """
    (
        input_node,
        src_node,
        dim_node,
        start_node,
        end_node,
        step_node,
    ) = node.inputs
    if step_node.data != 1:
        raise T2NErrorNotImplemented(
            f"slice_scatter step={step_node.data} (only step=1 supported)"
        )
    dim = pick_axis(input_node, dim_node.data)
    dim_size = input_node.shape[dim]
    if not isinstance(dim_size, int):
        raise T2NErrorNotImplemented(
            f"slice_scatter on dynamic dim {dim} not yet supported"
        )

    start = start_node.data if start_node.data is not None else 0
    end = end_node.data if end_node.data is not None else dim_size
    if start < 0:
        start += dim_size
    if end < 0:
        end += dim_size
    start = max(start, 0)
    end = min(end, dim_size)

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    src_ref = op_helper.get_or_add_tensor_variable_in_nnef(src_node)

    parts = []
    if start > 0:
        parts.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "slice",
                inputs=inp_ref,
                attrs={"axes": [dim], "begin": [0], "end": [start]},
                output_tensor_name_suffix="_sls_left",
            )
        )
    parts.append(src_ref)
    if end < dim_size:
        parts.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "slice",
                inputs=inp_ref,
                attrs={"axes": [dim], "begin": [end], "end": [dim_size]},
                output_tensor_name_suffix="_sls_right",
            )
        )

    if len(parts) == 1:
        # Full-axis replacement (start == 0 and end == dim_size). Emit a
        # no-op reshape so the output node still gets registered.
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "reshape",
            inputs=parts[0],
            attrs={"shape": list(input_node.shape)},
        )
    else:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "concat",
            inputs=parts,
            ensure_tuple=False,
            attrs={"axis": dim},
            force_consistent_inputs_shapes=False,
        )
    return []


@OP_REGISTRY.register()
def _pack_padded_sequence(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:_pack_padded_sequence' to NNEF."""
    raise T2NErrorNotImplemented(
        "support for .pack_padded_sequence not added in tract yet"
    )
    # input_node, lengths_node, batch_first_node = node.inputs[:3]
    # opacked_node, obatch_node = node.outputs
    # return ["pack_padded_sequence"]


def _diagonal_einsum_expr(rank: int) -> str:
    """Build the einsum expression `<leading>ii-><leading>i` for given rank.

    `tract_core_einsum` does not accept ellipsis in the version range we
    target, so we materialize concrete labels per rank. Up to rank 10
    is supported (8 leading labels + the two diagonal axes).
    """
    if rank < 2:
        raise T2NErrorNotImplemented(f"diagonal needs rank>=2, got {rank}")
    leading = "abcdefgh"[: rank - 2]
    if len(leading) < rank - 2:
        raise T2NErrorNotImplemented(
            f"diagonal rank {rank} exceeds einsum label budget"
        )
    return f"{leading}ii->{leading}i"


def _emit_eye(op_helper, node, n: int, m: int, dtype, suffix: str):
    """Emit a NNEF `eye(n, m)` fragment call and return its tensor ref."""
    return op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "eye",
        inputs=[],
        attrs={
            "n": int(n),
            "m": int(m),
            "to": TORCH_DTYPE_TO_TRACT_STR[dtype],
        },
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix=suffix,
    )


@OP_REGISTRY.register()
def diag(node, op_helper, inference_target, **kwargs):
    """Map `aten::diag(input, offset)` to NNEF (offset=0 only).

    - 1-D `(N,)` input: build `(N, N)` matrix with `input` on the
      diagonal: `input.unsqueeze(1) * eye(N, N)`.
    - 2-D square `(N, N)` input: extract the diagonal by
      `sum_reduce(input * eye(N, N), axes=[1])` then squeeze. Returns
      `(N,)`. Non-square 2-D extraction would need a strided slice on
      the flattened layout; raise for now.

    `offset != 0` is rejected; the existing `aten::diagonal` handler
    covers the off-diagonal case for 2-D inputs.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "diag requires `tract_core_range` via the `eye` fragment "
            "(TractNNEF target)"
        )
    input_node, offset_node = node.inputs[:2]
    offset = int(getattr(offset_node, "data", 0) or 0)
    if offset != 0:
        raise T2NErrorNotImplemented(
            f"diag: offset != 0 not supported (got {offset}); use "
            "`torch.diagonal` for non-zero offsets on 2-D inputs."
        )
    if input_node.rank == 1:
        n = input_node.shape[0]
        if not isinstance(n, int):
            raise T2NErrorNotImplemented(
                f"diag (1-D -> 2-D): dynamic length not supported (got {n})"
            )
        inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
        unsq = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=inp,
            attrs={"axes": [1]},
            output_tensor_name_suffix="_diag_unsq",
        )
        eye_t = _emit_eye(
            op_helper,
            node,
            n,
            n,
            input_node.dtype or torch.float32,
            "_diag_eye",
        )
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "mul",
            inputs=[unsq, eye_t],
            force_consistent_inputs_shapes=False,
        )
        return ["eye"]
    if input_node.rank == 2:
        m, n = input_node.shape
        if not (isinstance(m, int) and isinstance(n, int)):
            raise T2NErrorNotImplemented(
                "diag (2-D -> 1-D): dynamic shapes not supported "
                f"(got {input_node.shape})"
            )
        if m != n:
            raise T2NErrorNotImplemented(
                f"diag (2-D -> 1-D): non-square inputs not supported "
                f"(got {m}x{n}); the strided-slice path is left as a "
                "follow-up."
            )
        inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
        eye_t = _emit_eye(
            op_helper,
            node,
            m,
            n,
            input_node.dtype or torch.float32,
            "_diag_eye",
        )
        masked = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "mul",
            inputs=[inp, eye_t],
            output_tensor_name_suffix="_diag_mask",
            force_consistent_inputs_shapes=False,
        )
        reduced = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "sum_reduce",
            inputs=masked,
            attrs={"axes": [1]},
            output_tensor_name_suffix="_diag_reduce",
        )
        op_helper.add_single_output_op_from_nnef_tensors(
            node, "squeeze", inputs=reduced, attrs={"axes": [1]}
        )
        return ["eye"]
    raise T2NErrorNotImplemented(
        f"diag: only rank 1 or rank 2 supported (got rank {input_node.rank})"
    )


@OP_REGISTRY.register()
def diagflat(node, op_helper, inference_target, **kwargs):
    """Map `aten::diagflat(input, offset)` to NNEF.

    Flatten `input` to 1-D then build the square diagonal matrix:
    `diagflat(x) = diag(x.flatten())`. Reuses the `diag` 1-D->2-D path
    (eye + unsqueeze + mul). `offset != 0` is rejected.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "diagflat requires the `eye` fragment (TractNNEF target)"
        )
    input_node, offset_node = node.inputs[:2]
    offset = int(getattr(offset_node, "data", 0) or 0)
    if offset != 0:
        raise T2NErrorNotImplemented(
            f"diagflat: offset != 0 not supported (got {offset})"
        )
    if not all(isinstance(d, int) for d in input_node.shape):
        raise T2NErrorNotImplemented(
            f"diagflat: dynamic shape not supported (got {input_node.shape})"
        )
    n = 1
    for d in input_node.shape:
        n *= int(d)
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    flat = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=inp,
        attrs={
            "dtype": node.outputs[0].np_dtype,
            "shape": [n, 1],
        },
        output_tensor_name_suffix="_diagflat_flat",
    )
    eye_t = _emit_eye(
        op_helper,
        node,
        n,
        n,
        input_node.dtype or torch.float32,
        "_diagflat_eye",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=[flat, eye_t],
        force_consistent_inputs_shapes=False,
    )
    return ["eye"]


@OP_REGISTRY.register()
def diag_embed(node, op_helper, inference_target, **kwargs):
    """Map `aten::diag_embed(input, offset, dim1, dim2)` to NNEF.

    For input shape `(..., D)` and the default `dim1=-2`, `dim2=-1`:
    produce `(..., D, D)` with each leading slice on the diagonal of
    the trailing 2x2 block. `offset != 0` or non-default `dim1`/`dim2`
    are rejected.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "diag_embed requires the `eye` fragment (TractNNEF target)"
        )
    input_node, offset_node, dim1_node, dim2_node = node.inputs[:4]
    offset = int(getattr(offset_node, "data", 0) or 0)
    if offset != 0:
        raise T2NErrorNotImplemented(
            f"diag_embed: offset != 0 not supported (got {offset})"
        )
    rank = input_node.rank
    # `dim1` / `dim2` index the OUTPUT (rank + 1) axes, not the input.
    # The default (-2, -1) refers to the trailing 2x2 block of the
    # output. Reject anything other than the default layout for now.
    raw_d1 = dim1_node.data if dim1_node.data is not None else -2
    raw_d2 = dim2_node.data if dim2_node.data is not None else -1
    out_rank = rank + 1
    d1 = raw_d1 + out_rank if raw_d1 < 0 else raw_d1
    d2 = raw_d2 + out_rank if raw_d2 < 0 else raw_d2
    if (d1, d2) != (out_rank - 2, out_rank - 1):
        raise T2NErrorNotImplemented(
            "diag_embed: only the default (dim1=-2, dim2=-1) layout "
            f"is supported; got dim1={raw_d1}, dim2={raw_d2}"
        )
    d = input_node.shape[-1]
    if not isinstance(d, int):
        raise T2NErrorNotImplemented(
            f"diag_embed: dynamic last-dim size not supported (got {d})"
        )
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    unsq = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=inp,
        attrs={"axes": [rank]},
        output_tensor_name_suffix="_diag_embed_unsq",
    )
    eye_t = _emit_eye(
        op_helper,
        node,
        d,
        d,
        input_node.dtype or torch.float32,
        "_diag_embed_eye",
    )
    # Eye is (D, D); the unsqueezed input is (..., D, 1). Prepend
    # leading singletons so eye sits at the same rank as the unsqueezed
    # input, then broadcast-mul over the leading dims.
    leading = rank - 1
    if leading > 0:
        eye_t = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=eye_t,
            attrs={"axes": list(range(leading))},
            output_tensor_name_suffix="_diag_embed_eye_unsq",
        )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=[unsq, eye_t],
        force_consistent_inputs_shapes=False,
    )
    return ["eye"]


@OP_REGISTRY.register(torch_op_ids=["linalg_diagonal"])
def linalg_diagonal(node, op_helper, inference_target, **kwargs):
    """`aten::linalg_diagonal` is the alternate spelling of `diagonal`."""
    return diagonal(node, op_helper, inference_target, **kwargs)


@OP_REGISTRY.register()
def diagonal(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:diagonal' to NNEF (tract path).

    Strategy: bring (dim1, dim2) to the last two axes via `transpose`,
    slice each axis to the diagonal window, then evaluate
    `<leading>ii-><leading>i` with `tract_core_einsum`. The slice begin
    on each axis encodes the offset:

        begin_a1 = max(0, -offset)
        begin_a2 = max(0,  offset)
        L        = min(s1 - begin_a1, s2 - begin_a2)

    `offset` is interpreted in the user's `(dim1, dim2)` order; when we
    sort axes to `a1 < a2` the sign flips. Empty diagonals (`L <= 0`)
    are left as `T2NErrorNotImplemented` since static zero-extent axes
    are awkward to represent.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "diagonal requires `tract_core_einsum` (TractNNEF target)"
        )
    input_node, offset_node, dim1_node, dim2_node = node.inputs
    offset = offset_node.data
    rank = input_node.rank
    a1 = pick_axis(input_node, dim1_node.data)
    a2 = pick_axis(input_node, dim2_node.data)
    if a1 == a2:
        raise T2NErrorNotImplemented(f"diagonal dim1==dim2=={a1}")
    if a1 > a2:
        a1, a2 = a2, a1
        offset = -offset

    s1 = input_node.shape[a1]
    s2 = input_node.shape[a2]
    if not isinstance(s1, int) or not isinstance(s2, int):
        raise T2NErrorNotImplemented(
            f"diagonal on dynamic axes ({s1}, {s2}) not yet supported"
        )

    begin1 = max(0, -offset)
    begin2 = max(0, offset)
    n_diag = min(s1 - begin1, s2 - begin2)
    if n_diag <= 0:
        raise T2NErrorNotImplemented(
            f"diagonal with empty output: shapes ({s1}, {s2}), offset {offset}"
        )

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

    if begin1 != 0 or s1 != n_diag:
        inp_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "slice",
            inputs=inp_ref,
            attrs={
                "axes": [a1],
                "begin": [begin1],
                "end": [begin1 + n_diag],
            },
            output_tensor_name_suffix="_diag_slice_a1",
        )
    if begin2 != 0 or s2 != n_diag:
        inp_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "slice",
            inputs=inp_ref,
            attrs={
                "axes": [a2],
                "begin": [begin2],
                "end": [begin2 + n_diag],
            },
            output_tensor_name_suffix="_diag_slice_a2",
        )

    if not (a1 == rank - 2 and a2 == rank - 1):
        perm = [i for i in range(rank) if i not in (a1, a2)] + [a1, a2]
        inp_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "transpose",
            inputs=inp_ref,
            attrs={"axes": perm},
            output_tensor_name_suffix="_diag_perm",
        )

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_einsum",
        inputs=[inp_ref],
        ensure_tuple=False,
        force_consistent_inputs_shapes=False,
        attrs={
            "expr": _diagonal_einsum_expr(rank),
            "acc": "f32",
            "output": "",
        },
    )
    return ["tract_core"]


def _broadcast_index_to_input_shape(
    op_helper, node, input_node, index_node, dim
):
    """Broadcast a 1-D `index` to `input_node.shape` with `dim` sized `k`.

    `tract_core_scatter_elements` requires `indices` and `updates` to
    have the same shape as the iteration domain. Torch's index_* family
    ships a rank-1 `index`, so we unsqueeze it to the input rank and
    tile it across every non-`dim` axis.

    Tile repeats for dynamic non-`dim` axes are resolved at runtime via
    `resolve_attr_axis_size`; the broadcast shape returned to the
    caller mixes ints and `nnef.Identifier` for the same reason.
    """
    rank = input_node.rank
    if not isinstance(index_node.shape[0], int):
        raise T2NErrorNotImplemented(
            "index_* with dynamic-length index not supported"
        )
    k = int(index_node.shape[0])
    idx_ref = op_helper.get_or_add_tensor_variable_in_nnef(index_node)
    if rank == 1:
        return idx_ref, [k]

    repeats = []
    for axis in range(rank):
        if axis == dim:
            repeats.append(1)
        else:
            repeats.append(resolve_attr_axis_size(op_helper, input_node, axis))

    axes_to_add = [i for i in range(rank) if i != dim]
    idx_unsq = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=idx_ref,
        attrs={"axes": axes_to_add},
        output_tensor_name_suffix="_idx_unsq",
    )
    idx_bcast = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tile",
        inputs=idx_unsq,
        attrs={"repeats": repeats},
        output_tensor_name_suffix="_idx_bcast",
    )
    bcast_shape = list(input_node.shape)
    bcast_shape[dim] = k
    return idx_bcast, bcast_shape


def _emit_index_family_scatter(
    g,
    name_to_tensor,
    op_helper,
    node,
    input_node,
    dim,
    index_node,
    src_ref,
    reduction,
):
    """Common backbone for `index_fill` / `index_copy` / `index_add`.

    They differ only in `reduction` ('none' / 'add') and how `src_ref`
    is built (constant-fill vs. user-provided tensor); the indices
    broadcast and the scatter call are the same.
    """
    idx_ref, _ = _broadcast_index_to_input_shape(
        op_helper, node, input_node, index_node, dim
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_scatter_elements",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            idx_ref,
            src_ref,
        ],
        attrs={"axis": dim, "reduction": reduction},
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def index_fill(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:index_fill' to NNEF.

    `index_fill(self, dim, index, value)` writes the scalar `value` at
    every `(..., index[k], ...)` position along `dim`. Lowered to
    `tract_core_scatter_elements` with `reduction='none'` against an
    all-`value` constant of the broadcast shape.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    _check_scatter_reduction_supported(inference_target)
    input_node, dim_node, index_node, value_node = node.inputs
    dim = pick_axis(input_node, dim_node.data)
    if not isinstance(value_node, PythonConstant):
        raise T2NErrorNotImplemented("index_fill needs a constant value")

    bcast_shape = list(input_node.shape)
    if not isinstance(index_node.shape[0], int) or not all(
        isinstance(d, int) for d in bcast_shape
    ):
        raise T2NErrorNotImplemented(
            "index_fill requires static index length and input shape"
        )
    bcast_shape[dim] = int(index_node.shape[0])

    fill = torch.full(
        bcast_shape, float(value_node.data), dtype=input_node.dtype
    )
    src_const = PythonConstant(name=f"{node.outputs[0].name}_if_src", data=fill)
    src_ref = get_or_add_tensor_variable_in_nnef(g, src_const, name_to_tensor)
    return _emit_index_family_scatter(
        g,
        name_to_tensor,
        op_helper,
        node,
        input_node,
        dim,
        index_node,
        src_ref,
        reduction="none",
    )


@OP_REGISTRY.register()
def index_copy(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:index_copy' to NNEF.

    `index_copy(self, dim, index, source)` overwrites slabs along `dim`:
    `out[..., index[k], ...] = source[..., k, ...]`. Same scatter
    backbone as `index_fill`, just with the user's `source` directly.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    _check_scatter_reduction_supported(inference_target)
    input_node, dim_node, index_node, src_node = node.inputs
    dim = pick_axis(input_node, dim_node.data)
    src_ref = op_helper.get_or_add_tensor_variable_in_nnef(src_node)
    return _emit_index_family_scatter(
        g,
        name_to_tensor,
        op_helper,
        node,
        input_node,
        dim,
        index_node,
        src_ref,
        reduction="none",
    )


@OP_REGISTRY.register()
def index_add(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:index_add' to NNEF.

    `index_add(self, dim, index, source, alpha)` adds `alpha * source`
    into the input at the index slabs. We pre-multiply `source` by
    `alpha` (when not 1) and reuse the scatter backbone with
    `reduction='add'`.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    _check_scatter_reduction_supported(inference_target)
    input_node, dim_node, index_node, src_node, alpha_node = node.inputs
    dim = pick_axis(input_node, dim_node.data)
    alpha = alpha_node.data if isinstance(alpha_node, PythonConstant) else 1
    src_ref = op_helper.get_or_add_tensor_variable_in_nnef(src_node)
    if alpha != 1:
        alpha_const = PythonConstant(
            name=f"{node.outputs[0].name}_ia_alpha",
            data=torch.tensor(float(alpha), dtype=src_node.dtype),
        )
        alpha_ref = get_or_add_tensor_variable_in_nnef(
            g, alpha_const, name_to_tensor
        )
        src_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "mul",
            inputs=[src_ref, alpha_ref],
            output_tensor_name_suffix="_ia_scaled",
            force_consistent_inputs_shapes=False,
        )
    return _emit_index_family_scatter(
        g,
        name_to_tensor,
        op_helper,
        node,
        input_node,
        dim,
        index_node,
        src_ref,
        reduction="add",
    )


@OP_REGISTRY.register()
def index_put(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:index_put' (and `_`) to NNEF.

    `index_put(self, indices: Tensor?[], values, accumulate)` writes
    `values` into `self` at the positions indexed by `indices`. Only
    the `len(indices) == 1` case with a single 1-D int index along
    axis 0 is supported -- that's the `out[idx] = values` pattern that
    covers most realistic usage. Lowered to
    `tract_core_scatter_elements` with reduction `'add'` (when
    `accumulate=True`) or `'none'` (overwrite).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.torch_graph import FixedTensorList

    input_node, indices_node, values_node, accumulate_node = node.inputs
    if bool(accumulate_node.data):
        # Only the accumulate=True path needs the `reduction` NNEF attr
        # (which landed in tract 0.23.0-dev.4). Overwrite (=False) is the
        # default scatter_elements behaviour, available since at least
        # 0.21.x.
        _check_scatter_reduction_supported(inference_target)
    if not isinstance(indices_node, FixedTensorList):
        raise T2NErrorNotImplemented(
            f"index_put indices must be a FixedTensorList, got {indices_node!r}"
        )
    real_indices = [
        idx
        for idx in indices_node.data
        if not (isinstance(idx, PythonConstant) and idx.data is None)
    ]
    if len(real_indices) != 1:
        raise T2NErrorNotImplemented(
            f"index_put with {len(real_indices)} non-None indices "
            "not supported (only single-axis form covered)"
        )
    (index_node,) = real_indices
    if index_node.dtype not in (torch.int32, torch.int64):
        raise T2NErrorNotImplemented(
            f"index_put with bool / mask index not supported "
            f"(got dtype {index_node.dtype}); use scatter or "
            "masked_fill for the mask form"
        )
    accumulate = bool(accumulate_node.data)
    src_ref = op_helper.get_or_add_tensor_variable_in_nnef(values_node)
    return _emit_index_family_scatter(
        g,
        name_to_tensor,
        op_helper,
        node,
        input_node,
        dim=0,
        index_node=index_node,
        src_ref=src_ref,
        reduction="add" if accumulate else "none",
    )


@OP_REGISTRY.register()
def take(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:take' to NNEF.

    `take(self, index)` flattens `self` to 1-D and gathers along axis
    0. Lowered to a `reshape(input, shape=[-1])` followed by
    `tract_core_gather` on axis 0. The `-1` lets NNEF / tract derive
    the flat size at runtime from the input shape, so dynamic input
    axes work without any special-casing.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    input_node, index_node = node.inputs

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    flat = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=inp_ref,
        attrs={"shape": [-1]},
        output_tensor_name_suffix="_take_flat",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather",
        inputs=[
            flat,
            op_helper.get_or_add_tensor_variable_in_nnef(index_node),
        ],
        attrs={"axis": 0},
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


def _emit_bucketize(node, op_helper, *, input_node, boundaries_node, right):
    """Shared body for `bucketize` / `searchsorted`.

    For each value `v` in `input_node`, count the number of
    `boundaries` that satisfy the comparison: `<` for `right=False`
    (default; lower-bound search) or `<=` for `right=True`
    (upper-bound search). The count is the inserted-position index.

    Decomposition (1-D `boundaries` -- the typical case):

        x_exp     = unsqueeze(input, axes=[input.rank])  # (..., 1)
        cmp_mask  = (lt | le)(boundaries, x_exp)         # (..., N) bool
        cmp_int   = tract_core_cast(cmp_mask, to='i64')
        kept_dim  = sum_reduce(cmp_int, axes=[input.rank])
        output    = squeeze(kept_dim, axes=[input.rank])
    """
    if boundaries_node.rank != 1:
        raise T2NErrorNotImplemented(
            "bucketize/searchsorted with N-D boundaries (per-row) "
            "not supported; only 1-D boundaries shared across input"
        )
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor
    onode = node.outputs[0]
    base = onode.export_name
    rank = input_node.rank
    last_axis = rank  # axis we add for broadcast and reduce over

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    bnd_ref = op_helper.get_or_add_tensor_variable_in_nnef(boundaries_node)

    inp_shape = list(input_node.shape)
    n_b = boundaries_node.shape[0]
    bcast_shape = inp_shape + [int(n_b) if isinstance(n_b, int) else n_b]
    np_dtype = onode.np_dtype

    def _intermediate(name, shape):
        t = NTensor(g, name, dtype=np_dtype, shape=tuple(shape))
        name_to_tensor[name] = t
        return t

    # Unsqueeze input on the trailing axis.
    x_exp = _intermediate(f"{base}_bk_x_exp", inp_shape + [1])
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="unsqueeze",
        name=f"{x_exp.name}_op",
        inputs=(inp_ref,),
        outputs=(x_exp,),
        attribs={"axes": [last_axis]},
    )
    cmp_op = "le" if right else "lt"
    mask = _intermediate(f"{base}_bk_mask", bcast_shape)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type=cmp_op,
        name=f"{mask.name}_op",
        inputs=(bnd_ref, x_exp),
        outputs=(mask,),
        attribs={},
    )
    int_t = _intermediate(f"{base}_bk_int", bcast_shape)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="tract_core_cast",
        name=f"{int_t.name}_op",
        inputs=(mask,),
        outputs=(int_t,),
        attribs={"to": TORCH_DTYPE_TO_TRACT_STR[torch.int64]},
    )
    kd_shape = inp_shape + [1]
    kd = _intermediate(f"{base}_bk_kd", kd_shape)
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="sum_reduce",
        name=f"{kd.name}_op",
        inputs=(int_t,),
        outputs=(kd,),
        attribs={"axes": [last_axis]},
    )
    out_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        onode, prevent_variable=True
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="squeeze",
        name=f"{base}_bk_squeeze",
        inputs=(kd,),
        outputs=(out_ref,),
        attribs={"axes": [last_axis]},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def bucketize(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:bucketize' to NNEF.

    `bucketize(input, boundaries, out_int32, right)` returns the index
    in `boundaries` (1-D, sorted) for each value in `input`. Decomposed
    via broadcast-compare + sum_reduce of the comparison mask.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    input_node, boundaries_node, _out_int32_node, right_node = node.inputs
    return _emit_bucketize(
        node,
        op_helper,
        input_node=input_node,
        boundaries_node=boundaries_node,
        right=bool(right_node.data),
    )


@OP_REGISTRY.register()
def searchsorted(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:searchsorted' to NNEF.

    `searchsorted(sorted_seq, values, out_int32, right, side, sorter)`
    is `bucketize` with the args swapped (`sorted_seq` plays the role
    of `boundaries`, `values` plays the role of `input`). The `side`
    string overload supersedes `right` when present.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    sorted_seq_node = node.inputs[0]
    values_node = node.inputs[1]
    right_node = node.inputs[3] if len(node.inputs) > 3 else None
    side_node = node.inputs[4] if len(node.inputs) > 4 else None
    if (
        side_node is not None
        and not isinstance(side_node, PythonConstant)
        and side_node.data not in (None, "left", "right")
    ):
        raise T2NErrorNotImplemented(
            f"searchsorted side={side_node.data!r} not supported"
        )
    if (
        side_node is not None
        and isinstance(side_node, PythonConstant)
        and side_node.data == "right"
    ):
        right = True
    elif right_node is not None:
        right = bool(right_node.data) if right_node.data is not None else False
    else:
        right = False
    sorter_node = node.inputs[5] if len(node.inputs) > 5 else None
    if (
        sorter_node is not None
        and not isinstance(sorter_node, PythonConstant)
        and sorter_node.data is not None
    ):
        raise T2NErrorNotImplemented(
            "searchsorted with explicit `sorter` not supported"
        )
    return _emit_bucketize(
        node,
        op_helper,
        input_node=values_node,
        boundaries_node=sorted_seq_node,
        right=right,
    )
