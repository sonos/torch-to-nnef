import logging
from copy import copy

import nnef
import numpy as np
import torch

from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR, TORCH_TO_NUMPY_DTYPE
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    SimpleOpChainer,
    cast_and_add_nnef_operation,
    get_tract_dyn_axis_size_soc,
    pick_axis,
    pick_index_in_axis,
)
from torch_to_nnef.tensor import OpaqueTensorRef
from torch_to_nnef.torch_graph.ir_data import PythonConstant, TensorVariable

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


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

    has_concrete_values = True
    # we use this since by default pytorch generate max int64 value for end
    if begin_node.data is not None:
        if begin_node.data >= 0:
            begin = pick_index_in_axis(
                input_node, dim, begin_node.data, check_is_positive=False
            )
        else:
            begin = begin_node.data
            has_concrete_values = False
    else:
        has_concrete_values = False
        begin = nnef.Identifier(begin_node.export_name)

    if end_node.data is not None:
        if end_node.data >= 0:
            end = pick_index_in_axis(
                input_node, dim, end_node.data, check_is_positive=False
            )
        else:
            has_concrete_values = False
            end = end_node.data
    else:
        has_concrete_values = False
        end = nnef.Identifier(end_node.export_name)

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

    end = min(
        end,
        input_node.shape[dim],
    )
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
        decision = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            nnef_op_type="tract_core_cast",
            inputs=decision,
            attrs={"to": "bool"},
            force_full_output_tensor_name=f"{decision.name}_cast_bool",
        )
    casted_inputs = [decision]
    expected_dtype = TORCH_TO_NUMPY_DTYPE[target_torch_dtype]
    expected_dtype_tract = TORCH_DTYPE_TO_TRACT_STR[target_torch_dtype]
    for inp in inputs[1:]:
        if (
            inp.dtype != expected_dtype
            or (
                np.prod(inp.shape) == 1
                and inp.dtype not in [torch.float32, torch.int64]
            )
        ) and isinstance(op_helper.inference_target, TractNNEF):
            inp = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                nnef_op_type="tract_core_cast",
                inputs=inp,
                attrs={"to": expected_dtype_tract},
                force_full_output_tensor_name=f"{inp.name}_cast_{expected_dtype_tract}",
            )
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
    inputs = []

    for idx_node in indexes_node.data:
        i_ref = op_helper.get_or_add_tensor_variable_in_nnef(idx_node)
        casted_i_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_cast",
            inputs=[i_ref],
            attrs={"to": "TDim"},
            force_full_output_tensor_name=f"{i_ref.name}_as_tdim",
        )
        casted_unsqueezed_i_ref = (
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "unsqueeze",
                inputs=[casted_i_ref],
                attrs={"axes": [0]},
                force_full_output_tensor_name=f"{i_ref.name}_as_tdim_d1",
            )
        )
        inputs.append(casted_unsqueezed_i_ref)
    concat_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "concat",
        inputs=inputs,
        ensure_tuple=False,
        attrs={
            "axis": 0,
        },
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="indices_concat",
    )
    t_concat_ref = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=concat_ref,
        ensure_tuple=False,
        attrs={
            "axes": [1, 0],
        },
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix="indices_concat_t",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather_nd",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            t_concat_ref,
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
        shape_tensor_name = f"{input_nnef.name}_shape"
        soc = SimpleOpChainer(
            op_helper=op_helper, input_data_nodes=[input_node]
        )
        soc = soc.chain(
            "tract_core_shape_of",
            force_full_output_tensor_name=shape_tensor_name,
        )

        index_tensor_name = f"{input_nnef.name}_dim{dim}"
        if index_tensor_name not in op_helper.name_to_tensor:
            soc = soc.chain(
                "slice",
                attrs={
                    "axes": [0],
                    "begin": [dim],
                    "end": [dim + 1],
                    "stride": [1],
                },
                output_tensor_name_suffix=f"sliced{dim}",
            ).chain(
                "squeeze",
                attrs={
                    "axes": [0],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
        dim_size = nnef.Identifier(index_tensor_name)
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
        shape_tensor_name = f"{input_nnef.name}_shape"
        soc = SimpleOpChainer(
            op_helper=op_helper, input_data_nodes=[input_node]
        )
        soc = soc.chain(
            "tract_core_shape_of",
            force_full_output_tensor_name=shape_tensor_name,
        )

        index_tensor_name = f"{input_nnef.name}_dim{dim}"
        if index_tensor_name not in op_helper.name_to_tensor:
            soc = soc.chain(
                "slice",
                attrs={
                    "axes": [0],
                    "begin": [dim],
                    "end": [dim + 1],
                    "stride": [1],
                },
                output_tensor_name_suffix=f"sliced{dim}",
            ).chain(
                "squeeze",
                attrs={
                    "axes": [0],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
        dim_size = nnef.Identifier(index_tensor_name)
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


@OP_REGISTRY.register()
def scatter(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:scatter' to NNEF."""
    input_node, dim_node, indexes_node, src_node = node.inputs
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)

    # is a select with indexes
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_scatter_elements",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            op_helper.get_or_add_tensor_variable_in_nnef(
                indexes_node,
            ),
            op_helper.get_or_add_tensor_variable_in_nnef(
                src_node,
            ),
        ],
        attrs={
            "axis": dim_node.data,
        },
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def _pack_padded_sequence(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:_pack_padded_sequence' to NNEF."""
    raise T2NErrorNotImplemented(
        "support for .pack_padded_sequence not added in tract yet"
    )
    # input_node, lengths_node, batch_first_node = node.inputs[:3]
    # opacked_node, obatch_node = node.outputs
    # return ["pack_padded_sequence"]
