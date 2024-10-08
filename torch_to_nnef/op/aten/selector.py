import logging

import nnef
import numpy as np

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    get_tract_dyn_axis_size_soc,
    pick_axis,
    pick_index_in_axis,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

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
            raise TorchToNNEFNotImplementedError(inference_target)
        # Case with TractNNEF.version < 0.21.7 are handled upper
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "dyn_slice",
            inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
            attrs={
                "axis": pick_axis(input_node, dim),
                "begin": begin,
                "end": end,
                "stride": stride_node.data,
            },
            pass_quantization_params=True,
        )
        return ["dyn_slice"]

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
    """Old version of slice for tract version prior to 0.21.7"""
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


@OP_REGISTRY.register()
def where(node, op_helper, **kwargs):
    (condition_node, true_value_node, false_value_node) = node.inputs

    inputs = op_helper.data_nodes_to_nnef_tensors(
        [condition_node, true_value_node, false_value_node]
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type="select",
        inputs=inputs,
    )


@OP_REGISTRY.register()
def narrow(node, op_helper, **kwargs):
    """Fancy slice made in PyTorch

    torch.narrow(input, dim, start, length)

    example:

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> torch.narrow(x, 0, 0, 2)
        tensor([[ 1,  2,  3],
                [ 4,  5,  6]])

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


@OP_REGISTRY.register(torch_op_ids=["index"])
def index_(node, op_helper, inference_target, **kwargs):
    """
    fragment gather<?>(
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
        if not all(
            (isinstance(idx, PythonConstant) and idx.data is None)
            for idx in indexes_node.data[:-1]
        ):
            raise TorchToNNEFNotImplementedError(
                "index dim>1 implemented only with all prior dim slice being [:]"
            )

    custom_fragments = []
    if isinstance(inference_target, TractNNEF):
        op_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
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
        attrs={
            "axis": len(indexes_node.data) - 1,
        },
        force_consistent_inputs_shapes=False,
    )
    return custom_fragments


@OP_REGISTRY.register()
def embedding(node, op_helper, inference_target, **kwargs):
    (
        weight_node,
        indices_node,
        _,  # padding_idx_node
        _,  # scale_grad_by_freq_node
        _,  # sparse_node
    ) = node.inputs

    custom_fragments = []
    if isinstance(inference_target, TractNNEF):
        op_name = "tract_core_gather"
        custom_fragments += ["tract_core"]
    else:
        op_name = "gather"
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        op_name,
        inputs=op_helper.data_nodes_to_nnef_tensors(
            [weight_node, indices_node]
        ),
        attrs={"axis": 0},
    )
    return custom_fragments


@OP_REGISTRY.register()
def masked_fill(node, op_helper, inference_target, **kwargs):
    input_node, mask_node, value_node = node.inputs

    false_value_node = input_node
    false_nnef_tensor = op_helper.get_or_add_tensor_variable_in_nnef(
        false_value_node
    )
    # value is always a float according to torch spec
    true_value_node = value_node.into_tensor_variable()
    if true_value_node.data is not None:
        true_value_node.data = true_value_node.data.to(false_value_node.dtype)
    if inference_target.has_dynamic_axes:
        if not isinstance(inference_target, TractNNEF):
            raise TorchToNNEFNotImplementedError(inference_target)
        # repeats on non const not working in tract<=0.21.3
        # so while correct graph notation, tract will fail
        out = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_shape_of",
            inputs=false_nnef_tensor,
            output_tensor_name_suffix="shape_of_false",
        )

        # force rank to be the same
        true_value_node.data = true_value_node.data.repeat(
            *([1] * false_value_node.rank)
        )

        true_nnef_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tile",
            inputs=op_helper.get_or_add_tensor_variable_in_nnef(
                true_value_node, name_suffix="true_scalar"
            ),
            attrs={"repeats": nnef.Identifier(str(out.name))},
            output_tensor_name_suffix="true_expanded",
        )
    else:
        # Static expansion
        true_value_node.data = true_value_node.data.repeat(
            false_value_node.shape
        )
        true_value_node.dtype = false_value_node.dtype
        true_nnef_tensor = op_helper.get_or_add_tensor_variable_in_nnef(
            true_value_node
        )

    # tract need float where ?
    # mask_node.data = mask_node.data.float()
    # mask_node.dtype = mask_node.data.dtype
    condition_node = mask_node

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_type="select",
        inputs=[
            op_helper.get_or_add_tensor_variable_in_nnef(condition_node),
            true_nnef_tensor,
            false_nnef_tensor,
        ],
    )
