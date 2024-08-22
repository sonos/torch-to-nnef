import typing as T

import nnef
import numpy as np
import torch

from torch_to_nnef.dtypes import NUMPY_TO_TORCH_DTYPE
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.primitive.base import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.torch_graph import PythonConstant, TensorVariable

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def expand(g, node, name_to_tensor, inference_target, **kwargs):
    """
    Illustration of expand:
        torch.arange(9).reshape(3, 3).expand(2, 3, 3)

        Out[4]:
        tensor([[[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]],

                [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]]])

    which can be re-expressed as:
        torch.arange(9).reshape(3, 3).repeat(2).reshape(2, 3, 3)

    this allows us to express it as a NNEF tile followed by a reshape.

    """
    (input_node, shape_node) = node.inputs

    shapes = []
    for dim in shape_node.data:
        if isinstance(dim, PythonConstant):
            dim = dim.data
        elif isinstance(dim, TensorVariable):
            if not inference_target.has_dynamic_axes:
                dim = int(dim.data)
            else:
                dim = nnef.Identifier(dim.export_name)
        shapes.append(dim)

    repeats = _expand_build_repeats(
        g,
        name_to_tensor,
        input_node,
        shape_node,
        shapes,
        node,
        inference_target=inference_target,
    )

    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"repeats": repeats},
        output_tensor_name_suffix="repeat",
    )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=out,
        attrs={
            "shape": _fill_negone_with_dim_by_rank_order(input_node, shapes)
        },
    )
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
        return ["tract_core"]
    return []


def div_expand_repeat_build(
    g,
    name_to_tensor,
    node,
    input_shape_nnef_tensor,
    idx,
    input_dim,
    shape_dim,
    inference_target,
):
    output_tensor = add_tensor_variable_node_as_nnef_tensor(
        g,
        TensorVariable(
            name=f"{shape_dim}_expand_divided",
            data=None,
            shape=list(name_to_tensor[shape_dim].shape),
            dtype=NUMPY_TO_TORCH_DTYPE[name_to_tensor[shape_dim].dtype],
        ),
        name_to_tensor,
        name_suffix="",
        prevent_variable=True,
    )
    if inference_target.has_dynamic_axes:
        # repeats on non const not working in tract<=0.21.3
        # so while correct graph notation, tract will fail
        divisor_nnef_tensor_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "slice",
            inputs=input_shape_nnef_tensor,
            attrs={
                "axes": [0],
                "begin": [idx],
                "end": [idx + 1],
                "stride": [1],
            },
            output_tensor_name_suffix=f"axis{idx}_divisor_tensor",
        )
        divisor_nnef_tensor = add_single_output_op(  # scalar
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=(divisor_nnef_tensor_tensor,),
            attrs={"axes": [0]},
            output_tensor_name_suffix=f"axis{idx}_divisor",
        )
    else:
        divisor_nnef_tensor = get_or_add_tensor_variable_in_nnef(
            g,
            TensorVariable(
                name=f"{shape_dim}_expand_divisor",
                data=torch.tensor(input_dim, dtype=torch.int32),
                dtype=torch.int32,
                shape=[1],
            ),
            name_to_tensor,
        )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="div",
        inputs=(
            name_to_tensor[shape_dim],
            divisor_nnef_tensor,
        ),
        outputs=tuple([output_tensor]),
        attribs={},
        force_consistent_inputs_shapes=False,
    )
    if isinstance(inference_target, TractNNEF):
        output_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=output_tensor,
            attrs={
                "to": "tdim",
            },
            output_tensor_name_suffix=f"casted{idx}",
        )
    return output_tensor


def _append_repeats_on_existing_dims(
    g,
    name_to_tensor,
    node,
    input_node,
    shapes,
    input_shape_nnef_tensor,
    inference_target,
):
    repeats = []
    for idx, (input_dim, shape_dim) in enumerate(
        zip(input_node.shape, shapes[-len(input_node.shape) :])
    ):
        if shape_dim in [-1, input_dim] and isinstance(
            inference_target, TractNNEF
        ):
            output_tensor = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_cast",
                inputs=get_or_add_tensor_variable_in_nnef(
                    g,
                    TensorVariable(
                        name=f"{node.outputs[0].name}_{idx}_raw",
                        data=torch.tensor(1),
                        shape=[],
                        dtype=torch.int32,
                    ),
                    name_to_tensor,
                ),
                attrs={
                    "to": "tdim",
                },
                output_tensor_name_suffix=f"{idx}_casted",
            )
            repeats.append(nnef.Identifier(output_tensor.name))
        else:
            if input_dim > 1:
                if isinstance(shape_dim, nnef.Identifier):
                    assert input_shape_nnef_tensor is not None
                    repeats.append(
                        nnef.Identifier(
                            div_expand_repeat_build(
                                g,
                                name_to_tensor,
                                node,
                                input_shape_nnef_tensor,
                                idx,
                                input_dim,
                                shape_dim,
                                inference_target,
                            ).name
                        )
                    )
                else:
                    repeats.append(int(shape_dim / input_dim))
            else:
                # div per 1 hence shape_dim
                repeats.append(shape_dim)
    return repeats


def _expand_build_repeats(
    g, name_to_tensor, input_node, shape_node, shapes, node, inference_target
):
    input_shape_nnef_tensor = None
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
        input_shape_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_shape_of",
            inputs=get_or_add_tensor_variable_in_nnef(
                g, input_node, name_to_tensor
            ),
            output_tensor_name_suffix="shape_of_input",
        )
    repeats = _append_repeats_on_existing_dims(
        g,
        name_to_tensor,
        node,
        input_node,
        shapes,
        input_shape_nnef_tensor,
        inference_target,
    )

    if len(shape_node.data) - input_node.rank > 0:
        base_mul = 1
        mul_to_ids = []
        vals = shape_node.data
        if input_node.rank != 0:
            vals = vals[: -input_node.rank]
        for val in vals:
            if isinstance(val, TensorVariable):
                mul_to_ids.append(val)
            elif isinstance(val, PythonConstant):
                base_mul *= val.data
            else:
                base_mul *= val

        if mul_to_ids:
            if base_mul == 1 and len(mul_to_ids) == 1:
                base_mul = nnef.Identifier(mul_to_ids[0].export_name)
            else:
                raise TorchToNNEFNotImplementedError(
                    "In such case would need to apply mul chain ops "
                    "and replace base_mul with related assigned symbol"
                )
        repeats.insert(0, base_mul)
    return repeats


def _fill_negone_with_dim_by_rank_order(
    input_node, shapes: T.List[int]
) -> T.List[int]:
    """Cast each -1 encountered in shapes to incremental rank dim in input_node

    This use case was encountered in pytorch .expand operator

    where by example (picked from MHA in pytorch lib):
        # given v1.shape == (10, 1, 20, 30)
        v1.expand([-1, 1, -1, -1])
        # is equivalent to
        v1.expand([10, 1, 20, 30])

    We need to realise those shape at export since NNEF need concret dim value here
    no symbolics are handled

    """
    new_shapes = []
    for rank_id, s in enumerate(shapes):
        if s == -1:
            new_shapes.append(input_node.shape[rank_id])
        elif isinstance(s, nnef.Identifier) or s > 0:
            new_shapes.append(s)
        else:
            raise TorchToNNEFNotImplementedError("unexpected dim value: ", s)
    return new_shapes


@OP_REGISTRY.register()
def repeat(g, node, name_to_tensor, **kwargs):
    (input_node, axis_node) = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"repeats": axis_node.data},
    )


@OP_REGISTRY.register()
def repeat_interleave(g, node, name_to_tensor, inference_target, **kwargs):
    """this is same as np.repeat

    Equivalent with repeat:
        te = y
        new = te.unsqueeze(dim+1)
        new_repeats = [1] * (len(te.shape) + 1)
        new_repeats[ dim + 1 ] = n_repeat
        shapes = list(te.shape)
        shapes[dim] *= n_repeat
        new.repeat(new_repeats).reshape(shapes)

    """
    (input_node, n_repeats, axis_node, *_) = node.inputs
    if not isinstance(axis_node.data, int):
        raise NotImplementedError("case with flattening tensor not implemented")
    if not isinstance(n_repeats.data, int):
        raise NotImplementedError(
            "case with more than 1 dim repeats not implemented"
        )

    # unsqueeze
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [axis_node.data + 1]},
        output_tensor_name_suffix="unsqueeze",
    )

    # build repeats
    repeats = [1] * (input_node.rank + 1)
    repeats[axis_node.data + 1] = n_repeats.data

    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=out,
        attrs={"repeats": repeats},
        output_tensor_name_suffix="tile",
    )
    # need to compute shape live. if dynamix axes exists
    nnef_modules = []
    if inference_target.has_dynamic_axes:
        if not isinstance(inference_target, TractNNEF):
            raise TorchToNNEFNotImplementedError(inference_target)
        input_shape_nnef_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_shape_of",
            inputs=get_or_add_tensor_variable_in_nnef(
                g, input_node, name_to_tensor
            ),
            output_tensor_name_suffix="shape_of_input",
        )

        nnef_modules.append("tract_core")

        _repeats = [1] * (input_node.rank)
        _repeats[axis_node.data] = n_repeats.data
        _repeats = torch.from_numpy(np.array(_repeats))
        new_shape_out = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "mul",
            inputs=[
                input_shape_nnef_tensor,
                get_or_add_tensor_variable_in_nnef(
                    g,
                    TensorVariable(
                        name=f"repeat_of_{input_node.export_name}",
                        data=_repeats,
                        shape=list(_repeats.shape),
                        dtype=_repeats.dtype,
                    ),
                    name_to_tensor,
                ),
            ],
            output_tensor_name_suffix="new_shape",
            force_consistent_inputs_shapes=False,
        )
        new_shape_tdim_casted = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_cast",
            inputs=[new_shape_out],
            output_tensor_name_suffix="new_shape_as_tdim",
            force_consistent_inputs_shapes=False,
            attrs={"to": "tdim"},
        )
        new_shape = nnef.Identifier(new_shape_tdim_casted.name)
    else:
        new_shape = list(input_node.shape)
        new_shape[axis_node.data] *= n_repeats.data

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=out,
        attrs={"shape": new_shape},
    )
    return nnef_modules
