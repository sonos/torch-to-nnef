import typing as T

import nnef
import torch

from torch_to_nnef.dtypes import NUMPY_TO_TORCH_DTYPE
from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
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
def expand(
    g, node, name_to_tensor, nnef_spec_strict, has_dynamic_axes, **kwargs
):
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
            if nnef_spec_strict or not has_dynamic_axes:
                dim = int(dim.data)
            else:
                dim = nnef.Identifier(dim.export_name)
        shapes.append(dim)

    repeats = _expand_build_repeats(
        g, name_to_tensor, input_node, shape_node, shapes
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


def _expand_build_repeats(g, name_to_tensor, input_node, shape_node, shapes):
    repeats = []
    for input_dim, shape_dim in zip(
        input_node.shape, shapes[-len(input_node.shape) :]
    ):
        if shape_dim in [-1, input_dim]:
            repeats.append(1)
        else:
            if input_dim > 1:
                if isinstance(shape_dim, nnef.Identifier):
                    output_tensor = add_tensor_variable_node_as_nnef_tensor(
                        g,
                        TensorVariable(
                            name=f"{shape_dim}_expand_divided",
                            data=None,
                            shape=list(name_to_tensor[shape_dim].shape),
                            dtype=NUMPY_TO_TORCH_DTYPE[
                                name_to_tensor[shape_dim].dtype
                            ],
                        ),
                        name_to_tensor,
                        name_suffix="",
                        prevent_variable=True,
                    )
                    cast_and_add_nnef_operation(
                        name_to_tensor=name_to_tensor,
                        graph=g,
                        type="div",
                        inputs=(
                            name_to_tensor[shape_dim],
                            get_or_add_tensor_variable_in_nnef(
                                g,
                                TensorVariable(
                                    name=f"{shape_dim}_expand_divisor",
                                    data=torch.tensor(
                                        input_dim, dtype=torch.int32
                                    ),
                                    dtype=torch.int32,
                                    shape=[1],
                                ),
                                name_to_tensor,
                            ),
                        ),
                        outputs=tuple([output_tensor]),
                        attribs={},
                    )
                    repeats.append(nnef.Identifier(output_tensor.name))
                else:
                    repeats.append(int(shape_dim / input_dim))
            else:
                # div per 1 hence shape_dim
                repeats.append(shape_dim)

    if len(shape_node.data) - input_node.rank > 0:
        base_mul = 1
        mul_to_ids = []
        for val in shape_node.data[: -input_node.rank]:
            if isinstance(val, TensorVariable):
                mul_to_ids.append(val)
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
