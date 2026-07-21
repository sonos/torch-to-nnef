import typing as T

import nnef
import numpy as np
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    OpHelper,
    SimpleOpChainer,
    add_single_output_op,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
)
from torch_to_nnef.torch_graph import PythonConstant, TensorVariable
from torch_to_nnef.torch_graph.ir_data import Data

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register(torch_op_ids=["expand", "broadcast_to"])
def expand(node, inference_target, op_helper, **kwargs):
    """Translate operator `aten::expand` to NNEF.

    Illustration of expand:.
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
    for idx, dim in enumerate(shape_node.data):
        if isinstance(dim, PythonConstant):
            if inference_target.has_dynamic_axes:
                dim = dim.into_tensor_variable()
            else:
                dim = dim.data
        elif isinstance(dim, TensorVariable):
            if not inference_target.has_dynamic_axes:
                dim = int(dim.data)
        elif isinstance(dim, int) and inference_target.has_dynamic_axes:
            dim = PythonConstant(
                name=f"{shape_node.export_name}_{idx}", data=dim
            ).into_tensor_variable()
        shapes.append(dim)

    repeats = _expand_build_repeats(
        op_helper,
        input_node,
        shape_node,
        shapes,
        node,
        inference_target=inference_target,
    )

    nnef_input_tensor = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if input_node.rank != len(repeats) and isinstance(
        inference_target, TractNNEF
    ):
        qte_missing_dim = len(repeats) - input_node.rank
        assert qte_missing_dim > 0, qte_missing_dim

        nnef_input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=nnef_input_tensor,
            attrs={"axes": [0] * qte_missing_dim},
            output_tensor_name_suffix="unsqueeze_align",
        )

    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tile",
        inputs=nnef_input_tensor,
        attrs={"repeats": repeats},
        output_tensor_name_suffix="repeat",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=out,
        attrs={
            "shape": _fill_negone_with_dim_by_rank_order(
                op_helper, input_node, shapes, inference_target
            )
        },
    )
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
        return ["tract_core"]
    return []


def div_expand_repeat_build(
    input_shape_soc: SimpleOpChainer,
    input_dim: int,
    shape_dim: TensorVariable,
    inference_target,
):
    if not inference_target.has_dynamic_axes:
        raise T2NErrorNotImplemented()

    assert isinstance(input_dim, int)
    assert isinstance(shape_dim, TensorVariable)
    original_name = input_shape_soc.input_data_nodes[0].export_name
    original_name = original_name.replace("_shape", "")
    soc = (
        input_shape_soc.chain(
            "slice",
            attrs={
                "axes": [0],
                "begin": [input_dim],
                "end": [input_dim + 1],
                "stride": [1],
            },
            output_tensor_name_suffix=f"sliced{input_dim}",
            reuse_if_name_exists=True,
        )
        .chain(
            "squeeze",
            attrs={"axes": [0]},
            force_full_output_tensor_name=f"{original_name}_dim{input_dim}",
            reuse_if_name_exists=True,
        )
        .add_new_input_node(shape_dim, index=0)
        .chain(
            "div",
            force_consistent_inputs_shapes=False,
            force_full_output_tensor_name=f"{shape_dim.export_name}_expand_divided{input_dim}",
        )
    )
    if isinstance(inference_target, TractNNEF):
        soc = soc.chain(
            "tract_core_cast",
            attrs={
                "to": "tdim",
            },
            output_tensor_name_suffix="as_tdim",
        )
    return soc.input_data_nodes[0]


def _append_repeats_on_existing_dims(
    op_helper,
    node,
    input_node,
    shapes,
    input_shape_soc,
    inference_target,
):
    repeats = []
    for idx, (input_dim, shape_dim) in enumerate(
        zip(input_node.shape, shapes[-len(input_node.shape) :], strict=False)
    ):
        if not inference_target.has_dynamic_axes:
            assert isinstance(shape_dim, int), shape_dim
            assert isinstance(input_dim, int), input_dim
            if shape_dim == -1:
                repeats.append(1)
                continue
            if shape_dim < 0:
                raise T2NErrorNotImplemented(
                    f"expected positive but got {shape_dim}"
                )
            repeats.append(int(shape_dim / input_dim))
            continue
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(f"{inference_target}")

        if shape_dim == -1 or (
            isinstance(shape_dim, Data) and shape_dim.data == -1
        ):
            output_tensor = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "tract_core_cast",
                inputs=op_helper.get_or_add_tensor_variable_in_nnef(
                    TensorVariable(
                        name=f"{node.outputs[0].name}_{idx}_raw",
                        data=torch.tensor(1),
                        shape=[],
                        dtype=torch.int64,
                    ),
                ),
                attrs={
                    "to": "tdim",
                },
                output_tensor_name_suffix=f"{idx}_as_tdim",
            )
            repeats.append(nnef.Identifier(output_tensor.name))
        else:
            assert input_shape_soc is not None
            repeats.append(
                nnef.Identifier(
                    div_expand_repeat_build(
                        input_shape_soc.clone(),
                        idx,
                        shape_dim,
                        inference_target,
                    ).name
                )
            )
    return repeats


def _expand_build_repeats(
    op_helper, input_node, shape_node, shapes, node, inference_target
):
    input_shape_soc = None
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
        input_shape_soc = SimpleOpChainer(op_helper, [input_node]).chain(
            "tract_core_shape_of",
            force_full_output_tensor_name=f"{input_node.export_name}_shape",
            reuse_if_name_exists=True,
        )
    repeats = _append_repeats_on_existing_dims(
        op_helper,
        node,
        input_node,
        shapes,
        input_shape_soc,
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
                raise T2NErrorNotImplemented(
                    "In such case would need to apply mul chain ops "
                    "and replace base_mul with related assigned symbol"
                )
        repeats.insert(0, base_mul)
    return repeats


def _fill_negone_with_dim_by_rank_order(
    op_helper: OpHelper,
    input_node,
    shapes: T.List[int],
    inference_target: InferenceTarget,
) -> T.List[int]:
    """Cast each -1 encountered in shapes to incremental rank dim in input_node.

    This use case was encountered in pytorch .expand operator

    where for example (picked from MHA in pytorch lib):
        # given v1.shape == (10, 1, 20, 30)
        v1.expand([-1, 1, -1, -1])
        # is equivalent to
        v1.expand([10, 1, 20, 30])

    Things that also need to work:
    - torch.arange(12).expand(1, -1)
        # why NVIDIA dev do not use unsqueeze here ?
    - torch.arange(12).reshape(3, 4).expand(1, 4, -1, 4)
    will find the shape sequence in expand 'sizes' params

    We need to realise those shape at export since NNEF need concret dim value
    here no symbolics are handled

    """
    new_shapes = []
    align_index = len(shapes) - len(input_node.shape)
    for axis, s in enumerate(shapes):
        if isinstance(s, Data) and s.data == -1:
            s = s.data
        if inference_target.has_dynamic_axes:
            if isinstance(s, Data):
                if s.data is not None and s.data:
                    new_shapes.append(s.data)
                else:
                    new_shapes.append(
                        nnef.Identifier(
                            op_helper.name_to_tensor[s.export_name].name
                        )
                    )
            elif s == -1:
                if not isinstance(inference_target, TractNNEF):
                    raise T2NErrorNotImplemented(inference_target)
                new_shapes.append(
                    nnef.Identifier(
                        get_tract_dyn_axis_size_soc(
                            op_helper, input_node, axis - align_index
                        ).output_name
                    )
                )
            else:
                raise T2NErrorNotImplemented("unexpected dim value: ", s)
        elif s == -1:
            new_shapes.append(input_node.shape[axis - align_index])
        elif s > 0:
            new_shapes.append(s)
        else:
            raise T2NErrorNotImplemented("unexpected dim value: ", s)
    return new_shapes


@OP_REGISTRY.register()
def repeat(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:repeat' to NNEF."""
    (input_node, axis_node) = node.inputs
    nnef_input_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if input_node.rank != len(axis_node.data) and isinstance(
        inference_target, TractNNEF
    ):
        qte_missing_dim = len(axis_node.data) - input_node.rank
        assert qte_missing_dim > 0, qte_missing_dim

        nnef_input_tensor = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=nnef_input_tensor,
            attrs={"axes": [0] * qte_missing_dim},
            output_tensor_name_suffix="unsqueeze_align",
        )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=nnef_input_tensor,
        attrs={"repeats": axis_node.data},
    )


@OP_REGISTRY.register()
def repeat_interleave(g, node, name_to_tensor, inference_target, **kwargs):
    """This is same as np.repeat.

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
        raise T2NErrorNotImplemented(
            "case with flattening tensor not implemented"
        )
    axis = axis_node.data
    if axis < 0:
        axis += input_node.rank
    repeat_count = _static_repeat_interleave_count(n_repeats)
    if repeat_count is None:
        repeat_count = _repeat_interleave_count_from_shapes(
            input_node, node.outputs[0], axis
        )
    if repeat_count is None:
        raise T2NErrorNotImplemented(
            "case with more than 1 dim repeats not implemented: "
            f"type={type(n_repeats).__name__}, data={n_repeats.data!r}, "
            f"traced={getattr(n_repeats, '_traced_data', None)!r}, "
            f"shape={getattr(n_repeats, 'shape', None)!r}"
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
        attrs={"axes": [axis + 1]},
        output_tensor_name_suffix="unsqueeze",
    )

    # build repeats
    repeats = [1] * (input_node.rank + 1)
    repeats[axis + 1] = repeat_count

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
            raise T2NErrorNotImplemented(inference_target)
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
        _repeats[axis] = repeat_count
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
        new_shape[axis] *= repeat_count

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=out,
        attrs={"shape": new_shape},
    )
    return nnef_modules


def _static_repeat_interleave_count(n_repeats) -> T.Optional[int]:
    value = n_repeats.data
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())

    traced = getattr(n_repeats, "_traced_data", None)
    if isinstance(traced, torch.Tensor) and traced.numel() == 1:
        return int(traced.item())
    return None


def _repeat_interleave_count_from_shapes(
    input_node, output_node, axis: int
) -> T.Optional[int]:
    if input_node.shape is None or output_node.shape is None:
        return None
    input_dim = input_node.shape[axis]
    output_dim = output_node.shape[axis]
    if not isinstance(input_dim, int) or not isinstance(output_dim, int):
        return None
    if input_dim <= 0 or output_dim % input_dim != 0:
        return None
    return output_dim // input_dim


@OP_REGISTRY.register()
def tile(g, node, name_to_tensor, op_helper, **kwargs):
    """Map PyTorch: 'aten:tile' to NNEF.

    `torch.tile(x, dims)` differs from `torch.repeat` only in how rank
    mismatch between `dims` and `x.dim()` is handled:

    - ``len(dims) > x.dim()``: treat ``x`` as if it had leading size-1
      dims (same as `repeat`); we unsqueeze upstream.
    - ``len(dims) < x.dim()``: prepend 1s to ``dims`` so its length
      matches ``x.dim()`` (this is the only case `repeat` rejects).
    """
    (input_node, dims_node) = node.inputs
    raw_dims = list(dims_node.data)

    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    rank = input_node.rank
    if len(raw_dims) > rank:
        n_unsqueeze = len(raw_dims) - rank
        inp_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=inp_ref,
            attrs={"axes": list(range(n_unsqueeze))},
            output_tensor_name_suffix="_tile_unsqueeze",
        )
        aligned_dims = raw_dims
    elif len(raw_dims) < rank:
        aligned_dims = [1] * (rank - len(raw_dims)) + raw_dims
    else:
        aligned_dims = raw_dims

    # Each entry can be a literal int (the common case) or a runtime
    # `TensorVariable` (e.g. `x.tile((batch, 1))` where `batch` came
    # from `aten::size`). Runtime entries are forwarded to NNEF as an
    # `Identifier` so tract resolves them at eval time.
    repeats = []
    for d in aligned_dims:
        if isinstance(d, int):
            repeats.append(d)
        elif isinstance(d, PythonConstant) and isinstance(d.data, int):
            repeats.append(d.data)
        elif isinstance(d, TensorVariable):
            op_helper.get_or_add_tensor_variable_in_nnef(d)
            repeats.append(nnef.Identifier(d.export_name))
        else:
            raise T2NErrorNotImplemented(
                f"tile dim of unsupported type {type(d).__name__}: {d}"
            )

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tile",
        inputs=inp_ref,
        attrs={"repeats": repeats},
    )
