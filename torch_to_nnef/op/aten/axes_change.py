import typing as T

import nnef
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten.complex import (
    is_complex_dtype_and_complex_only_supported_as_lastdim,
)
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
    pick_axis,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def view(
    g,
    node,
    name_to_tensor,
    torch_graph,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:view' to NNEF."""
    (input_node, axis_node) = node.inputs
    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=inference_target.has_dynamic_axes,
    )
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        dim_data.append(2)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


@OP_REGISTRY.register()
def unflatten(
    g,
    node,
    name_to_tensor,
    torch_graph,
    op_helper,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:unflatten' to NNEF."""
    (input_node, axis_node, new_shape_chunk_node) = node.inputs
    assert isinstance(axis_node, PythonConstant), (
        "axis is supposed to be static"
    )

    rank_data = pick_axis(input_node, axis_node.data)

    partial_dim_data = get_list_of_int(
        new_shape_chunk_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=inference_target.has_dynamic_axes,
    )

    if inference_target.has_dynamic_axes:
        dim_data = []
        for dim in range(rank_data):
            # Reuse centralized dynamic axis extraction helper
            get_tract_dyn_axis_size_soc(op_helper, input_node, dim)
            dim_data.append(
                nnef.Identifier(f"{input_node.export_name}_dim{dim}")
            )
    else:
        dim_data = input_node.shape[:rank_data]
    dim_data = dim_data + partial_dim_data + input_node.shape[rank_data + 1 :]
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        dim_data.append(2)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


@OP_REGISTRY.register()
def transpose(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:transpose' to NNEF."""
    (input_node, dim0_node, dim1_node) = node.inputs
    dim0 = pick_axis(input_node, dim0_node.data)
    dim1 = pick_axis(input_node, dim1_node.data)

    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        raise T2NErrorNotImplemented(
            "complex transpose without tract complex feature flag"
        )

    new_dims_ranks = []
    for _ in range(node.outputs[0].rank):
        if _ == dim0:
            new_dims_ranks.append(dim1)
        elif _ == dim1:
            new_dims_ranks.append(dim0)
        else:
            new_dims_ranks.append(_)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": new_dims_ranks},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def permute(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:permute' to NNEF."""
    (input_node, dims_node) = node.inputs
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_axis(input_node, _) for _ in dims_node.data]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def unsqueeze(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:unsqueeze' to NNEF."""
    (input_node, axis_node) = node.inputs

    axis = pick_axis(input_node, axis_node.data)
    if axis_node.data < 0:
        axis += 1
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [axis]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def squeeze(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:squeeze' to NNEF."""
    (input_node, axis_node) = node.inputs
    dim = axis_node.data
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "squeeze",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_axis(input_node, dim)]},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def flatten(g, node, name_to_tensor, inference_target, **kwargs):
    """Translate operator: `aten::flatten` to NNEF.

    PyTorch ``flatten(start_dim, end_dim)`` flattens dims in
    ``[start_dim, end_dim]`` *inclusive*; NNEF reshape uses ``axis_count``
    (number of axes to replace), so convert as
    ``axis_count = end_dim - start_dim + 1`` after normalizing negative
    indices via :func:`pick_axis`.

    fragment reshape<?>(
        input: tensor<?>,
        shape: integer[],
        axis_start: integer = 0,
        axis_count: integer = -1
    ) -> ( output: tensor<?> );
    """
    (input_node, start_dim, end_dim) = node.inputs
    onode = node.outputs[0]
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        raise T2NErrorNotImplemented(
            "complex flatten without tract complex feature flag"
        )
    raw_start = start_dim.data if start_dim.data is not None else 0
    raw_end = end_dim.data if end_dim.data is not None else -1
    axis_start = pick_axis(input_node, raw_start)
    axis_end = pick_axis(input_node, raw_end)
    axis_count = axis_end - axis_start + 1
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={
            "dtype": onode.np_dtype,
            "shape": [-1],
            "axis_start": axis_start,
            "axis_count": axis_count,
        },
    )


@OP_REGISTRY.register()
def reshape(
    g,
    node,
    name_to_tensor,
    torch_graph,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:reshape' to NNEF."""
    (input_node, axis_node) = node.inputs

    dim_data = get_list_of_int(
        axis_node,
        torch_graph,
        name_to_tensor=name_to_tensor,
        accept_none=1,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        force_none_as_tensor_ref=True,
    )
    if is_complex_dtype_and_complex_only_supported_as_lastdim(
        input_node.dtype, inference_target
    ):
        dim_data.append(2)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": dim_data},
    )


@OP_REGISTRY.register()
def flip(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:flip' to NNEF.

    Decomposes to a chain of `tract_core_gather` calls, one per axis,
    with a constant reversed-index tensor `[N-1, ..., 0]`. Static-shape
    only: a dynamic axis size raises `T2NErrorNotImplemented` (would
    require building the index at runtime via `tract_core_range` over
    `tract_core_shape_of(input)[axis]`).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "flip requires `tract_core_gather` (TractNNEF target)"
        )
    input_node, dims_node = node.inputs
    raw_dims = list(dims_node.data) if dims_node.data is not None else []

    seen = set()
    axes = []
    for d in raw_dims:
        a = pick_axis(input_node, d)
        if a not in seen:
            seen.add(a)
            axes.append(a)

    cur_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)

    if not axes:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=cur_ref,
            attrs={"shape": list(input_node.shape)},
        )
        return []

    for axis in axes:
        if not isinstance(input_node.shape[axis], int):
            raise T2NErrorNotImplemented(
                f"flip on dynamic axis {axis} not yet supported"
            )

    for i, axis in enumerate(axes):
        n = input_node.shape[axis]
        idx_const = PythonConstant(
            name=f"{node.outputs[0].export_name}_flip_idx_{axis}",
            data=torch.arange(n - 1, -1, -1, dtype=torch.int64),
        )
        idx_ref = get_or_add_tensor_variable_in_nnef(
            g, idx_const, name_to_tensor
        )
        is_last = i == len(axes) - 1
        cur_ref = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_gather",
            inputs=[cur_ref, idx_ref],
            attrs={"axis": axis},
            force_consistent_inputs_shapes=False,
            output_tensor_name_suffix=("" if is_last else f"_flip_axis{axis}"),
        )
    return ["tract_core"]


@OP_REGISTRY.register()
def t(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:t' to NNEF.

    `Tensor.t()` is a 2D-only transpose: rank 0 / 1 inputs pass through,
    rank 2 swaps axes (0, 1). Higher ranks are a torch error and we
    don't try to be friendlier than the source.
    """
    (input_node,) = node.inputs
    rank = input_node.rank
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    if rank < 2:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=inp_ref,
            attrs={"shape": list(input_node.shape)},
        )
        return []
    if rank > 2:
        raise T2NErrorNotImplemented(f"aten::t expects rank<=2, got {rank}")
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=inp_ref,
        attrs={"axes": [1, 0]},
        pass_quantization_params=True,
    )


def _emit_static_expand(
    g,
    node,
    name_to_tensor,
    op_helper,
    input_node,
    target_shape: T.List[int],
    op_label: str,
):
    """Emit `expand`-style broadcasting to a static target shape.

    Mirrors `aten::expand`'s static path: prepend size-1 source dims as
    needed via `unsqueeze`, then `tile` per axis where the source dim
    is 1 and the target dim is larger. Source dims that are non-1 must
    already match the target dim.
    """
    src_shape = list(input_node.shape)
    if not all(isinstance(d, int) for d in src_shape):
        raise T2NErrorNotImplemented(
            f"{op_label}: dynamic source shape {src_shape} not yet supported"
        )
    rank_diff = len(target_shape) - len(src_shape)
    if rank_diff < 0:
        raise T2NErrorNotImplemented(
            f"{op_label}: target rank ({len(target_shape)}) must be "
            f">= source rank ({len(src_shape)})"
        )
    padded_src_shape = [1] * rank_diff + src_shape
    repeats = []
    for s, t in zip(padded_src_shape, target_shape, strict=True):
        if s == t or t == -1:
            repeats.append(1)
        elif s == 1:
            repeats.append(t)
        else:
            raise T2NErrorNotImplemented(
                f"{op_label}: dim {s} cannot be expanded to {t} "
                "(source dim must be 1 or equal to target)"
            )

    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    if rank_diff > 0:
        inp_ref = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "unsqueeze",
            inputs=inp_ref,
            attrs={"axes": list(range(rank_diff))},
            output_tensor_name_suffix="_expand_unsqueeze",
        )
    if all(r == 1 for r in repeats):
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=inp_ref,
            attrs={"shape": list(target_shape)},
        )
    else:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tile",
            inputs=inp_ref,
            attrs={"repeats": repeats},
        )


def _emit_shape_borrow_reshape(g, node, name_to_tensor, op_label: str):
    """Reshape `self` to match `other`'s static shape.

    Used by ``reshape_as`` / ``view_as``. The second input's shape
    must be an int tuple at parse time.
    """
    input_node, other_node = node.inputs
    target_shape = list(other_node.shape)
    if not all(isinstance(d, int) for d in target_shape):
        raise T2NErrorNotImplemented(
            f"{op_label}: dynamic shape from second input not yet "
            f"supported (got {target_shape})"
        )
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "reshape",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"shape": target_shape},
    )


@OP_REGISTRY.register()
def expand_as(g, node, name_to_tensor, op_helper, **kwargs):
    """Map PyTorch: 'aten:expand_as' to NNEF.

    `x.expand_as(y)` is `x.expand(y.size())`: broadcast (tile) along
    size-1 axes to match `y`'s static shape.
    """
    input_node, other_node = node.inputs
    target_shape = list(other_node.shape)
    if not all(isinstance(d, int) for d in target_shape):
        raise T2NErrorNotImplemented(
            f"expand_as: dynamic shape from second input not yet "
            f"supported (got {target_shape})"
        )
    _emit_static_expand(
        g,
        node,
        name_to_tensor,
        op_helper,
        input_node,
        target_shape,
        op_label="expand_as",
    )


@OP_REGISTRY.register()
def reshape_as(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:reshape_as' to NNEF.

    Equivalent to `reshape` with shape borrowed from the second input.
    """
    _emit_shape_borrow_reshape(g, node, name_to_tensor, "reshape_as")


@OP_REGISTRY.register()
def view_as(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:view_as' to NNEF.

    Equivalent to `reshape` with shape borrowed from the second input;
    NNEF reshape covers torch's view semantics for contiguous inputs.
    """
    _emit_shape_borrow_reshape(g, node, name_to_tensor, "view_as")


@OP_REGISTRY.register()
def broadcast_to(g, node, name_to_tensor, op_helper, **kwargs):
    """Map PyTorch: 'aten:broadcast_to' to NNEF.

    Equivalent to `expand` with explicit `sizes`. Static-shape only.
    """
    input_node, sizes_node = node.inputs
    sizes = list(sizes_node.data)
    if not all(isinstance(d, int) for d in sizes):
        raise T2NErrorNotImplemented(
            f"broadcast_to: dynamic sizes not yet supported (got {sizes})"
        )
    _emit_static_expand(
        g,
        node,
        name_to_tensor,
        op_helper,
        input_node,
        sizes,
        op_label="broadcast_to",
    )


def _emit_atleast_nd(g, node, name_to_tensor, n: int):
    """Map `aten::atleast_{n}d` to NNEF.

    Torch promotes 0/.../n-1 rank inputs to n-d by prepending size-1
    leading dims; rank >= n inputs pass through unchanged. The NNEF
    `unsqueeze` axes list says where the new axes go in the *output*
    shape, so we use `[0, 1, ..., missing-1]`.
    """
    (input_node,) = node.inputs
    rank = input_node.rank
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    if rank >= n:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=inp_ref,
            attrs={"shape": list(input_node.shape)},
        )
        return
    missing = n - rank
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=inp_ref,
        attrs={"axes": list(range(missing))},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def atleast_1d(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:atleast_1d' to NNEF."""
    _emit_atleast_nd(g, node, name_to_tensor, 1)


@OP_REGISTRY.register()
def atleast_2d(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:atleast_2d' to NNEF."""
    _emit_atleast_nd(g, node, name_to_tensor, 2)


@OP_REGISTRY.register()
def atleast_3d(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:atleast_3d' to NNEF.

    Note: torch's atleast_3d differs from atleast_1d/2d for rank 1
    inputs: `[N]` is reshaped to `[1, N, 1]` (not `[1, 1, N]`). Match
    that explicitly.
    """
    (input_node,) = node.inputs
    rank = input_node.rank
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    if rank >= 3:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=inp_ref,
            attrs={"shape": list(input_node.shape)},
        )
        return
    if rank == 0:
        axes = [0, 1, 2]
    elif rank == 1:
        # `[N]` -> `[1, N, 1]`: prepend axis 0, append axis 2.
        axes = [0, 2]
    else:  # rank == 2: `[H, W]` -> `[H, W, 1]`
        axes = [2]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=inp_ref,
        attrs={"axes": axes},
        pass_quantization_params=True,
    )
