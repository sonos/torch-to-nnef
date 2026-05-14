import typing as T

import nnef
import torch
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten.complex import (
    is_complex_dtype_and_complex_only_supported_as_lastdim,
)
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    cast_and_add_nnef_operation,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
    pick_axis,
    resolve_attr_axis_size,
)
from torch_to_nnef.torch_graph import FixedTensorList
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


@OP_REGISTRY.register(torch_op_ids=["numpy_T"])
def numpy_t(g, node, name_to_tensor, **kwargs):
    """Map PyTorch `aten::numpy_T` (`Tensor.T`) to NNEF.

    `Tensor.T` reverses every axis -- it is the rank-N generalisation of
    matrix transpose. Equivalent to `permute([N-1, N-2, ..., 0])`.
    """
    (input_node,) = node.inputs
    perm = list(range(input_node.rank - 1, -1, -1))
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": perm},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register(torch_op_ids=["mT", "mH", "matrix_H"])
def matrix_transpose(g, node, name_to_tensor, **kwargs):
    """Map `aten::mT` / `aten::mH` / `aten::matrix_H` to NNEF.

    `matrix_H` is the native-functions schema name for the Hermitian
    transpose property (`Tensor.H`); aliased here since for real
    dtypes it has the same semantics as `mT` / `mH` (axis swap on the
    last two dims).

    Both ops swap the last two axes of a rank-`>=` 2 tensor. `mH` is the
    conjugate-transpose; for real-valued tensors (the only ones NNEF /
    tract carry without the complex feature flag) it is identical to
    `mT`, so a single emitter handles both.
    """
    (input_node,) = node.inputs
    if input_node.rank < 2:
        raise T2NErrorNotImplemented(
            f"mT / mH require rank >= 2; got {input_node.rank}"
        )
    perm = list(range(input_node.rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": perm},
        pass_quantization_params=True,
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


def _emit_flip_chain(
    g,
    node,
    name_to_tensor,
    inference_target,
    input_node,
    raw_dims,
    op_label: str,
):
    """Emit a per-axis `tract_core_gather` chain.

    Shared by `flip` / `fliplr` / `flipud` / `rot90`. Each axis becomes
    one `tract_core_gather` with a constant reversed-index tensor
    `[N-1, ..., 0]`. Static-shape only: a dynamic axis size raises
    `T2NErrorNotImplemented` (would require building the index at
    runtime via `tract_core_range` over
    `tract_core_shape_of(input)[axis]`). The last gather lands in
    `node.outputs[0]` so the caller doesn't need to thread an output
    name through.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            f"{op_label} requires `tract_core_gather` (TractNNEF target)"
        )
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
        return ["tract_core"]

    for axis in axes:
        if not isinstance(input_node.shape[axis], int):
            raise T2NErrorNotImplemented(
                f"{op_label} on dynamic axis {axis} not yet supported"
            )

    for i, axis in enumerate(axes):
        n = input_node.shape[axis]
        idx_const = PythonConstant(
            name=f"{node.outputs[0].export_name}_{op_label}_idx_{axis}",
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
            output_tensor_name_suffix=(
                "" if is_last else f"_{op_label}_axis{axis}"
            ),
        )
    return ["tract_core"]


@OP_REGISTRY.register()
def flip(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch `aten::flip(input, dims)` to NNEF."""
    input_node, dims_node = node.inputs
    raw_dims = list(dims_node.data) if dims_node.data is not None else []
    return _emit_flip_chain(
        g,
        node,
        name_to_tensor,
        inference_target,
        input_node,
        raw_dims,
        op_label="flip",
    )


@OP_REGISTRY.register()
def fliplr(g, node, name_to_tensor, inference_target, **kwargs):
    """Map `aten::fliplr` (`torch.fliplr`) to NNEF.

    Reverses elements along axis 1 (per torch's rank>=2 convention).
    """
    (input_node,) = node.inputs
    if input_node.rank < 2:
        raise T2NErrorNotImplemented(
            f"fliplr requires rank >= 2; got {input_node.rank}"
        )
    return _emit_flip_chain(
        g,
        node,
        name_to_tensor,
        inference_target,
        input_node,
        raw_dims=[1],
        op_label="fliplr",
    )


@OP_REGISTRY.register()
def flipud(g, node, name_to_tensor, inference_target, **kwargs):
    """Map `aten::flipud` (`torch.flipud`) to NNEF.

    Reverses elements along axis 0 (per torch's rank>=1 convention).
    """
    (input_node,) = node.inputs
    if input_node.rank < 1:
        raise T2NErrorNotImplemented(
            f"flipud requires rank >= 1; got {input_node.rank}"
        )
    return _emit_flip_chain(
        g,
        node,
        name_to_tensor,
        inference_target,
        input_node,
        raw_dims=[0],
        op_label="flipud",
    )


@OP_REGISTRY.register()
def rot90(g, node, name_to_tensor, inference_target, op_helper, **kwargs):
    """Map `aten::rot90(input, k, dims)` to NNEF.

    Rotates by `90 * k` degrees in the plane `(dims[0], dims[1])`.
    The rotation direction is from `dims[0]` toward `dims[1]`, matching
    torch's convention. Decomposed per the standard `flip + transpose`
    identity:

    * `k % 4 == 0`: identity (single `reshape` with the same shape so
      the named output tensor is materialised).
    * `k % 4 == 1`: `flip(dims[1]) -> transpose(dims[0], dims[1])`.
    * `k % 4 == 2`: `flip([dims[0], dims[1]])`.
    * `k % 4 == 3`: `transpose(dims[0], dims[1]) -> flip(dims[1])`.
    """
    input_node, k_node, dims_node = node.inputs
    if not isinstance(k_node, PythonConstant):
        raise T2NErrorNotImplemented("rot90: k must be statically known")
    raw_dims = list(dims_node.data) if dims_node.data is not None else [0, 1]
    if len(raw_dims) != 2:
        raise T2NErrorNotImplemented(
            f"rot90: dims must be 2-element list; got {raw_dims!r}"
        )
    d0 = pick_axis(input_node, raw_dims[0])
    d1 = pick_axis(input_node, raw_dims[1])
    if d0 == d1:
        raise T2NErrorNotImplemented(
            f"rot90: dims must be distinct; got [{d0}, {d1}]"
        )
    k = int(k_node.data) % 4
    if k == 0:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "reshape",
            inputs=get_or_add_tensor_variable_in_nnef(
                g, input_node, name_to_tensor
            ),
            attrs={"shape": list(input_node.shape)},
        )
        return []
    if k == 2:
        return _emit_flip_chain(
            g,
            node,
            name_to_tensor,
            inference_target,
            input_node,
            raw_dims=[d0, d1],
            op_label="rot90",
        )

    # k in {1, 3}: one flip + one transpose, ordering swaps with k.
    if not isinstance(input_node.shape[d1], int):
        raise T2NErrorNotImplemented(
            f"rot90 on dynamic axis {d1} not yet supported"
        )
    perm = list(range(input_node.rank))
    perm[d0], perm[d1] = perm[d1], perm[d0]

    if k == 1:
        # flip(d1) first, then transpose.
        flip_axis_n = input_node.shape[d1]
        idx_const = PythonConstant(
            name=f"{node.outputs[0].export_name}_rot90_idx_{d1}",
            data=torch.arange(flip_axis_n - 1, -1, -1, dtype=torch.int64),
        )
        idx_ref = get_or_add_tensor_variable_in_nnef(
            g, idx_const, name_to_tensor
        )
        flipped = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_gather",
            inputs=[
                get_or_add_tensor_variable_in_nnef(
                    g, input_node, name_to_tensor
                ),
                idx_ref,
            ],
            attrs={"axis": d1},
            force_consistent_inputs_shapes=False,
            output_tensor_name_suffix="_rot90_flip",
        )
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "transpose",
            inputs=flipped,
            attrs={"axes": perm},
        )
        return ["tract_core"]

    # k == 3: transpose first, then flip.
    transposed = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": perm},
        output_tensor_name_suffix="_rot90_t",
    )
    # After transpose, dim d1 still holds the previous d0's size.
    flip_axis_n = input_node.shape[d0]
    idx_const = PythonConstant(
        name=f"{node.outputs[0].export_name}_rot90_idx_{d1}",
        data=torch.arange(flip_axis_n - 1, -1, -1, dtype=torch.int64),
    )
    idx_ref = get_or_add_tensor_variable_in_nnef(g, idx_const, name_to_tensor)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather",
        inputs=[transposed, idx_ref],
        attrs={"axis": d1},
        force_consistent_inputs_shapes=False,
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def t(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:t' to NNEF.

    `Tensor.t()` is a 2D-only transpose: rank 0 / 1 inputs pass through,
    rank 2 swaps axes (0, 1). Higher ranks are a torch error and we
    don't try to be friendlier than the source.

    The rank<2 passthrough uses `remap_node` instead of emitting a
    no-op `reshape` so the input flows through untouched -- correct
    for dynamic-axes graphs (a literal `shape=` attr would lose
    symbolic dims).
    """
    (input_node,) = node.inputs
    rank = input_node.rank
    if rank < 2:
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return []
    if rank > 2:
        raise T2NErrorNotImplemented(f"aten::t expects rank<=2, got {rank}")
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=inp_ref,
        attrs={"axes": [1, 0]},
        pass_quantization_params=True,
    )
    return []


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
    for src, tgt in zip(padded_src_shape, target_shape, strict=True):
        if tgt in (src, -1):
            repeats.append(1)
        elif src == 1:
            repeats.append(tgt)
        else:
            raise T2NErrorNotImplemented(
                f"{op_label}: dim {src} cannot be expanded to {tgt} "
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


def _emit_shape_borrow_reshape(
    g, node, name_to_tensor, op_helper, inference_target, op_label: str
):
    """Reshape `self` to match `other`'s shape.

    Used by ``reshape_as`` / ``view_as``. Handles both static and
    dynamic-axes targets:

    - Static-axes: pass `other`'s shape as a literal int list.
    - Dynamic-axes: emit `tract_core_shape_of(other)` (cached) and
      pull each axis size out via slice + squeeze + cast-to-tdim,
      forwarding it to `reshape`'s `shape` attr as an `Identifier`
      so tract resolves the per-axis size at runtime.
    """
    input_node, other_node = node.inputs
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
        target_shape = []
        for i in range(other_node.rank):
            # Side effect: emits the chain that produces the
            # `<other>_dim{i}` runtime tensor (reused across axes).
            get_tract_dyn_axis_size_soc(op_helper, other_node, i)
            target_shape.append(
                nnef.Identifier(f"{other_node.export_name}_dim{i}")
            )
        custom_fragments = ["tract_core"]
    else:
        target_shape = list(other_node.shape)
        if not all(isinstance(d, int) for d in target_shape):
            raise T2NErrorNotImplemented(
                f"{op_label}: target shape has non-int entries "
                f"({target_shape}); enable dynamic_axes if `other` is "
                "produced by a runtime-shape op."
            )
        custom_fragments = []
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
    return custom_fragments


@OP_REGISTRY.register()
def expand_as(g, node, name_to_tensor, op_helper, **kwargs):
    """Map PyTorch: 'aten:expand_as' to NNEF.

    `x.expand_as(y)` is `x.expand(y.size())`: broadcast (tile) along
    size-1 axes to match `y`'s shape.

    Static-shape only for now: when `y` carries non-int (TDim) shape
    entries, raises with a hint to use `aten::expand` directly. The
    dynamic path would need the runtime per-axis repeat machinery
    that already lives in `aten::expand` (see
    `op/aten/expand.py::_append_repeats_on_existing_dims`); a
    follow-up should refactor that helper so this op can share it.
    """
    input_node, other_node = node.inputs
    target_shape = list(other_node.shape)
    if not all(isinstance(d, int) for d in target_shape):
        raise T2NErrorNotImplemented(
            f"expand_as: target shape has non-int entries "
            f"({target_shape}); for dynamic-axes models call "
            "`aten::expand(x, runtime_sizes)` directly until the "
            "runtime path lands in this handler."
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
def reshape_as(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:reshape_as' to NNEF.

    Equivalent to `reshape` with shape borrowed from the second input.
    Supports dynamic-axes via runtime `tract_core_shape_of(other)`.
    """
    return _emit_shape_borrow_reshape(
        g, node, name_to_tensor, op_helper, inference_target, "reshape_as"
    )


@OP_REGISTRY.register()
def view_as(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:view_as' to NNEF.

    Equivalent to `reshape` with shape borrowed from the second input;
    NNEF reshape covers torch's view semantics for contiguous inputs.
    Supports dynamic-axes via runtime `tract_core_shape_of(other)`.
    """
    return _emit_shape_borrow_reshape(
        g, node, name_to_tensor, op_helper, inference_target, "view_as"
    )


# `aten::broadcast_to(x, sizes)` is the same operation as
# `aten::expand(x, sizes)` for inference purposes -- both do
# broadcasting from size-1 source dims to the target sizes. It is
# registered alongside `aten::expand` in `op/aten/expand.py` so the
# full dynamic-axes machinery (runtime shape extraction, per-axis
# repeats) is reused without duplication.


def _emit_atleast_nd(g, node, name_to_tensor, torch_graph, n: int):
    """Map `aten::atleast_{n}d` to NNEF.

    Torch promotes 0/.../n-1 rank inputs to n-d by prepending size-1
    leading dims; rank >= n inputs pass through unchanged. The NNEF
    `unsqueeze` axes list says where the new axes go in the *output*
    shape, so we use `[0, 1, ..., missing-1]`. The rank-already-met
    branch aliases input -> output via ``remap_node`` rather than
    emitting a no-op reshape, so dynamic-axes graphs preserve their
    symbolic dims.
    """
    (input_node,) = node.inputs
    rank = input_node.rank
    if rank >= n:
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return
    missing = n - rank
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
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
def atleast_1d(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:atleast_1d' to NNEF."""
    _emit_atleast_nd(g, node, name_to_tensor, torch_graph, 1)


@OP_REGISTRY.register()
def atleast_2d(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:atleast_2d' to NNEF."""
    _emit_atleast_nd(g, node, name_to_tensor, torch_graph, 2)


@OP_REGISTRY.register()
def atleast_3d(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: 'aten:atleast_3d' to NNEF.

    Note: torch's atleast_3d differs from atleast_1d/2d for rank 1
    inputs: `[N]` is reshaped to `[1, N, 1]` (not `[1, 1, N]`). Match
    that explicitly. The rank>=3 passthrough aliases via
    ``remap_node`` so symbolic dims under dynamic-axes survive.
    """
    (input_node,) = node.inputs
    rank = input_node.rank
    if rank >= 3:
        torch_graph.remap_node(from_node=node.outputs[0], to_node=input_node)
        return
    if rank == 0:
        axes = [0, 1, 2]
    elif rank == 1:
        # `[N]` -> `[1, N, 1]`: prepend axis 0, append axis 2.
        axes = [0, 2]
    else:  # rank == 2: `[H, W]` -> `[H, W, 1]`
        axes = [2]
    inp_ref = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "unsqueeze",
        inputs=inp_ref,
        attrs={"axes": axes},
        pass_quantization_params=True,
    )


@OP_REGISTRY.register()
def movedim(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:movedim' to NNEF as `transpose`.

    `movedim(x, src, dst)` repositions the source axis so it ends up
    at `dst`, sliding the others left/right; the result is a permutation
    of the input's axes. Builds the explicit `[axes]` list and emits
    a single transpose.
    """
    input_node, src_node, dst_node = node.inputs
    rank = input_node.rank

    # Normalize possibly-list inputs (movedim accepts int or sequences).
    raw_src = src_node.data
    raw_dst = dst_node.data
    if isinstance(raw_src, int):
        raw_src = [raw_src]
    if isinstance(raw_dst, int):
        raw_dst = [raw_dst]
    src_axes = [pick_axis(input_node, s) for s in raw_src]
    dst_axes = [pick_axis(input_node, d) for d in raw_dst]
    if len(src_axes) != len(dst_axes):
        raise T2NErrorNotImplemented(
            f"movedim: src ({src_axes}) and dst ({dst_axes}) lengths differ"
        )

    # Build the permutation: keep all axes not in `src` in original order,
    # then insert each src_axes[i] at dst_axes[i].
    remaining = [a for a in range(rank) if a not in src_axes]
    perm = list(remaining)
    for s, d in sorted(
        zip(src_axes, dst_axes, strict=True), key=lambda p: p[1]
    ):
        perm.insert(d, s)

    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "transpose",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": perm},
        pass_quantization_params=True,
    )


def _pixel_reshuffle(node, op_helper, *, downscale: bool):
    """Shared backbone for `pixel_shuffle` / `pixel_unshuffle`.

    The two ops are inverses of each other:

    * `pixel_shuffle(x, r)` (`downscale=False`):
        (..., C*r^2, H, W) → reshape to (..., C, r, r, H, W)
        → transpose so the spatial-multiplier axes come right after
        their host axes → (..., C, H, r, W, r) → reshape to
        (..., C, H*r, W*r).

    * `pixel_unshuffle(x, r)` (`downscale=True`): the reverse fold.

    Both require the relevant spatial axes (last 3 for shuffle, last 2
    for unshuffle) to be statically known so we can compute the
    intermediate reshape shapes; otherwise we'd need a `tract_core_shape_of`
    chain to express them, which isn't worth the complexity for these
    fixed-rank ops.
    """
    input_node, factor_node = node.inputs
    r = int(factor_node.data)
    shape = list(input_node.shape)
    rank = len(shape)
    if rank < 3:
        raise T2NErrorNotImplemented(
            f"pixel_(un)shuffle expects rank>=3, got {rank}"
        )

    if downscale:
        h_dim, w_dim = shape[-2], shape[-1]
        if not all(isinstance(d, int) for d in (shape[-3], h_dim, w_dim)):
            raise T2NErrorNotImplemented(
                "pixel_unshuffle requires static C/H/W"
            )
        if h_dim % r != 0 or w_dim % r != 0:
            raise T2NErrorNotImplemented(
                f"pixel_unshuffle: H/W ({h_dim}, {w_dim}) not divisible by {r}"
            )
        c, h, w = shape[-3], h_dim // r, w_dim // r
        leading = shape[:-3]
        # (..., C, H, r, W, r)
        split_shape = list(leading) + [c, h, r, w, r]
        # → (..., C, r, r, H, W)
        perm = list(range(len(leading))) + [
            len(leading),
            len(leading) + 2,
            len(leading) + 4,
            len(leading) + 1,
            len(leading) + 3,
        ]
        # → (..., C*r^2, H, W)
        final_shape = list(leading) + [c * r * r, h, w]
    else:
        if not all(isinstance(d, int) for d in shape[-3:]):
            raise T2NErrorNotImplemented("pixel_shuffle requires static C/H/W")
        c_in, h, w = shape[-3], shape[-2], shape[-1]
        if c_in % (r * r) != 0:
            raise T2NErrorNotImplemented(
                f"pixel_shuffle: C ({c_in}) not divisible by r*r ({r * r})"
            )
        c = c_in // (r * r)
        leading = shape[:-3]
        # (..., C, r, r, H, W)
        split_shape = list(leading) + [c, r, r, h, w]
        # → (..., C, H, r, W, r)
        perm = list(range(len(leading))) + [
            len(leading),
            len(leading) + 3,
            len(leading) + 1,
            len(leading) + 4,
            len(leading) + 2,
        ]
        # → (..., C, H*r, W*r)
        final_shape = list(leading) + [c, h * r, w * r]

    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    split = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=inp_ref,
        attrs={"shape": split_shape},
        output_tensor_name_suffix="_ps_split",
    )
    permuted = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=split,
        attrs={"axes": perm},
        output_tensor_name_suffix="_ps_perm",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=permuted,
        attrs={"shape": final_shape},
    )


@OP_REGISTRY.register()
def pixel_shuffle(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:pixel_shuffle' to NNEF.

    Standard sub-pixel rearrangement: pulls every `r*r` channel block
    out as an `r*r` spatial tile, multiplying H/W by `r` and dividing C
    by `r*r`. Lowered to reshape + transpose + reshape (no fragment
    needed: stdlib only).
    """
    _pixel_reshuffle(node, op_helper, downscale=False)


@OP_REGISTRY.register()
def pixel_unshuffle(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:pixel_unshuffle' (inverse of `pixel_shuffle`)."""
    _pixel_reshuffle(node, op_helper, downscale=True)


@OP_REGISTRY.register(["channel_shuffle", "native_channel_shuffle"])
def channel_shuffle(node, op_helper, **kwargs):
    """Map PyTorch: `aten::channel_shuffle(self, groups)`.

    Reshape `(N, C, *spatial)` -> `(N, g, C/g, *spatial)`, transpose
    axes 1 and 2, then reshape back to `(N, C, *spatial)`. Used by
    ShuffleNet-family architectures.
    """
    input_node, groups_node = node.inputs
    groups = int(groups_node.data)
    shape = list(input_node.shape)
    if len(shape) < 2:
        raise T2NErrorNotImplemented(
            f"channel_shuffle expects rank>=2, got {len(shape)}"
        )
    c = shape[1]
    if not isinstance(c, int):
        raise T2NErrorNotImplemented(
            "channel_shuffle on dynamic channel axis not yet supported"
        )
    if c % groups != 0:
        raise T2NErrorNotImplemented(
            f"channel_shuffle: C ({c}) not divisible by groups ({groups})"
        )
    n = shape[0]
    spatial = shape[2:]
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    # (N, C, *spatial) -> (N, g, C/g, *spatial)
    reshaped = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=[inp],
        attrs={"shape": [n, groups, c // groups, *spatial]},
        output_tensor_name_suffix="_chs_split",
    )
    # Swap axes 1 and 2.
    perm = list(range(2 + len(spatial) + 1))
    perm[1], perm[2] = 2, 1
    transposed = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=[reshaped],
        attrs={"axes": perm},
        output_tensor_name_suffix="_chs_perm",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=[transposed],
        attrs={"shape": [n, c, *spatial]},
    )
    return []


def _emit_broadcast_to(op_helper, node, src_ref, target_shape, output_idx):
    """Emit `tract_core_broadcast` to `node.outputs[output_idx]`.

    `tract_core_broadcast` maps to tract's `MultiBroadcastTo`, which is
    exactly torch's `broadcast_tensors` semantics: take an input that's
    already broadcast-compatible with `target_shape` and replicate the
    size-1 axes up to the target size.

    `target_shape` entries may be plain `int` (static) or
    `nnef.Identifier` (resolved at runtime from a `tract_core_shape_of`
    chain emitted upstream).
    """
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor
    onode = node.outputs[output_idx]
    out_ref = op_helper.get_or_add_tensor_variable_in_nnef(
        onode, prevent_variable=True
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="tract_core_broadcast",
        name=f"{onode.export_name}_op",
        inputs=(src_ref,),
        outputs=out_ref,
        attribs={"shape": list(target_shape)},
    )


def _make_ntensor_with_shape(g, name_to_tensor, name, shape, np_dtype):
    """Register an `NTensor` with an explicit shape.

    Rank-changing intermediates can't go through
    `add_single_output_op_from_nnef_tensors`, which inherits
    `node.outputs[0].shape` and asserts single-output.
    """
    tensor = NTensor(g, name, dtype=np_dtype, shape=tuple(shape))
    name_to_tensor[name] = tensor
    return tensor


@OP_REGISTRY.register()
def broadcast_tensors(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:broadcast_tensors' to NNEF.

    `broadcast_tensors([t0, t1, ...])` returns each input expanded to
    the common broadcast shape. Each output is a separate
    `tract_core_broadcast(t_i, shape=common)` call -- the common
    shape is whatever torch traced into `node.outputs[i].shape` (all
    outputs share it).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    (input_list_node,) = node.inputs
    assert isinstance(input_list_node, FixedTensorList)
    assert len(input_list_node.data) == len(node.outputs)
    target_shape = list(node.outputs[0].shape)
    for idx, (in_data, _out_node) in enumerate(
        zip(input_list_node.data, node.outputs, strict=True)
    ):
        src_ref = op_helper.get_or_add_tensor_variable_in_nnef(in_data)
        _emit_broadcast_to(op_helper, node, src_ref, target_shape, idx)
    return ["tract_core"]


def _emit_meshgrid_one_axis(
    op_helper, node, in_data, target_shape_attr, axis, output_idx
):
    """Emit one meshgrid output: reshape + broadcast to target shape.

    Reshape `in_data` (rank 1) to `(1, .., size, .., 1)` at `axis`,
    then broadcast to `target_shape_attr` and write to
    `node.outputs[output_idx]`. `target_shape_attr` is a list of
    op-attr values (`int` or `nnef.Identifier`), allowing the
    runtime-extracted symbolic size for dynamic axes.
    """
    g = op_helper.g
    name_to_tensor = op_helper.name_to_tensor
    rank = len(target_shape_attr)
    # Use the 1-D input's axis-0 size for the inserted axis (resolved
    # symbolically when that axis is dynamic).
    axis_size = resolve_attr_axis_size(op_helper, in_data, axis=0)
    reshape_shape = [1] * rank
    reshape_shape[axis] = axis_size
    # The intermediate NTensor's `.shape` is metadata used by t2n's
    # rank-align pass; tract reads the actual size from the reshape
    # op attribute. For dynamic axes we just write 1 there -- the
    # reshape result is rank-correct and tract will derive the actual
    # size from the symbolic attribute at runtime.
    onode = node.outputs[output_idx]
    declared_shape = [1] * rank
    if isinstance(axis_size, int):
        declared_shape[axis] = axis_size
    src_ref = op_helper.get_or_add_tensor_variable_in_nnef(in_data)
    intermediate = _make_ntensor_with_shape(
        g,
        name_to_tensor,
        f"{onode.export_name}_mg_reshape",
        declared_shape,
        onode.np_dtype,
    )
    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="reshape",
        name=f"{intermediate.name}_op",
        inputs=(src_ref,),
        outputs=intermediate,
        attribs={"shape": reshape_shape},
    )
    _emit_broadcast_to(
        op_helper, node, intermediate, target_shape_attr, output_idx
    )


@OP_REGISTRY.register()
def meshgrid(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:meshgrid' to NNEF.

    `meshgrid([t0, .., tN-1], indexing)` returns N rank-N tensors. Each
    output is the corresponding input reshaped to put its size on the
    proper axis (then broadcast to the full N-dim shape). With
    `indexing='ij'` axis i is input i; with `indexing='xy'` the first
    two axes are swapped (xy is matrix-style, ij is index-style).
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    if len(node.inputs) == 2:
        input_list_node, indexing_node = node.inputs
        indexing = indexing_node.data
    else:
        (input_list_node,) = node.inputs
        indexing = "ij"
    assert isinstance(input_list_node, FixedTensorList)
    assert len(input_list_node.data) == len(node.outputs)
    n = len(input_list_node.data)

    def axis_for_input(i: int) -> int:
        if indexing == "xy" and n >= 2:
            if i == 0:
                return 1
            if i == 1:
                return 0
        return i

    # Build the broadcast target shape from each input's axis-0 size.
    # The size resolver emits a `tract_core_shape_of` chain and returns
    # an `nnef.Identifier` for dynamic inputs; otherwise it's a plain int.
    target_shape_attr = [None] * n
    for i, in_data in enumerate(input_list_node.data):
        target_shape_attr[axis_for_input(i)] = resolve_attr_axis_size(
            op_helper, in_data, axis=0
        )

    for i, in_data in enumerate(input_list_node.data):
        _emit_meshgrid_one_axis(
            op_helper,
            node,
            in_data,
            target_shape_attr,
            axis=axis_for_input(i),
            output_idx=i,
        )
    return ["tract_core"]


@OP_REGISTRY.register()
def unfold(node, op_helper, **kwargs):
    """Map PyTorch `aten::unfold` (Tensor.unfold) to NNEF.

    Signature: `unfold(self, dimension, size, step)`. Extracts overlapping
    windows of length `size` along `dimension`, advancing by `step`.
    The result has rank `R + 1`: the `dimension` axis becomes
    `n_windows = (D - size) // step + 1`, and a new trailing axis of
    length `size` is appended.

    Decomposed as `n_windows` `slice` ops along `dimension` followed by
    a `stack` along that same axis; if `dimension` is not the last
    axis of the input, an extra `transpose` moves the size axis to the
    end (matching torch's "appended-at-back" layout).
    """
    input_node, dim_node, size_node, step_node = node.inputs
    if not isinstance(dim_node, PythonConstant):
        raise T2NErrorNotImplemented("unfold requires a static dimension")
    if not isinstance(size_node, PythonConstant) or not isinstance(
        step_node, PythonConstant
    ):
        raise T2NErrorNotImplemented("unfold requires static size and step")
    axis = pick_axis(input_node, dim_node.data)
    size = int(size_node.data)
    step = int(step_node.data)
    if size <= 0 or step <= 0:
        raise T2NErrorNotImplemented(
            f"unfold needs positive size/step; got size={size}, step={step}"
        )
    dim_size = int(input_node.shape[axis])
    if dim_size < size:
        raise T2NErrorNotImplemented(
            f"unfold: dim size {dim_size} smaller than window {size}"
        )
    n_windows = (dim_size - size) // step + 1
    input_rank = input_node.rank
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    # Emit one slice per window and stack along `axis`.
    window_refs = []
    for j in range(n_windows):
        begin = j * step
        window_refs.append(
            op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "slice",
                inputs=inp,
                attrs={
                    "axes": [axis],
                    "begin": [begin],
                    "end": [begin + size],
                    "stride": [1],
                },
                output_tensor_name_suffix=f"_unfold_w{j}",
            )
        )
    # After stack, shape is (..., n_windows, size, *rest_after_axis).
    # Torch wants size at the back; if `axis` is already the last input
    # axis then the size axis lands at the end naturally and we emit the
    # final tensor directly from `stack`. Otherwise emit `stack` to an
    # intermediate and `transpose` it to move the size axis to the end.
    if axis + 1 == input_rank:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "stack",
            inputs=window_refs,
            attrs={"axis": axis},
            ensure_tuple=False,
        )
        return []
    stacked = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "stack",
        inputs=window_refs,
        attrs={"axis": axis},
        ensure_tuple=False,
        output_tensor_name_suffix="_unfold_stack",
    )
    stacked_rank = input_rank + 1
    perm = (
        list(range(axis + 1)) + list(range(axis + 2, stacked_rank)) + [axis + 1]
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=stacked,
        attrs={"axes": perm},
    )
    return []


def _as_pair(node_value, name: str):
    """Unwrap PyTorch's `[h, w]` list inputs into a (h, w) int tuple."""
    if hasattr(node_value, "data"):
        node_value = node_value.data
    if hasattr(node_value, "tolist"):
        node_value = node_value.tolist()
    if not isinstance(node_value, (list, tuple)) or len(node_value) != 2:
        raise T2NErrorNotImplemented(
            f"{name} must be a 2-element list; got {node_value!r}"
        )
    return int(node_value[0]), int(node_value[1])


@OP_REGISTRY.register()
def im2col(node, op_helper, **kwargs):
    """Map PyTorch `aten::im2col` (a.k.a. `F.unfold`) to NNEF.

    Signature: `im2col(self, kernel_size, dilation, padding, stride)` for
    a rank-4 input `(N, C, H, W)`. Output is `(N, C * kH * kW, L)` where
    `L = oH * oW` and:

    * `oH = (H + 2*pH - dH*(kH - 1) - 1) // sH + 1`
    * `oW = (W + 2*pW - dW*(kW - 1) - 1) // sW + 1`

    No tract / NNEF op exposes this directly (we probed
    `tract_core_im2col` / `im2col` -- both unknown), so we decompose:

    1. zero-pad the input along H and W if `padding > 0`;
    2. for every kernel position `(di, dj)`, take a strided
       2-axis `slice` -- begin=`[di*dH, dj*dW]`, stride=`[sH, sW]`,
       length `oH x oW`;
    3. `stack` the `kH * kW` slices along a new axis at position 2,
       then `reshape` `(N, C, kH*kW, oH, oW)` -> `(N, C*kH*kW, oH*oW)`.

    Iteration order is `di` outer, `dj` inner, matching torch's flat
    output-channel index `c*kH*kW + di*kW + dj`.
    """
    (
        input_node,
        kernel_node,
        dilation_node,
        padding_node,
        stride_node,
    ) = node.inputs
    if input_node.rank != 4:
        raise T2NErrorNotImplemented(
            f"im2col expects a rank-4 input (N, C, H, W); got rank "
            f"{input_node.rank}"
        )
    kh, kw = _as_pair(kernel_node, "kernel_size")
    dh, dw = _as_pair(dilation_node, "dilation")
    ph, pw = _as_pair(padding_node, "padding")
    sh, sw = _as_pair(stride_node, "stride")
    n, c, h, w = (int(d) for d in input_node.shape)
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    rcpt_h = dh * (kh - 1) + 1
    rcpt_w = dw * (kw - 1) + 1
    if padded_h < rcpt_h or padded_w < rcpt_w:
        raise T2NErrorNotImplemented(
            f"im2col: padded input ({padded_h}, {padded_w}) too small "
            f"for receptive field ({rcpt_h}, {rcpt_w})"
        )
    o_h = (padded_h - rcpt_h) // sh + 1
    o_w = (padded_w - rcpt_w) // sw + 1
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if ph > 0 or pw > 0:
        inp = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "pad",
            inputs=inp,
            attrs={
                "padding": [(0, 0), (0, 0), (ph, ph), (pw, pw)],
                "value": 0.0,
            },
            output_tensor_name_suffix="_im2col_pad",
        )
    slice_refs = []
    for di in range(kh):
        for dj in range(kw):
            begin_h = di * dh
            begin_w = dj * dw
            slice_refs.append(
                op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "slice",
                    inputs=inp,
                    attrs={
                        "axes": [2, 3],
                        "begin": [begin_h, begin_w],
                        "end": [
                            begin_h + (o_h - 1) * sh + 1,
                            begin_w + (o_w - 1) * sw + 1,
                        ],
                        "stride": [sh, sw],
                    },
                    output_tensor_name_suffix=f"_im2col_s{di}_{dj}",
                )
            )
    stacked = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "stack",
        inputs=slice_refs,
        attrs={"axis": 2},
        ensure_tuple=False,
        output_tensor_name_suffix="_im2col_stack",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=stacked,
        attrs={"shape": [n, c * kh * kw, o_h * o_w]},
    )
    return []


@OP_REGISTRY.register()
def col2im(node, op_helper, **kwargs):
    """Map PyTorch `aten::col2im` (a.k.a. `F.fold`) to NNEF.

    Signature: `col2im(self, output_size, kernel_size, dilation, padding,
    stride)` for a rank-3 input `(N, C * kH * kW, L)`. Inverse of
    `im2col`: places each of the `kH * kW` "kernel offsets" of the input
    at strided positions inside a `(N, C, output_H + 2*pH, output_W +
    2*pW)` canvas, summing overlaps, then crops the canvas to
    `(N, C, output_H, output_W)`.

    Tract has no NNEF-level `col2im` / `scatter_add-with-reduction` on
    the version we target, so we decompose per kernel offset:

    1. reshape input `(N, C*kH*kW, n_h*n_w)` -> `(N, C, kH, kW, n_h, n_w)`;
    2. for every `(di, dj)`:
       a. slice the per-offset feature map `(N, C, n_h, n_w)`;
       b. spread it to `(N, C, n_h*sH, n_w*sW)` -- reshape with two
          size-1 axes then `pad` axis 3 by `(0, sH-1)` and axis 5 by
          `(0, sW-1)` to insert zeros between elements -- then reshape
          to flatten the spread axes; trim trailing zeros by slicing to
          `((n_h-1)*sH + 1, (n_w-1)*sW + 1)`;
       c. pad to the canvas size `(padded_H, padded_W)` with left/top
          offsets `(di*dH, dj*dW)`;
    3. sum the `kH * kW` placed contributions (`add` chain);
    4. crop the leading / trailing `(pH, pW)` rows / cols of the
       canvas to land on `(N, C, output_H, output_W)`.

    A future tract release that exposes a native col2im (or scatter-add
    with sum reduction) lets us replace this chain with a single op
    behind a version gate.
    """
    (
        input_node,
        output_size_node,
        kernel_node,
        dilation_node,
        padding_node,
        stride_node,
    ) = node.inputs
    if input_node.rank != 3:
        raise T2NErrorNotImplemented(
            f"col2im expects rank-3 input (N, C*kH*kW, L); got rank "
            f"{input_node.rank}"
        )
    out_h, out_w = _as_pair(output_size_node, "output_size")
    kh, kw = _as_pair(kernel_node, "kernel_size")
    dh, dw = _as_pair(dilation_node, "dilation")
    ph, pw = _as_pair(padding_node, "padding")
    sh, sw = _as_pair(stride_node, "stride")
    n = int(input_node.shape[0])
    channels_packed = int(input_node.shape[1])
    if channels_packed % (kh * kw) != 0:
        raise T2NErrorNotImplemented(
            f"col2im: input channel dim {channels_packed} not divisible "
            f"by kH*kW={kh * kw}"
        )
    c = channels_packed // (kh * kw)
    padded_h = out_h + 2 * ph
    padded_w = out_w + 2 * pw
    rcpt_h = dh * (kh - 1) + 1
    rcpt_w = dw * (kw - 1) + 1
    if padded_h < rcpt_h or padded_w < rcpt_w:
        raise T2NErrorNotImplemented(
            f"col2im: padded output ({padded_h}, {padded_w}) too small "
            f"for receptive field ({rcpt_h}, {rcpt_w})"
        )
    n_h = (padded_h - rcpt_h) // sh + 1
    n_w = (padded_w - rcpt_w) // sw + 1
    if int(input_node.shape[2]) != n_h * n_w:
        raise T2NErrorNotImplemented(
            f"col2im: input L={input_node.shape[2]} != n_h*n_w={n_h * n_w}"
        )

    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    # Step 1: (N, C*kH*kW, L) -> (N, C, kH, kW, n_h, n_w).
    reshaped = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=inp,
        attrs={"shape": [n, c, kh, kw, n_h, n_w]},
        output_tensor_name_suffix="_col2im_reshape",
    )

    # Compactly-named helper for the trim length of the spread block.
    spread_h_trim = (n_h - 1) * sh + 1
    spread_w_trim = (n_w - 1) * sw + 1

    contribution_refs = []
    for di in range(kh):
        for dj in range(kw):
            slc = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "slice",
                inputs=reshaped,
                attrs={
                    "axes": [2, 3],
                    "begin": [di, dj],
                    "end": [di + 1, dj + 1],
                    "stride": [1, 1],
                },
                output_tensor_name_suffix=f"_c2i_sl{di}_{dj}",
            )
            # Drop the now-singleton (kH, kW) axes -> (N, C, n_h, n_w).
            slc = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "squeeze",
                inputs=slc,
                attrs={"axes": [2, 3]},
                output_tensor_name_suffix=f"_c2i_sq{di}_{dj}",
            )
            # Spread with stride: (n_h, n_w) -> (n_h*sH, n_w*sW) with
            # zeros between elements. Only needed when stride > 1.
            if sh > 1 or sw > 1:
                slc = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "reshape",
                    inputs=slc,
                    attrs={"shape": [n, c, n_h, 1, n_w, 1]},
                    output_tensor_name_suffix=f"_c2i_rs{di}_{dj}",
                )
                slc = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "pad",
                    inputs=slc,
                    attrs={
                        "padding": [
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, sh - 1),
                            (0, 0),
                            (0, sw - 1),
                        ],
                        "value": 0.0,
                    },
                    output_tensor_name_suffix=f"_c2i_pd{di}_{dj}",
                )
                slc = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "reshape",
                    inputs=slc,
                    attrs={"shape": [n, c, n_h * sh, n_w * sw]},
                    output_tensor_name_suffix=f"_c2i_fl{di}_{dj}",
                )
                # Trim trailing zero rows / cols.
                slc = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "slice",
                    inputs=slc,
                    attrs={
                        "axes": [2, 3],
                        "begin": [0, 0],
                        "end": [spread_h_trim, spread_w_trim],
                        "stride": [1, 1],
                    },
                    output_tensor_name_suffix=f"_c2i_tr{di}_{dj}",
                )
            # Pad into the canvas at offset (di*dH, dj*dW).
            pad_top = di * dh
            pad_bottom = padded_h - pad_top - spread_h_trim
            pad_left = dj * dw
            pad_right = padded_w - pad_left - spread_w_trim
            slc = op_helper.add_single_output_op_from_nnef_tensors(
                node,
                "pad",
                inputs=slc,
                attrs={
                    "padding": [
                        (0, 0),
                        (0, 0),
                        (pad_top, pad_bottom),
                        (pad_left, pad_right),
                    ],
                    "value": 0.0,
                },
                output_tensor_name_suffix=f"_c2i_cv{di}_{dj}",
            )
            contribution_refs.append(slc)

    # Step 3: sum kH*kW contributions via a chain of binary `add` ops.
    accumulated = contribution_refs[0]
    n_contribs = len(contribution_refs)
    for idx in range(1, n_contribs):
        accumulated = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "add",
            inputs=(accumulated, contribution_refs[idx]),
            output_tensor_name_suffix=f"_c2i_acc{idx}",
        )
    # Step 4: crop the (pH, pW) padding. We always emit the final
    # `slice` (even when `pH == pW == 0` it just runs as a whole-tensor
    # slice) so the canvas accumulator never needs to double as the
    # named final output.
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "slice",
        inputs=accumulated,
        attrs={
            "axes": [2, 3],
            "begin": [ph, pw],
            "end": [ph + out_h, pw + out_w],
            "stride": [1, 1],
        },
    )
    return []
