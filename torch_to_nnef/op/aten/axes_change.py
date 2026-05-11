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
