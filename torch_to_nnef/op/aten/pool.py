import typing as T

import nnef
import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.aten.reducer import reducer_helper
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    get_tract_dyn_axis_size_soc,
)
from torch_to_nnef.torch_graph import Data
from torch_to_nnef.torch_graph.ir_data import PythonConstant, TensorVariable

OP_REGISTRY = AtenOpRegistry()

TRACT_SUPPORT_DYNAMIC_POOLING = False  # to update if that's the case 1 day


def _pooling_op(
    nnef_op_name: str,
    node_inputs: T.List[Data],
    node,
    op_helper,
):
    """Generic pool operator translation from aten to NNEF.

    NNEF (avg|max)_pool params (not dimension specific):.
    input: tensor<scalar>,
    size: integer[],
    border: string = 'constant',
    padding: (integer,integer)[] = [],
    stride: integer[] = [],
    dilation: integer[] = [] )

    """
    (
        input_node,
        kernel_size_node,
        stride_node,
        padding_node,
        dilation_node,
        ceil_mode_node,
    ) = node_inputs

    if ceil_mode_node and ceil_mode_node.data:
        raise T2NErrorNotImplemented(
            "Use of ceil to compute output shape is not implem"
        )

    padding = padding_node.data or []
    kernel_size = kernel_size_node.data or []
    stride = stride_node.data or []
    # PyTorch defaults `stride` to `kernel_size` when empty / unspecified;
    # tract's pool ops require stride length matching the input rank, so
    # propagate that default here.
    if not stride:
        stride = list(kernel_size)
    if dilation_node:
        dilation = dilation_node.data or []
    else:
        dilation = [1 for _ in stride]

    # peculiarity of tract implementation
    # apparently tract does expect max_pool to be always 2d only (including
    # input.shape)
    onode = node.outputs[0]
    if onode.rank > len(kernel_size):
        missing_n_dims = onode.rank - len(kernel_size)
        kernel_size = ([1] * missing_n_dims) + kernel_size
        stride = ([1] * missing_n_dims) + stride
        dilation = ([1] * missing_n_dims) + dilation

        # pre 0.19.0 padding order differ
        if (
            isinstance(op_helper.inference_target, TractNNEF)
            and op_helper.inference_target.version < "0.19.0"
        ):
            padding = padding + ([0] * missing_n_dims)
        else:
            padding = ([0] * missing_n_dims) + padding

    inputs = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    attrs = {
        "size": list(kernel_size),
        "padding": [
            (pad, pad) if isinstance(pad, int) else pad for pad in padding
        ],
        "stride": list(stride),
        "dilation": list(dilation),
        "border": "constant",
    }
    if len(node.outputs) == 1:
        return op_helper.add_single_output_op_from_nnef_tensors(
            node,
            nnef_op_name,
            inputs=inputs,
            attrs=attrs,
        )
    if len(node.outputs) == 2:
        return op_helper.add_multi_output_op_from_nnef_tensors(
            node,
            nnef_op_name,
            inputs=inputs,
            attrs=attrs,
        )

    raise T2NErrorNotImplemented(
        f"Pooling with {len(node.outputs)} outputs "
        "is not supported in NNEF/TractNNEF yet"
    )


@OP_REGISTRY.register()
def max_pool1d(g, node, op_helper, **kwargs):
    """Map PyTorch: 'aten:max_pool1d' to NNEF."""
    _pooling_op("max_pool", node.inputs, node, op_helper)


@OP_REGISTRY.register()
def avg_pool1d(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:avg_pool1d' to NNEF."""
    count_include_pad = node.inputs[-1].data
    if not count_include_pad:
        raise T2NErrorNotImplemented("not implemented count_include_pad=False")
    inputs_name_tuple = node.inputs[:-1]  # count_include_pad excluded
    inputs_name_tuple.insert(4, None)  # set missing dilation

    # Dilation is available
    _pooling_op("avg_pool", inputs_name_tuple, node, op_helper)


@OP_REGISTRY.register(["max_pool2d", "max_pool3d"])
def max_pool_nd(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:max_pool2d', 'aten:max_pool3d' to NNEF."""
    _pooling_op("max_pool", node.inputs, node, op_helper)


@OP_REGISTRY.register(
    [
        "max_pool1d_with_indices",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
    ]
)
def max_pool_nd_with_indices(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:max_pool{1,2,3}d_with_indices' to NNEF.

    Lowers to NNEF stdlib's `max_pool_with_index` fragment which
    returns both the pooled values and the (per-window argmax)
    indices. Tract only -- the fragment requires the
    `argmax_pool` + `sample` primitives behind it.
    """
    if not isinstance(op_helper.inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "max_pool_with_index is not supported in TractNNEF yet"
        )
    _pooling_op("max_pool_with_index", node.inputs, node, op_helper)
    return ["tract_core"]


@OP_REGISTRY.register(["avg_pool2d", "avg_pool3d"])
def avg_pool_nd(node, op_helper, **kwargs):
    """Map PyTorch: 'aten:avg_pool(2|3)d', 'aten:max_pool3d' to NNEF.

    Cpp func parameters:.
    (const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override

    _pooling_op expect:

    (input_node,
    kernel_size_node,
    stride_node,
    padding_node,
    dilation_node,
    ceil_mode_node)
    """
    count_include_pad = node.inputs[-2].data
    if not count_include_pad:
        raise T2NErrorNotImplemented("not implemented count_include_pad=False")

    divisor_overide = node.inputs[-1].data
    if divisor_overide:
        raise T2NErrorNotImplemented(
            f"not implemented divisor_override={divisor_overide}"
        )
    inputs_tups = node.inputs[:-2]
    inputs_tups.insert(4, None)
    _pooling_op("avg_pool", inputs_tups, node, op_helper)


def _adaptive_pool(nnef_op_name: str, op_helper, node):
    (
        input_node,
        pool_values_node,
    ) = node.inputs

    pool_values = pool_values_node.data
    if not all(
        dim and dim > 0 for dim in input_node.shape[-len(pool_values) :]
    ):
        raise T2NErrorNotImplemented(
            "dynamic dim used in adaptive pool is not Implemented yet"
        )
    # fixed at export auto adaptation
    onode = node.outputs[0]
    is_reducer = all(pv == 1 for pv in pool_values)
    if (
        TRACT_SUPPORT_DYNAMIC_POOLING
        and isinstance(op_helper.inference_target, TractNNEF)
        and op_helper.inference_target.has_dynamic_axes
        and not is_reducer
    ):
        stride = []
        start_ix = input_node.rank - len(pool_values) - 1
        for axis_offset, pool_val in zip(
            range(start_ix, input_node.rank),
            pool_values,
            strict=False,
        ):
            axis = start_ix + axis_offset
            soc = get_tract_dyn_axis_size_soc(op_helper, input_node, axis=axis)
            numerator_nnef = op_helper.name_to_tensor[soc.output_name]
            if pool_val == 1:
                out = numerator_nnef
            else:
                pool_val_nnef = op_helper.get_or_add_tensor_variable_in_nnef(
                    PythonConstant(
                        name=f"{onode.export_name}_pool_val{axis}",
                        data=pool_val,
                    )
                )
                out = op_helper.add_single_output_op_from_nnef_tensors(
                    node,
                    "div",
                    inputs=(
                        numerator_nnef,
                        pool_val_nnef,
                    ),
                    output_tensor_name_suffix=f"stride_{axis}",
                    maybe_cast_align_tract=False,  # here you want to stay TDim
                )
            stride.append(nnef.Identifier(out.name))
    else:
        stride = [
            int(in_tensor_dim // pool_val)
            for pool_val, in_tensor_dim in zip(
                pool_values, input_node.shape[-len(pool_values) :], strict=False
            )
        ]

    if onode.rank > len(stride):
        missing_n_dims = onode.rank - len(stride)
        stride = ([1] * missing_n_dims) + stride

    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if is_reducer:
        reduce_node = node
        axes_node = reduce_node.inputs[1]
        axes_node.name += "_reducer"
        axes_node.set_data(
            [input_node.rank - _ - 1 for _ in range(len(axes_node.data))][::-1]
        )
        node.inputs.append(
            PythonConstant(
                name=f"{reduce_node.outputs[0].export_name}_keep_dim", data=True
            )
        )

        return reducer_helper(
            {
                "max_pool": "max_reduce",
                "avg_pool": "mean_reduce",
            }[nnef_op_name],
            reduce_node,
            op_helper,
        )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        nnef_op_name,
        inputs=inp,
        attrs={
            "size": list(stride),
            "padding": [(0, 0) for _ in stride],
            "stride": list(stride),
            "dilation": [1 for _ in stride],
            "border": "ignore",
        },
    )
    return []


# warning! no support for return_indice=True
@OP_REGISTRY.register(
    ["adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d"]
)
def adaptive_avg_poolnd(g, node, op_helper, **kwargs):
    """Map PyTorch: 'aten:adaptive_avg_pool{1,2,3}d' to NNEF."""
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("avg_pool", op_helper, node)


# warning! no support for return_indice=True
@OP_REGISTRY.register(
    ["adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"]
)
def adaptive_max_poolnd(node, op_helper, **kwargs):
    """Map PyTorch: adaptive_max_pool{1,2,3}d to NNEF."""
    node.outputs = node.outputs[:1]
    # WARNING will liklely only work with full defined shapes in shape
    _adaptive_pool("max_pool", op_helper, node)


# Min tract release exposing `tract_core_resize` / `tract_core_grid_sample`
# (the clean Resize subset + GridSample moved into tract-core). Set to the
# release that ships the binding once it is published.
RESIZE_MIN_TRACT_VERSION = "0.24.0"


def _is_tract_with_resize(inference_target) -> bool:
    return (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= RESIZE_MIN_TRACT_VERSION
    )


def _const_f32_vector(op_helper, name: str, values: T.List[float]):
    """Add a 1-D f32 constant and return an `nnef.Identifier` to it."""
    const = PythonConstant(
        name=name, data=torch.tensor(values, dtype=torch.float32)
    )
    ref = op_helper.get_or_add_tensor_variable_in_nnef(const)
    return nnef.Identifier(ref.name)


def _emit_core_resize(
    node,
    op_helper,
    *,
    interpolator: str,
    coord_transformer: str,
    nearest_mode: str = "floor",
):
    """Emit `tract_core_resize` for a PyTorch `upsample_*` node.

    `scale_factors`, when present, become a constant full-rank `scales`
    vector (leading non-spatial axes get scale `1.0`); this stays correct
    with dynamic input spatial dims since the output dim is
    `input_dim * scale`. Otherwise the explicit `output_size` becomes a
    constant full-rank `sizes` vector, which requires statically known
    non-spatial dims (a fully dynamic case would need runtime `shape_of`).
    """
    input_node = node.inputs[0]
    size_node = node.inputs[1]
    scale_factor_node = node.inputs[-1]
    input_rank = input_node.rank
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    out_name = node.outputs[0].name

    attrs = {
        "interpolator": interpolator,
        "coord_transformer": coord_transformer,
        "nearest_mode": nearest_mode,
    }

    scales = getattr(scale_factor_node, "data", None)
    if scales is not None and all(isinstance(s, float) for s in scales):
        leading = input_rank - len(scales)
        full = [1.0] * leading + [float(s) for s in scales]
        attrs["scales"] = _const_f32_vector(
            op_helper, f"{out_name}_resize_scales", full
        )
    else:
        sizes = getattr(size_node, "data", None)
        if not sizes:
            raise T2NErrorNotImplemented(
                "upsample without scale_factor or output_size"
            )
        if hasattr(sizes, "tolist"):
            sizes = sizes.tolist()
        spatial = [int(s) for s in sizes]
        leading = input_rank - len(spatial)
        lead_dims = []
        for dim in input_node.shape[:leading]:
            dim_int = (
                dim if isinstance(dim, int) else getattr(dim, "data", None)
            )
            if not isinstance(dim_int, int):
                raise T2NErrorNotImplemented(
                    "upsample by output_size needs statically known "
                    f"non-spatial dims (got shape {input_node.shape}); the "
                    "dynamic case needs runtime shape_of (not yet supported)"
                )
            lead_dims.append(dim_int)
        full = [float(d) for d in lead_dims] + [float(s) for s in spatial]
        attrs["sizes"] = _const_f32_vector(
            op_helper, f"{out_name}_resize_sizes", full
        )

    op_helper.add_single_output_op_from_nnef_tensors(
        node, "tract_core_resize", inputs=inp, attrs=attrs
    )
    return ["tract_core"]


@OP_REGISTRY.register(
    ["upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d"]
)
def upsample_nearest_nd(node, op_helper, **kwargs):
    """Map PyTorch `aten::upsample_nearest{1,2,3}d` to NNEF.

    On tract releases exposing `tract_core_resize` this lowers to a single
    `resize` (nearest / asymmetric / floor). Older targets fall back to the
    `debox` path (tract >= 0.22 with `upsample_with_debox=True`), the
    rank-generic reshape/tile trick, or the legacy 2-D `deconv`.
    """
    if _is_tract_with_resize(op_helper.inference_target):
        return _emit_core_resize(
            node,
            op_helper,
            interpolator="nearest",
            coord_transformer="asymmetric",
            nearest_mode="floor",
        )
    (input_node, size_node, scale_factor_node) = node.inputs
    if size_node.data:
        raise T2NErrorNotImplemented("size in upsampling not defined in NNEF")
    if scale_factor_node.data is None or not all(
        isinstance(_, float) for _ in scale_factor_node.data
    ):
        raise T2NErrorNotImplemented(
            f"unable to export scale_factor {scale_factor_node.data}"
        )

    scales = [int(sf) for sf in scale_factor_node.data]
    spatial_rank = len(scales)
    input_rank = input_node.rank
    # NNEF convention here mirrors torch's `(N, C, *spatial)` layout
    # so the leading non-spatial axes are upsampled with stride 1.
    leading = input_rank - spatial_rank
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if (
        isinstance(op_helper.inference_target, TractNNEF)
        and op_helper.inference_target.version > "0.22.0"
        and op_helper.inference_target.upsample_with_debox
    ):
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "debox",
            inputs=inp,
            attrs={
                "size": [1] * leading + scales,
                "stride": [1] * leading + scales,
                "padding": [(0, 0)] * input_rank,
            },
        )
        return []
    static_shapes = not (
        isinstance(op_helper.inference_target, TractNNEF)
        and op_helper.inference_target.has_dynamic_axes
    )
    if spatial_rank != 2 or static_shapes:
        # Rank-generic nearest upsample: insert a size-1 axis after
        # each spatial axis, tile it by the scale, then collapse back.
        # `(..., d, 1) -> tile by s -> (..., d, s) -> reshape (..., d*s)`
        # is the standard reshape/tile trick for nearest-neighbour
        # replication; works on any rank and bypasses the rank-4
        # restriction of the deconv path. It needs static spatial dims,
        # so the deconv path below still handles the dynamic-axes case.
        # Preferred over deconv: the deconv lowering emits a broadcast
        # `Mul` that tract 0.23.0's `OptMatMul` fuse pass mis-substitutes
        # when an adjacent conv consumes the upsample output.
        in_shape = [int(d) for d in input_node.shape]
        expanded = list(in_shape[:leading])
        for axis_i in range(spatial_rank):
            expanded.extend([in_shape[leading + axis_i], 1])
        out_intermediate = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "reshape",
            inputs=inp,
            attrs={"shape": expanded},
            output_tensor_name_suffix="_up_reshape",
        )
        tile_repeats = [1] * leading
        for axis_i in range(spatial_rank):
            tile_repeats.extend([1, scales[axis_i]])
        out_intermediate = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tile",
            inputs=out_intermediate,
            attrs={"repeats": tile_repeats},
            output_tensor_name_suffix="_up_tile",
        )
        final_shape = list(in_shape[:leading])
        for axis_i in range(spatial_rank):
            final_shape.append(in_shape[leading + axis_i] * scales[axis_i])
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "reshape",
            inputs=out_intermediate,
            attrs={"shape": final_shape},
        )
        return []
    # NOTE: legacy 2-D path. Suboptimal compared to ONNX `resize`;
    # ideally tract grows a proper `debox` for older versions too.
    kernel_data = torch.ones([1, 1, 1, 1] + scales)
    kernel = TensorVariable(
        name=f"{node.outputs[0].export_name}_kernel",
        data=kernel_data,
        shape=kernel_data.shape,
        dtype=input_node.dtype,
    )
    bias = PythonConstant(
        name=f"{node.outputs[0].export_name}_bias",
        data=0,
    )
    out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "deconv",
        inputs=(
            inp,
            op_helper.get_or_add_tensor_variable_in_nnef(kernel),
            op_helper.get_or_add_tensor_variable_in_nnef(bias),
        ),
        attrs={
            "stride": [1, 1] + scales,
            "padding": [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        },
        output_tensor_name_suffix="_deconv",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "squeeze",
        inputs=out,
        attrs={"axes": [0, 1]},
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    return []


@OP_REGISTRY.register(
    [
        "_upsample_nearest_exact1d",
        "_upsample_nearest_exact2d",
        "_upsample_nearest_exact3d",
    ]
)
def upsample_nearest_exact_nd(node, op_helper, **kwargs):
    """Map `aten::_upsample_nearest_exact{1,2,3}d` to `tract_core_resize`.

    The "exact" variant centres samples (half-pixel) and rounds, unlike
    plain nearest which floors from the asymmetric grid.
    """
    if not _is_tract_with_resize(op_helper.inference_target):
        raise T2NErrorNotImplemented(
            "_upsample_nearest_exact* needs tract_core_resize "
            f"(tract >= {RESIZE_MIN_TRACT_VERSION})"
        )
    return _emit_core_resize(
        node,
        op_helper,
        interpolator="nearest",
        coord_transformer="half_pixel",
        nearest_mode="round_prefer_ceil",
    )


@OP_REGISTRY.register(
    ["upsample_linear1d", "upsample_bilinear2d", "upsample_trilinear3d"]
)
def upsample_linear_nd(node, op_helper, **kwargs):
    """Map `aten::upsample_{linear1d,bilinear2d,trilinear3d}` to resize.

    `align_corners` selects the coordinate transform; PyTorch's
    `align_corners=False` matches ONNX `pytorch_half_pixel`.
    """
    if not _is_tract_with_resize(op_helper.inference_target):
        raise T2NErrorNotImplemented(
            "linear upsample needs tract_core_resize "
            f"(tract >= {RESIZE_MIN_TRACT_VERSION})"
        )
    align_corners = bool(node.inputs[2].data)
    coord = "align_corners" if align_corners else "pytorch_half_pixel"
    return _emit_core_resize(
        node, op_helper, interpolator="linear", coord_transformer=coord
    )


@OP_REGISTRY.register(["upsample_bicubic2d"])
def upsample_bicubic2d(node, op_helper, **kwargs):
    """Map `aten::upsample_bicubic2d` to `tract_core_resize` (cubic)."""
    if not _is_tract_with_resize(op_helper.inference_target):
        raise T2NErrorNotImplemented(
            "bicubic upsample needs tract_core_resize "
            f"(tract >= {RESIZE_MIN_TRACT_VERSION})"
        )
    align_corners = bool(node.inputs[2].data)
    coord = "align_corners" if align_corners else "pytorch_half_pixel"
    return _emit_core_resize(
        node, op_helper, interpolator="cubic", coord_transformer=coord
    )


@OP_REGISTRY.register(["grid_sampler", "grid_sampler_2d", "grid_sampler_3d"])
def grid_sampler(node, op_helper, **kwargs):
    """Map `aten::grid_sampler{,_2d,_3d}` to `tract_core_grid_sample`.

    `(input, grid, interpolation_mode, padding_mode, align_corners)` with
    the integer enums decoded to tract's string options.
    """
    if not _is_tract_with_resize(op_helper.inference_target):
        raise T2NErrorNotImplemented(
            "grid_sampler needs tract_core_grid_sample "
            f"(tract >= {RESIZE_MIN_TRACT_VERSION})"
        )
    input_node, grid_node, interp_node, padding_node, align_node = node.inputs
    mode = {0: "bilinear", 1: "nearest", 2: "bicubic"}[int(interp_node.data)]
    padding = {0: "zeros", 1: "border", 2: "reflection"}[int(padding_node.data)]
    align_corners = bool(align_node.data)
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    grid = op_helper.get_or_add_tensor_variable_in_nnef(grid_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_grid_sample",
        inputs=[inp, grid],
        attrs={
            "mode": mode,
            "padding_mode": padding,
            "align_corners": align_corners,
        },
    )
    return ["tract_core"]
