import logging

import nnef
import numpy as np
import torch

from torch_to_nnef.dtypes import (
    NUMPY_DTYPE_TO_STR,
    SCALAR_TYPE_TO_PYTORCH_TYPE,
    TORCH_DTYPE_TO_TRACT_STR,
)
from torch_to_nnef.exceptions import T2NErrorNotImplemented, T2NErrorTract
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    get_list_of_int,
    get_or_add_tensor_variable_in_nnef,
    get_tract_dyn_axis_size_soc,
    resolve_attr_axis_size,
    unary_output_op_without_attr,
)
from torch_to_nnef.torch_graph import (
    MAP_TO_NOP,
    FixedTensorList,
    TensorVariable,
)
from torch_to_nnef.torch_graph.ir_data import PythonConstant

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def arange(g, node, name_to_tensor, inference_target, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    if len(node.inputs) == 4:
        # for now should never happen since dtype info is
        (start_node, end_node, step_node, dtype_node) = node.inputs
    else:
        raise T2NErrorNotImplemented(
            f"arange with {len(node.inputs)} inputs (see `ir_helpers` module)"
        )

    # see SCALAR_TYPE_TO_PYTORCH_TYPE for reference index:
    #   3 = int32, 4 = int64, 6 = float32, 7 = float64
    # float64 shows up in RoPE-style position embeds; NNEF runtimes execute
    # the arange as f32 which is fine for integer-range / position math.
    if dtype_node.data not in [6, None, 4, 3, 7]:
        raise T2NErrorNotImplemented(
            f"dtype {dtype_node} not implemented for arange"
        )

    if inference_target.has_dynamic_axes or isinstance(
        inference_target, TractNNEF
    ):
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorNotImplemented(inference_target)
        if inference_target.version < "0.20.0":
            raise T2NErrorTract(
                "please update to latest tract to use 'tract_core_range'"
            )

        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_range",
            inputs=[
                get_or_add_tensor_variable_in_nnef(
                    g, start_node, name_to_tensor
                ),
                get_or_add_tensor_variable_in_nnef(g, end_node, name_to_tensor),
            ]
            + (
                []
                if isinstance(step_node, PythonConstant)
                else [
                    get_or_add_tensor_variable_in_nnef(
                        g, step_node, name_to_tensor
                    ),
                ]
            ),
            attrs={"step": step_node.data}
            if isinstance(step_node, PythonConstant)
            else {},
        )
        return ["tract_core"]
    if start_node.data is None or end_node.data is None:
        raise T2NErrorNotImplemented(
            "Dynamic arange not handled in strict NNEF For now"
        )

    node.outputs[0].data.set_data(
        torch.arange(start_node.data, end_node.data, step=step_node.data)
    )
    add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
    )
    return []


def _generic_auto_tensor_expansion(
    shape_node,
    node,
    g,
    torch_graph,
    name_to_tensor,
    has_dynamic_axes,
    dtype=torch.float32,
    tensor_build_fn=torch.ones,
):
    """In case the tensor need to be dependant on shape of another."""
    if isinstance(shape_node, (list, tuple)) and all(
        isinstance(d, (int, str)) for d in shape_node
    ):
        dim_data = shape_node
    else:
        dim_data = get_list_of_int(
            shape_node,
            torch_graph,
            name_to_tensor=name_to_tensor,
            has_dynamic_axes=has_dynamic_axes,
        )
    fixed_dim = []
    to_expand_dim = {}
    for dim_idx, dim_any in enumerate(dim_data):
        if isinstance(dim_any, str):
            fixed_dim.append(1)
            to_expand_dim[dim_idx] = dim_any
        else:
            assert isinstance(dim_any, int), dim_any
            fixed_dim.append(dim_any)

    base_tensor_node = node.outputs[0]
    new_output_tensor = tensor_build_fn(fixed_dim, dtype=dtype)
    node.outputs[0].set_data(
        new_output_tensor, force_dtype=True, force_shape=True
    )
    if to_expand_dim and has_dynamic_axes:
        LOGGER.debug(
            "the aten::ones replaced by constant traced values"
            " with additional expansion (follows NNEF spec)."
        )
        cached_input = get_or_add_tensor_variable_in_nnef(
            g, base_tensor_node, name_to_tensor, name_suffix="to_be_expanded"
        )
        repeats = [1 for _ in range(len(fixed_dim))]
        for k, v in to_expand_dim.items():
            repeats[k] = v
        if cached_input.dtype == np.int64:
            cached_input = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "tract_core_cast",
                inputs=cached_input,
                attrs={"to": NUMPY_DTYPE_TO_STR[np.int64]},
                output_tensor_name_suffix=f"{cached_input.name}_casted",
            )

        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tile",
            inputs=cached_input,
            attrs={"repeats": repeats},
        )
    else:
        # late bug catching
        if base_tensor_node.data.dtype != base_tensor_node.dtype:
            LOGGER.warning(
                "late 'dtype' miss-alignment catched in "
                "_generic_auto_tensor_expansion"
            )
            base_tensor_node.set_data(
                base_tensor_node.data.to(base_tensor_node.dtype),
                force_dtype=True,
            )
        add_tensor_variable_node_as_nnef_tensor(
            g,
            base_tensor_node,
            name_to_tensor,
        )


@OP_REGISTRY.register()
def ones(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """This operator can not be exactly exported to NNEF.

    In general NNEF spec is against dynamism it could provide so

    we implement it as a simple constant variable.

    """
    (input_node, *_) = node.inputs
    dtype = torch.float32
    if len(_) > 0:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[_[0].data]
    return _generic_auto_tensor_expansion(
        input_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.ones,
    )


def _x_like(
    g,
    torch_graph,
    name_to_tensor,
    node,
    inference_target,
    tensor_build_fn,
    **kwargs,
):
    (input_node, *_) = node.inputs
    dtype = input_node.dtype
    if len(_) > 0:
        dtype_node = _[0]
        # `dtype_node.data` is the integer scalar-type code; 0 maps to
        # `torch.uint8` and is FALSY in Python. `if X.data:` would
        # silently drop `dtype=torch.uint8` requests. Use is-None.
        if dtype_node.data is not None:
            dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]

    shape_node = input_node.shape
    if (
        isinstance(inference_target, TractNNEF)
        and inference_target.has_dynamic_axes
    ):
        # in this case we need to get full expansion of input_node shape
        input_tensor = get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        )
        shape_tensor_name = f"{input_tensor.name}_shape"
        shape_tensor = add_single_output_op(
            g,
            node,
            name_to_tensor,
            "tract_core_shape_of",
            inputs=(input_tensor,),
            force_full_output_tensor_name=shape_tensor_name,
        )
        shape_node = FixedTensorList(name="recomposed_shape_node", data=[])
        for dim in range(
            input_node.rank
        ):  # assume always same rank at each graph run
            index_tensor_name = f"{shape_tensor_name}_{dim}"
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "slice",
                inputs=(shape_tensor,),
                attrs={
                    "axes": [0],
                    "begin": [dim],
                    "end": [dim + 1],
                    "stride": [1],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
            index_tensor_name = f"{shape_tensor_name}_{dim}_scalar"
            out = add_single_output_op(
                g,
                node,
                name_to_tensor,
                "squeeze",
                inputs=(out,),
                attrs={
                    "axes": [0],
                },
                force_full_output_tensor_name=index_tensor_name,
            )
            shape_node.data.append(
                TensorVariable(
                    name=str(out.name),
                    data=None,
                    shape=[1],
                    dtype=input_node.dtype,
                )
            )
    return _generic_auto_tensor_expansion(
        shape_node,  # not dynamic for now
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=tensor_build_fn,
    )


@OP_REGISTRY.register()
def zeros_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use exapnsion

    """
    return _x_like(tensor_build_fn=torch.zeros, **kwargs)


def _new_x(
    node,
    g,
    torch_graph,
    name_to_tensor,
    inference_target,
    tensor_build_fn,
    dtype_idx=2,
):
    """Shared body for ``aten::new_{empty,full,ones}``.

    All share the layout
    ``(input, size, [value,] dtype, layout, device, pin_memory)``;
    only `tensor_build_fn` (and the dtype slot for ``new_full``)
    differs. Re-uses `_generic_auto_tensor_expansion` so dynamic-axes
    sizes flow through the same `tract_core_shape_of` + tile path
    as `ones`/`zeros`.
    """
    inputs = node.inputs
    input_node = inputs[0]
    shape_node = inputs[1]
    dtype_node = inputs[dtype_idx]
    if dtype_node.data is not None:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
    else:
        dtype = input_node.dtype
    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=tensor_build_fn,
    )


@OP_REGISTRY.register()
def new_empty(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:new_empty' to NNEF.

    Materialises as zeros (NNEF has no "uninitialized" tensor and
    real callers immediately fill the buffer); shape comes from
    `size`, dtype defaults to the source tensor's dtype.
    """
    return _new_x(
        node,
        g,
        torch_graph,
        name_to_tensor,
        inference_target,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def new_ones(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:new_ones' to NNEF."""
    return _new_x(
        node,
        g,
        torch_graph,
        name_to_tensor,
        inference_target,
        tensor_build_fn=torch.ones,
    )


@OP_REGISTRY.register()
def new_full(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:new_full' to NNEF.

    Layout: `(input, size, fill_value, dtype, layout, device,
    pin_memory)`. The fill value is folded at trace time into a
    custom build fn so the constant materialises with the right
    value.
    """
    fill_value = node.inputs[2].data

    def full_fn(*args, **fn_kwargs):
        return torch.ones(*args, **fn_kwargs) * fill_value

    return _new_x(
        node,
        g,
        torch_graph,
        name_to_tensor,
        inference_target,
        tensor_build_fn=full_fn,
        # dtype is at index 3 (one slot later than empty/ones).
        dtype_idx=3,
    )


@OP_REGISTRY.register()
def one_hot(g, node, name_to_tensor, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:one_hot' to NNEF.

    Two paths:

    - **TractNNEF**: emit `tract_core_one_hot(input, axis, dim)` and
      cast to int64. Tract has the op natively
      (`core/src/ops/array/one_hot.rs`) so this is the fastest path
      and produces the smallest graph.
    - **Pure NNEF**: emit the `one_hot` fragment which decomposes to
      `eq(unsqueeze(input, axis), classes) -> select` using stdlib
      ops. The classes constant is baked at trace time, pre-reshaped
      to `(1,) * input.rank + (num_classes,)` so NNEF `eq`'s
      exact-rank broadcast rule is satisfied.

    Torch's `one_hot(input, num_classes)` appends the one-hot dim as
    the trailing axis, so `axis = input.rank` (the new last position
    in the rank-(R+1) output) regardless of which path we take.
    """
    input_node, num_classes_node = node.inputs
    if not isinstance(num_classes_node.data, int):
        raise T2NErrorNotImplemented(
            "aten::one_hot with dynamic num_classes not yet supported"
        )
    num_classes = int(num_classes_node.data)
    axis = input_node.rank
    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)

    if isinstance(inference_target, TractNNEF):
        onehot_out = op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_one_hot",
            inputs=inp_ref,
            attrs={"axis": axis, "dim": num_classes},
            output_tensor_name_suffix="_oh",
        )
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            "tract_core_cast",
            inputs=onehot_out,
            attrs={"to": "i64"},
        )
        return ["tract_core"]

    # Pure-NNEF path: bake the classes constant pre-shaped to match
    # input.rank + 1 so the fragment's broadcast `eq` works without
    # extra unsqueezes on the classes side.
    classes_shape = (1,) * input_node.rank + (num_classes,)
    classes_const = PythonConstant(
        name=f"{node.outputs[0].name}_oh_classes",
        data=torch.arange(num_classes, dtype=input_node.dtype).reshape(
            classes_shape
        ),
    )
    classes_ref = get_or_add_tensor_variable_in_nnef(
        g, classes_const, name_to_tensor
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "one_hot",
        inputs=[inp_ref, classes_ref],
        attrs={"axis": axis},
        force_consistent_inputs_shapes=False,
    )
    return ["one_hot"]


@OP_REGISTRY.register()
def zero(**kwargs):
    """Map PyTorch: 'aten:zero' (and ``zero_``) to NNEF.

    `Tensor.zero_()` writes 0 into every position regardless of the
    original value (even NaN / +/-Inf). Reuse the `_x_like` machinery
    that already powers `zeros_like` so we materialise a true constant
    of zeros matching the input's shape and dtype: correct on every
    input, and shares the same dynamic-axes path (`tract_core_shape_of`
    + tile expansion) when shapes aren't known at trace time.

    Earlier this was implemented as `sub(x, x)`; that produced 0 for
    finite inputs but NaN for NaN inputs (since `NaN - NaN == NaN`),
    which silently diverged from `zero_`'s set-everything-to-0
    semantics.
    """
    return _x_like(tensor_build_fn=torch.zeros, **kwargs)


@OP_REGISTRY.register()
def empty_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use expansion

    """
    return _x_like(tensor_build_fn=torch.zeros, **kwargs)


@OP_REGISTRY.register()
def ones_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use expansion

    """
    return _x_like(tensor_build_fn=torch.ones, **kwargs)


@OP_REGISTRY.register()
def full_like(**kwargs):
    """Operator can not be exactly exported to NNEF if dynamic.

    With tract we use use expansion

    """
    fill_value = kwargs["node"].inputs.pop(1).data
    return _x_like(
        tensor_build_fn=lambda sh, dtype: torch.full(
            sh, fill_value, dtype=dtype
        ),
        **kwargs,
    )


@OP_REGISTRY.register()
def new_zeros(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:new_zeros' to NNEF."""
    (
        input_node,  # input_node,
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs

    # See note above: 0 maps to `torch.uint8` and is falsy in Python;
    # use is-None to avoid silently dropping uint8 dtype requests.
    if dtype_node.data is not None:
        dtype = SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
    else:
        dtype = input_node.dtype

    assert shape_node.data

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def zeros(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:zeros' to NNEF."""
    (
        shape_node,
        dtype_node,
        _,  # ? example PythonConstant(data=0, ...)
        _,  # device_node,
        _,  # requires_grad_node
    ) = node.inputs
    LOGGER.warning(
        "the aten::zeros replaced by constant traced values "
        "(follows NNEF spec)."
        "Keeping dynamism would require custom operator in tract internals."
    )
    dtype = (
        SCALAR_TYPE_TO_PYTORCH_TYPE[dtype_node.data]
        if dtype_node.data
        else torch.float32
    )
    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=dtype,
        tensor_build_fn=torch.zeros,
    )


@OP_REGISTRY.register()
def full(g, node, name_to_tensor, torch_graph, inference_target, **kwargs):
    """Map PyTorch: 'aten:full' to NNEF."""
    (shape_node, val_node, _, _, _, _) = node.inputs  # device_node,  # False

    def full_fn(*args, **kwargs):
        return torch.ones(*args, **kwargs) * val_node.data

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=torch.float32,
        tensor_build_fn=full_fn,
    )


@OP_REGISTRY.register(["fill", "fill_"])
def fill(
    g, node, name_to_tensor, torch_graph, inference_target, op_helper, **kwargs
):
    """Map PyTorch: 'aten:fill', 'aten:fill_' to NNEF."""
    (input_node, val_node, *_) = node.inputs  # device_node,  # False

    def full_fn(*args, **kwargs):
        return torch.ones(*args, **kwargs) * val_node.data

    if inference_target.has_dynamic_axes:
        dims_nnef = []
        for ix, _ in enumerate(input_node.shape[:]):
            soc = get_tract_dyn_axis_size_soc(op_helper, input_node, ix)
            dims_nnef.append(soc.output_name)
    else:
        dims_nnef = input_node.shape[:]
    shape_node = dims_nnef

    return _generic_auto_tensor_expansion(
        shape_node,
        node,
        g,
        torch_graph,
        name_to_tensor,
        has_dynamic_axes=inference_target.has_dynamic_axes,
        dtype=input_node.dtype,
        tensor_build_fn=full_fn,
    )


@OP_REGISTRY.register(torch_op_ids=["copy", "clone"])
def copy(
    g, node, name_to_tensor, inference_target, torch_graph, null_ref, **kwargs
):
    """Map PyTorch: 'aten:copy', 'aten:clone' to NNEF."""
    if not isinstance(inference_target, TractNNEF):
        # nnef spec include copy fragment
        return unary_output_op_without_attr(
            nnef_op_type="copy",
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
    return []


@OP_REGISTRY.register(
    torch_op_ids=[_.replace("aten::", "") for _ in MAP_TO_NOP]
)
def _post_graph_creation_remap(g, node, name_to_tensor, torch_graph, **kwargs):
    """Map PyTorch: no-ops to NNEF.

    List of no-ops:
        'aten:prim::NumToTensor',
        'aten:prim::ListConstruct',
        'aten:ScalarImplicit',
        'aten:alias'
    """
    torch_graph.remap_node(node.outputs[0], node.inputs[0])


def _trilu(g, name_to_tensor, node, inference_target, is_upper: bool = True):
    (input_node, diag_node) = node.inputs
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented("trilu need `tract_core_trilu`")

    if inference_target.version < "0.21.3":
        raise T2NErrorNotImplemented(
            "triu need `tract_core_trilu` from tract >= 0.21.4 "
            "(prior nnef deserialization was failing)"
        )

    # k = 0
    # upper =true
    if isinstance(diag_node, PythonConstant):
        k_diag = diag_node.data
    else:
        k_diag_tensor = get_or_add_tensor_variable_in_nnef(
            g, diag_node, name_to_tensor
        )
        k_diag = nnef.Identifier(k_diag_tensor.name)
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_trilu",
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
        ],
        attrs={"upper": is_upper, "k": k_diag},
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def triu(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:triu' to NNEF."""
    return _trilu(g, name_to_tensor, node, inference_target, is_upper=True)


@OP_REGISTRY.register()
def tril(
    g,
    node,
    name_to_tensor,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:tril' to NNEF."""
    return _trilu(g, name_to_tensor, node, inference_target, is_upper=False)


@OP_REGISTRY.register()
def eye(g, node, name_to_tensor, op_helper, **kwargs):
    """Map PyTorch: 'aten:eye' to NNEF via the `eye` fragment.

    Builds the identity at runtime from index ranges + broadcast-eq,
    so both static `n` (Python int from the trace) and dynamic `n`
    (a `TensorVariable` produced by e.g. `aten::size`) work without
    baking an `n*n` constant into the graph. This is critical for
    attention-mask construction in LLMs where `n = seq_len` is a
    dynamic axis.
    """
    n_inputs = len(node.inputs)
    if n_inputs == 5:
        # eye(n, dtype, layout, device, pin_memory)
        n_node = node.inputs[0]
        m_node = node.inputs[0]
    elif n_inputs == 6:
        # eye(n, m, dtype, layout, device, pin_memory)
        n_node = node.inputs[0]
        m_node = node.inputs[1]
    else:
        raise T2NErrorNotImplemented(
            f"aten::eye with {n_inputs} inputs (expected 5 or 6)"
        )

    onode = node.outputs[0]
    out_dtype = onode.dtype or torch.float32
    dtype_str = TORCH_DTYPE_TO_TRACT_STR[out_dtype]

    def _to_integer_param(d):
        """Resolve `n` / `m` for the eye fragment.

        Pass a literal int when known statically, an `Identifier`
        otherwise so a runtime-computed `n` flows through the fragment
        to `tract_core_range` and the identity is sized at runtime.
        """
        if isinstance(d, PythonConstant) and isinstance(d.data, int):
            return d.data
        # Register the runtime tensor in the NNEF graph so the
        # `Identifier` we hand back resolves to a real wire.
        op_helper.get_or_add_tensor_variable_in_nnef(d)
        return nnef.Identifier(d.export_name)

    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "eye",
        inputs=[],
        attrs={
            "n": _to_integer_param(n_node),
            "m": _to_integer_param(m_node),
            # Param renamed away from `dtype` since the NNEF writer
            # special-cases that attr key as a numpy-dtype lookup.
            "to": dtype_str,
        },
        force_consistent_inputs_shapes=False,
    )
    return ["eye"]


@OP_REGISTRY.register()
def vander(g, node, name_to_tensor, op_helper, **kwargs):
    """Map `aten::vander(self, N, increasing)` to NNEF.

    Vandermonde matrix: `out[i, j] = self[i]^exponent[j]` where
    `exponent = arange(N)` if `increasing` else `arange(N-1, -1, -1)`.
    `N` is the number of columns (defaults to `len(self)` if `None`).
    Bake `exponent` as a constant; the per-element pow is a runtime
    broadcast.
    """
    input_node, n_node, inc_node = node.inputs[:3]
    if input_node.rank != 1:
        raise T2NErrorNotImplemented(
            f"aten::vander: 1-D input required, got rank {input_node.rank}"
        )
    n_in = input_node.shape[0]
    if isinstance(n_node, PythonConstant) and isinstance(n_node.data, int):
        n_cols = int(n_node.data)
    elif (
        isinstance(n_node, PythonConstant)
        and n_node.data is None
        and isinstance(n_in, int)
    ):
        n_cols = n_in
    else:
        raise T2NErrorNotImplemented(
            "aten::vander: dynamic N (None with non-static input length) "
            "not supported"
        )
    increasing = bool(getattr(inc_node, "data", False))
    if increasing:
        exponents = torch.arange(n_cols, dtype=torch.float32)
    else:
        exponents = torch.arange(n_cols - 1, -1, -1, dtype=torch.float32)

    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    inp_2d = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=inp,
        attrs={"axes": [1]},
        output_tensor_name_suffix="_vander_unsq",
    )

    base_name = node.outputs[0].export_name
    exp_const = PythonConstant(name=f"{base_name}_vander_exp", data=exponents)
    exp_ref = op_helper.get_or_add_tensor_variable_in_nnef(exp_const)
    exp_2d = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=exp_ref,
        attrs={"axes": [0]},
        output_tensor_name_suffix="_vander_exp_unsq",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "pow",
        inputs=[inp_2d, exp_2d],
        force_consistent_inputs_shapes=False,
    )


@OP_REGISTRY.register()
def linspace(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:linspace' to a NNEF constant tensor.

    `aten::linspace(start, end, steps, dtype, layout, device,
    pin_memory)` is fully determined at trace time when `start`, `end`,
    `steps` are static, which is the common case. Bake the result via
    `torch.linspace` and register as a constant.
    """
    start_node, end_node, steps_node = node.inputs[:3]
    if not all(
        isinstance(n, PythonConstant) and isinstance(n.data, (int, float))
        for n in (start_node, end_node, steps_node)
    ):
        raise T2NErrorNotImplemented(
            "aten::linspace with dynamic start/end/steps not yet supported"
        )
    onode = node.outputs[0]
    out_dtype = onode.dtype or torch.float32
    onode.set_data(
        torch.linspace(
            start_node.data,
            end_node.data,
            int(steps_node.data),
            dtype=out_dtype,
        ),
        force_dtype=True,
        force_shape=True,
    )
    add_tensor_variable_node_as_nnef_tensor(g, onode, name_to_tensor)


def _emit_trilu_indices(g, node, name_to_tensor, torch_fn, op_name: str):
    """Shared body for aten::tril_indices / aten::triu_indices.

    Signature: `(row, col, offset, dtype, layout, device, pin_memory)`.
    Output is `(2, N)` Long where N is the number of (row, col) pairs
    in the (lower / upper) triangle of a `(row, col)` matrix at the
    given offset.
    """
    row_node, col_node, offset_node = node.inputs[:3]
    for n, name in ((row_node, "row"), (col_node, "col")):
        if not (isinstance(n, PythonConstant) and isinstance(n.data, int)):
            raise T2NErrorNotImplemented(
                f"aten::{op_name}: dynamic {name} not supported"
            )
    offset = 0
    if isinstance(offset_node, PythonConstant) and isinstance(
        offset_node.data, int
    ):
        offset = int(offset_node.data)
    onode = node.outputs[0]
    out_dtype = onode.dtype or torch.int64
    onode.set_data(
        torch_fn(
            int(row_node.data),
            int(col_node.data),
            offset=offset,
            dtype=out_dtype,
        ),
        force_dtype=True,
        force_shape=True,
    )
    add_tensor_variable_node_as_nnef_tensor(g, onode, name_to_tensor)


@OP_REGISTRY.register()
def tril_indices(g, node, name_to_tensor, **kwargs):
    """Map `aten::tril_indices(row, col, offset, ...)` to a NNEF constant."""
    _emit_trilu_indices(
        g, node, name_to_tensor, torch.tril_indices, "tril_indices"
    )


@OP_REGISTRY.register()
def triu_indices(g, node, name_to_tensor, **kwargs):
    """Map `aten::triu_indices(row, col, offset, ...)` to a NNEF constant."""
    _emit_trilu_indices(
        g, node, name_to_tensor, torch.triu_indices, "triu_indices"
    )


@OP_REGISTRY.register()
def logspace(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:logspace' to a NNEF constant tensor.

    `aten::logspace(start, end, steps, base, dtype, layout, device,
    pin_memory)`: `base ** linspace(start, end, steps)`. Bake at trace
    time when `start`, `end`, `steps`, and `base` are Python constants
    (the common case).
    """
    start_node, end_node, steps_node, base_node = node.inputs[:4]
    if not all(
        isinstance(n, PythonConstant) and isinstance(n.data, (int, float))
        for n in (start_node, end_node, steps_node, base_node)
    ):
        raise T2NErrorNotImplemented(
            "aten::logspace with dynamic start/end/steps/base not yet supported"
        )
    onode = node.outputs[0]
    out_dtype = onode.dtype or torch.float32
    onode.set_data(
        torch.logspace(
            start_node.data,
            end_node.data,
            int(steps_node.data),
            base=float(base_node.data),
            dtype=out_dtype,
        ),
        force_dtype=True,
        force_shape=True,
    )
    add_tensor_variable_node_as_nnef_tensor(g, onode, name_to_tensor)


def _emit_window_constant(g, node, name_to_tensor, torch_fn, op_name, **extra):
    """Shared helper for the `*_window` family.

    aten signature is `(window_length, periodic?, [op-specific args], dtype,
    layout, device, pin_memory)`. `window_length` and `periodic` are
    always positional 0 / 1; any op-specific args (e.g. `beta` for
    kaiser) are passed via `extra`. The result is computed at export
    time via the corresponding `torch` function and emitted as a
    constant tensor.
    """
    inputs = node.inputs
    length_node = inputs[0]
    if not (
        isinstance(length_node, PythonConstant)
        and isinstance(length_node.data, int)
    ):
        raise T2NErrorNotImplemented(
            f"aten::{op_name} with dynamic length not yet supported"
        )
    periodic = True
    if (
        len(inputs) >= 2
        and isinstance(inputs[1], PythonConstant)
        and isinstance(inputs[1].data, bool)
    ):
        periodic = inputs[1].data
    onode = node.outputs[0]
    out_dtype = onode.dtype or torch.float32
    onode.set_data(
        torch_fn(
            int(length_node.data), periodic=periodic, dtype=out_dtype, **extra
        ),
        force_dtype=True,
        force_shape=True,
    )
    add_tensor_variable_node_as_nnef_tensor(g, onode, name_to_tensor)


@OP_REGISTRY.register()
def hann_window(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:hann_window' to a NNEF constant tensor."""
    _emit_window_constant(
        g, node, name_to_tensor, torch.hann_window, "hann_window"
    )


@OP_REGISTRY.register()
def hamming_window(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:hamming_window' to a NNEF constant tensor."""
    _emit_window_constant(
        g, node, name_to_tensor, torch.hamming_window, "hamming_window"
    )


@OP_REGISTRY.register()
def blackman_window(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:blackman_window' to a NNEF constant tensor."""
    _emit_window_constant(
        g, node, name_to_tensor, torch.blackman_window, "blackman_window"
    )


@OP_REGISTRY.register()
def bartlett_window(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:bartlett_window' to a NNEF constant tensor."""
    _emit_window_constant(
        g, node, name_to_tensor, torch.bartlett_window, "bartlett_window"
    )


@OP_REGISTRY.register()
def kaiser_window(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:kaiser_window' to a NNEF constant tensor.

    Reads an optional `beta` (default `12.0`) at input index 2 when the
    full positional form is used.
    """
    beta = 12.0
    inputs = node.inputs
    if (
        len(inputs) >= 3
        and isinstance(inputs[2], PythonConstant)
        and isinstance(inputs[2].data, (int, float))
    ):
        beta = float(inputs[2].data)
    _emit_window_constant(
        g, node, name_to_tensor, torch.kaiser_window, "kaiser_window", beta=beta
    )


@OP_REGISTRY.register()
def affine_grid_generator(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
):
    """Map PyTorch: 'aten:affine_grid_generator' to NNEF.

    `affine_grid_generator(theta, size, align_corners)` builds a
    sampling grid for `F.grid_sample`. The base grid -- a fixed set
    of normalized coordinates in `[-1, 1]` (or pixel-centered for
    `align_corners=False`) -- depends only on the output spatial
    shape, so we bake it as a constant tensor at trace time. The
    only runtime cost is a single matmul against `theta`.

    Currently 2-D only (theta shape `(N, 2, 3)`, output `(N, H, W,
    2)`). 3-D affine_grid would need a `(D*H*W, 4)` base grid +
    reshape to `(N, D, H, W, 3)`.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    theta_node, size_node, align_corners_node = node.inputs
    size_data = size_node.data
    if hasattr(size_data, "tolist"):
        size_data = size_data.tolist()
    if len(size_data) != 4:
        raise T2NErrorNotImplemented(
            f"affine_grid_generator: only 2-D (rank-4 size) supported; "
            f"got size={size_data}"
        )
    # H / W must be static (we bake the base grid as a constant); the
    # batch dim N can be dynamic (resolved via `resolve_attr_axis_size`
    # below).

    def _as_static_int(v, name):
        if isinstance(v, int):
            return v
        if hasattr(v, "data") and isinstance(v.data, int):
            return v.data
        raise T2NErrorNotImplemented(
            f"affine_grid_generator: {name} must be statically known; got {v!r}"
        )

    h = _as_static_int(size_data[2], "H")
    w = _as_static_int(size_data[3], "W")
    align_corners = bool(align_corners_node.data)
    if theta_node.rank != 3 or tuple(theta_node.shape[1:]) != (2, 3):
        raise T2NErrorNotImplemented(
            f"affine_grid_generator: theta shape must be (N, 2, 3); "
            f"got {theta_node.shape}"
        )

    if align_corners:
        ys = torch.linspace(-1.0, 1.0, h)
        xs = torch.linspace(-1.0, 1.0, w)
    else:
        ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h)
        xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), torch.ones(h * w)],
        dim=1,
    )
    # Bake base as (1, 3, H*W) so it matches theta's rank-3 shape for
    # NNEF matmul (same-rank requirement).
    base_t = base.t().contiguous().unsqueeze(0).to(theta_node.dtype)
    onode = node.outputs[0]
    base_const = PythonConstant(
        name=f"{onode.name}_ag_base",
        data=base_t,
    )
    base_ref = op_helper.get_or_add_tensor_variable_in_nnef(base_const)
    theta_ref = op_helper.get_or_add_tensor_variable_in_nnef(theta_node)
    mm_out = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "matmul",
        inputs=[theta_ref, base_ref],
        attrs={"transposeA": False, "transposeB": False},
        output_tensor_name_suffix="ag_mm",
        force_consistent_inputs_shapes=False,
    )
    transposed = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "transpose",
        inputs=mm_out,
        attrs={"axes": [0, 2, 1]},
        output_tensor_name_suffix="ag_t",
    )
    # Resolve N from theta's runtime shape so the reshape works with a
    # dynamic batch axis (otherwise tract refuses to unify a concrete
    # N against the symbolic axis-0 dim).
    n_attr = resolve_attr_axis_size(op_helper, theta_node, axis=0)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=transposed,
        attrs={"shape": [n_attr, h, w, 2]},
    )
