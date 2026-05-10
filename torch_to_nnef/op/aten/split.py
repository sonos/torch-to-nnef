from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_multi_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    pick_axis,
)
from torch_to_nnef.torch_graph import PythonConstant

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register(
    torch_op_ids=["split_with_sizes", "unsafe_split_with_sizes"]
)
def split_with_sizes(g, node, name_to_tensor, **kwargs):
    """Translate `aten::split_with_sizes` to NNEF.

    NNEF spec has a `split` op (value, axis, ratios -> tensor[]) but tract
    does not register it, so we re-express each output as a `slice`.

    ``ratio_node`` may be a ``PythonConstant`` (literal sizes from the trace)
    or a ``TensorVariable`` whose data is shape-derived (e.g. fused-qkv
    splits like ``x.shape[-1] // 3``); both cases are unwrapped to plain ints.
    """
    (input_node, ratio_node, axis_node) = node.inputs
    assert isinstance(axis_node, PythonConstant)
    if ratio_node.data is None:
        raise T2NErrorNotImplemented(
            "split_with_sizes requires statically-known sizes"
        )
    ratio_data = ratio_node.data
    if hasattr(ratio_data, "tolist"):
        ratio_data = ratio_data.tolist()

    def _as_int(x):
        if hasattr(x, "data"):
            x = x.data
        if hasattr(x, "item"):
            x = x.item()
        return int(x)

    ratio_data = [_as_int(x) for x in ratio_data]
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node, n_elements in zip(node.outputs, ratio_data, strict=False):
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if n_elements <= 0:
            raise T2NErrorNotImplemented("unexpected n_elements<=0")
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [pick_axis(input_node, axis_node.data)],
                "begin": [current_dim_elm_idx],
                "end": [current_dim_elm_idx + n_elements],
                "stride": [1],
            },
        )
        if inputs.quant:
            out.quant = inputs.quant
        current_dim_elm_idx += n_elements


@OP_REGISTRY.register()
def unbind(g, node, name_to_tensor, **kwargs):
    """Unbind is `unstack` in NNEF."""
    input_node, axis_node = node.inputs
    add_multi_output_op(
        g,
        node,
        name_to_tensor,
        "unstack",
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axis": pick_axis(input_node, axis_node.data)},
        ensure_tuple=False,
    )


@OP_REGISTRY.register(torch_op_ids=["chunk", "unsafe_chunk"])
def chunk(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:chunk' (and `unsafe_chunk`) to NNEF.

    `unsafe_chunk` has identical inference-time semantics to `chunk`;
    the only difference is the autograd-graph promise around in-place
    writes, which doesn't apply on the export path.
    """
    (input_node, n_chunk_node, axis_node) = node.inputs
    assert n_chunk_node.data == len(node.outputs)
    assert len({tuple(o.shape) for o in node.outputs}) == 1, (
        "all chunk are not equal"
    )
    n_elements = node.outputs[0].shape[axis_node.data]
    current_dim_elm_idx = 0
    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node in node.outputs:
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [pick_axis(input_node, axis_node.data)],
                "begin": [current_dim_elm_idx],
                "end": [current_dim_elm_idx + n_elements],
                "stride": [1],
            },
        )
        current_dim_elm_idx += n_elements


def _tensor_split_compute_boundaries(dim_size, sections_or_indices):
    """Resolve `indices_or_sections` into a list of split-axis boundaries.

    - int N: split into N chunks. The first `dim_size % N` chunks have
      `ceil(dim_size / N)` elements; the rest have `floor(dim_size / N)`.
      Boundaries are `[k1, k1+k2, ..., k1+...+kN-1]` (length N-1).
    - int list: the values are direct boundary indices (length k);
      torch produces k+1 chunks separated by these.
    """
    if isinstance(sections_or_indices, int):
        n = sections_or_indices
        if n <= 0:
            raise T2NErrorNotImplemented(
                f"tensor_split requires positive sections, got {n}"
            )
        big = dim_size % n
        big_size = (dim_size + n - 1) // n
        small_size = dim_size // n
        sizes = [big_size] * big + [small_size] * (n - big)
        boundaries = []
        cur = 0
        for s in sizes[:-1]:
            cur += s
            boundaries.append(cur)
        return boundaries
    return [int(i) for i in sections_or_indices]


@OP_REGISTRY.register()
def tensor_split(g, node, name_to_tensor, **kwargs):
    """Map PyTorch: 'aten:tensor_split' to NNEF.

    Generalised split that allows uneven sections (unlike `split` /
    `chunk`). Two overloads are supported:

    * `tensor_split(self, sections: int, dim)` -- divide into N
      approximately-equal chunks; the first `dim_size % N` chunks
      take one extra element.
    * `tensor_split(self, indices: int[], dim)` -- split at the
      given boundary indices; produces `len(indices) + 1` chunks.

    Each output is a `slice` of the input along `dim`. Static-axis
    only: the boundaries depend on `dim_size`, which we resolve at
    trace time.
    """
    input_node, sections_node, axis_node = node.inputs
    axis = pick_axis(input_node, axis_node.data)
    dim_size = input_node.shape[axis]
    if not isinstance(dim_size, int):
        raise T2NErrorNotImplemented(
            f"tensor_split on dynamic axis {axis} not supported"
        )

    raw = sections_node.data
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list):
        sections_or_indices = [
            int(x.data) if isinstance(x, PythonConstant) else int(x)
            for x in raw
        ]
    else:
        sections_or_indices = int(raw)
    boundaries = _tensor_split_compute_boundaries(dim_size, sections_or_indices)
    bounds = [0, *boundaries, dim_size]
    assert len(bounds) - 1 == len(node.outputs), (
        f"tensor_split: expected {len(bounds) - 1} outputs, "
        f"got {len(node.outputs)}"
    )

    inputs = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    for out_node, begin, end in zip(
        node.outputs, bounds[:-1], bounds[1:], strict=True
    ):
        out = add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            prevent_variable=True,
        )
        cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="slice",
            inputs=inputs,
            outputs=tuple([out]),
            attribs={
                "axes": [axis],
                "begin": [begin],
                "end": [end],
                "stride": [1],
            },
        )
