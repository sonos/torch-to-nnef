"""Conservative per-tensor dynamic-axis tracking for symbolic-dim export.

Under ``dynamic_axes`` only some tensor axes are symbolic; the rest keep their
traced constant size. Without knowing which is which, t2n forces *every*
shape-derived value dynamic (see ``_if_dyn_shape_may_remove_resolved_dim``), so
a ``split`` whose sizes come from ``x.shape[-1]`` on a static axis fails to
lower. This pass over-approximates, per tensor, the set of axis indices that are
dynamic, so ``aten::size`` on a provably-static axis can still fold to a
constant.

Safety invariant: the returned set is an OVER-approximation of the truly-dynamic
axes. An axis is reported static (absent from the set) only when provably so, so
a genuinely symbolic dimension is never baked to a constant. Any op without a
precise rule marks all of its output axes dynamic.
"""

import typing as T
from math import prod

from torch_to_nnef.torch_graph.ir_data import TensorVariable
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_CONTIGUOUS_KIND,
    ATEN_LINEAR,
    ATEN_VIEW_KIND,
)

#: reshape-like kinds handled by the (conservative) view rule.
_VIEW_KINDS = {ATEN_VIEW_KIND, ATEN_CONTIGUOUS_KIND, "aten::reshape"}

#: split-like kinds: each output keeps the input axes except the (statically
#: sized) split axis.
_SPLIT_KINDS = {
    "aten::split_with_sizes",
    "aten::unsafe_split_with_sizes",
    "aten::split",
    "aten::chunk",
}


def _rank(node) -> T.Optional[int]:
    shape = getattr(node, "shape", None)
    return len(shape) if shape is not None else None


def _const_dim(node, rank: int) -> int:
    """Normalize a (possibly negative) constant dim argument to [0, rank)."""
    data = getattr(node, "data", None)
    dim = int(data) if isinstance(data, (int, float)) else 0
    return dim % rank if rank else dim


def _view_rule(src, out, dyn_of) -> T.Set[int]:
    """Dynamic axes of a reshape output, conservatively.

    Only the case where a leading prefix of axes is preserved 1:1 and the
    reshaped remainder is entirely static is resolved precisely; anything else
    falls back to "all dynamic".
    """
    if _rank(src) is None or _rank(out) is None:
        return set(range(_rank(out) or 0))
    si, so = list(src.shape), list(out.shape)
    k = 0
    while k < min(len(si), len(so)) and si[k] == so[k]:
        k += 1
    src_dyn = dyn_of(src)
    # a dynamic axis inside the reshaped remainder -> cannot track it -> all dyn
    if any(a >= k for a in src_dyn):
        return set(range(len(so)))
    if prod(si[k:]) != prod(so[k:]):
        return set(range(len(so)))
    # leading axes preserved (dyn inherited); reshaped remainder is static
    return {a for a in src_dyn if a < k}


def _broadcast_align(op, out, dyn_of, full) -> T.Set[int]:
    """Right-aligned broadcasting dynamic-axis rule (over-approximating).

    Applies only when the output rank is at least every tensor input's rank
    (elementwise / broadcast / keepdim reduction); a rank drop (e.g. non-keepdim
    reduction, squeeze) is not this shape and falls back to all-dynamic.
    """
    orank = _rank(out)
    if orank is None:
        return set()
    tensor_inputs = [
        i
        for i in op.inputs
        if isinstance(i, TensorVariable) and _rank(i) is not None
    ]
    if not tensor_inputs or any(_rank(i) > orank for i in tensor_inputs):
        return full(out)
    oshape = list(out.shape)
    result: T.Set[int] = set()
    for node in tensor_inputs:
        ish = list(node.shape)
        offset = orank - len(ish)
        node_dyn = dyn_of(node)
        for ia in node_dyn:
            oa = ia + offset
            # dynamic only if this axis is not broadcast (sizes match)
            if ish[ia] == oshape[oa]:
                result.add(oa)
    return result


def compute_dynamic_axis_map(
    ir_graph, dynamic_axes_by_name: T.Dict[str, T.Dict[int, str]]
) -> T.Dict[str, T.Set[int]]:
    """Map ``tensor.name -> set(dynamic axis indices)`` (over-approximation)."""
    dyn: T.Dict[str, T.Set[int]] = {}

    def full(node) -> T.Set[int]:
        rank = _rank(node)
        return set(range(rank)) if rank else set()

    def dyn_of(node) -> T.Set[int]:
        if not isinstance(node, TensorVariable):
            return set()
        if node.name in dyn:
            return dyn[node.name]
        if node.data is not None:  # constant / weight -> static
            return set()
        return full(node)  # unknown intermediate -> conservatively dynamic

    for inp in ir_graph.inputs:
        if not isinstance(inp, TensorVariable):
            continue
        rank = _rank(inp) or 0
        axes = dynamic_axes_by_name.get(inp.export_name) or {}
        dyn[inp.name] = {a % rank for a in axes} if rank else set()

    for op in _iter_in_dependency_order(ir_graph):
        _process_op(op, dyn, dyn_of, full)
    return dyn


def _iter_in_dependency_order(ir_graph):
    """Yield ops so every op comes after the ops producing its inputs.

    ``op_nodes`` is not guaranteed topological, and a rule reading an
    unresolved input would fall back to "all dynamic"; a cycle/unresolvable
    tail is yielded as-is (resolved conservatively).
    """
    producer = {
        o.name: op
        for op in ir_graph.op_nodes
        for o in op.outputs
        if isinstance(o, TensorVariable)
    }
    done: T.Set[int] = set()
    remaining = list(ir_graph.op_nodes)
    while remaining:
        pending = []
        progressed = False
        for op in remaining:
            deps = [
                producer[i.name]
                for i in op.inputs
                if isinstance(i, TensorVariable) and i.name in producer
            ]
            if all(id(d) in done for d in deps):
                yield op
                done.add(id(op))
                progressed = True
            else:
                pending.append(op)
        if not progressed:  # cycle / unresolvable -> yield rest as-is
            yield from pending
            return
        remaining = pending


def _process_op(op, dyn, dyn_of, full) -> None:
    """Set ``dyn[out]`` for each tensor output of ``op``."""
    outs = [o for o in op.outputs if isinstance(o, TensorVariable)]
    if not outs:
        return
    if op.kind in _SPLIT_KINDS:
        src = op.inputs[0]
        dim = _const_dim(op.inputs[-1], (_rank(src) or 1))
        src_dyn = {a for a in dyn_of(src) if a != dim}
        for out in outs:
            dyn[out.name] = set(src_dyn)
    elif len(outs) > 1:  # other multi-output -> conservative
        for out in outs:
            dyn[out.name] = full(out)
    else:
        dyn[outs[0].name] = _output_dyn(op, outs[0], dyn_of, full)


def _output_dyn(op, out, dyn_of, full) -> T.Set[int]:
    """Dynamic-axis set of an op's single output (over-approximating)."""
    kind = op.kind
    if kind == ATEN_LINEAR:
        # leading dims preserved; last axis -> out_features (static)
        rank = _rank(out) or 0
        return {a for a in dyn_of(op.inputs[0]) if a < rank - 1}
    if kind in _VIEW_KINDS:
        return _view_rule(op.inputs[0], out, dyn_of)
    if kind == "aten::unsqueeze":
        dim = _const_dim(op.inputs[1], (_rank(out) or 1))
        return {a + 1 if a >= dim else a for a in dyn_of(op.inputs[0])}
    if kind == "aten::select":
        src = op.inputs[0]
        dim = _const_dim(op.inputs[1], (_rank(src) or 1))
        return {a - 1 if a > dim else a for a in dyn_of(src) if a != dim}
    if kind == "aten::index_select":
        src, dim_node, index = op.inputs[0], op.inputs[1], op.inputs[2]
        dim = _const_dim(dim_node, (_rank(src) or 1))
        res = {a for a in dyn_of(src) if a != dim}
        if dyn_of(index):  # gathered axis follows the (dynamic) index
            res.add(dim)
        return res
    # generic elementwise / broadcast / keepdim-reduction: an output axis is
    # dynamic iff some tensor input has a dynamic, size-matching axis aligned to
    # it (right-aligned, NumPy broadcasting). Covers add/mul/norm/activation and
    # keepdim reductions (RMSNorm etc.).
    return _broadcast_align(op, out, dyn_of, full)


def size_query_is_dynamic(
    dynamic_axis_map: T.Dict[str, T.Set[int]], input_node, axis: int
) -> bool:
    """Whether ``aten::size(input_node, axis)`` reads a dynamic axis.

    Conservative: if the tensor is unknown to the map, treat as dynamic.
    """
    rank = _rank(input_node)
    if rank:
        axis %= rank
    if not isinstance(input_node, TensorVariable):
        return True
    if input_node.name not in dynamic_axis_map:
        return input_node.data is None  # constants are static
    return axis in dynamic_axis_map[input_node.name]
