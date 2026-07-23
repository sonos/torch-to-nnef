"""Conservative per-tensor dynamic-axis tracking for symbolic-dim export.

Under ``dynamic_axes`` only some tensor axes are symbolic; the rest keep their
traced constant size. Without knowing which is which, t2n forces *every*
shape-derived value dynamic (see ``_if_dyn_shape_may_remove_resolved_dim``), so
a ``split`` whose sizes come from ``x.shape[-1]`` on a static axis fails to
lower. This pass over-approximates, per tensor, the set of axis indices that are
dynamic, so ``aten::size`` on a provably-static axis can still fold.

Rules are purely STRUCTURAL: dynamic-ness is decided from op structure (reshape
target literals, axis arguments, right-aligned broadcasting), never by comparing
traced shape *values*. That matters because a dynamic axis is often traced as
size 1 (e.g. batch), which a value comparison cannot tell apart from a static 1.

Safety invariant: the returned set OVER-approximates the truly-dynamic axes. An
axis is reported static (absent) only when provably so, so a symbolic dim is
never baked to a constant. Any op without a precise rule marks all its outputs
dynamic.
"""

import typing as T

from torch_to_nnef.torch_graph.ir_data import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.torch_graph.torch_const import ATEN_LINEAR, ATEN_VIEW_KIND

#: reshape-like kinds whose output axes follow the target-shape spec.
_VIEW_KINDS = {ATEN_VIEW_KIND, "aten::reshape"}

#: split-like kinds: each output keeps the input axes except the split axis
#: (when the split sizes are compile-time constants).
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


def _is_static_literal(node) -> bool:
    """A non-negative integer literal reshape dim -> a fixed, static size."""
    return (
        isinstance(node, PythonConstant)
        and isinstance(node.data, int)
        and node.data >= 0
    )


def _all_constant(node) -> bool:
    if isinstance(node, PythonConstant):
        return True
    if isinstance(node, FixedTensorList):
        return all(isinstance(e, PythonConstant) for e in node.data)
    return False


def _view_rule(op, out, full) -> T.Set[int]:
    """Reshape output dynamic axes from the target-shape spec (structural).

    An output axis is static iff its target-shape element is a non-negative
    integer literal; a ``-1`` (inferred) or a size-derived element is treated
    as dynamic (conservative -- ``-1`` may absorb a symbolic dim).
    """
    target = op.inputs[1] if len(op.inputs) > 1 else None
    orank = _rank(out)
    if not isinstance(target, FixedTensorList) or orank is None:
        return full(out)
    elems = target.data
    if len(elems) != orank:  # e.g. complex trailing axis appended downstream
        return full(out)
    return {i for i, e in enumerate(elems) if not _is_static_literal(e)}


def _broadcast_align(op, out, dyn_of, full) -> T.Set[int]:
    """Right-aligned broadcasting rule (structural, over-approximating).

    An output axis is dynamic if any tensor input has a dynamic axis aligned to
    it from the right. No value comparison, so a broadcast size-1 axis is simply
    over-approximated. Applies only when the output rank is at least every
    input's rank (elementwise / broadcast / keepdim reduction); a rank drop
    falls back to all-dynamic.
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
    result: T.Set[int] = set()
    for node in tensor_inputs:
        offset = orank - _rank(node)
        for ia in dyn_of(node):
            result.add(ia + offset)
    return result


def _output_dyn(op, out, dyn_of, full) -> T.Set[int]:
    """Dynamic-axis set of an op's single output (over-approximating)."""
    kind = op.kind
    if kind == ATEN_LINEAR:
        # leading dims preserved; last axis -> out_features (static)
        rank = _rank(out) or 0
        return {a for a in dyn_of(op.inputs[0]) if a < rank - 1}
    if kind in _VIEW_KINDS:
        return _view_rule(op, out, full)
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
    # generic elementwise / broadcast / keepdim-reduction
    return _broadcast_align(op, out, dyn_of, full)


def _process_op(op, dyn, dyn_of, full) -> None:
    """Set ``dyn[out]`` for each tensor output of ``op``."""
    outs = [o for o in op.outputs if isinstance(o, TensorVariable)]
    if not outs:
        return
    if op.kind in _SPLIT_KINDS:
        src = op.inputs[0]
        dim = _const_dim(op.inputs[-1], (_rank(src) or 1))
        src_dyn = dyn_of(src)
        # the split axis is static only if the split sizes are constant
        if len(op.inputs) > 1 and _all_constant(op.inputs[1]):
            src_dyn = {a for a in src_dyn if a != dim}
        for out in outs:
            dyn[out.name] = set(src_dyn)
    elif len(outs) > 1:  # other multi-output -> conservative
        for out in outs:
            dyn[out.name] = full(out)
    else:
        dyn[outs[0].name] = _output_dyn(op, outs[0], dyn_of, full)


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
