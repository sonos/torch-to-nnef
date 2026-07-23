"""Conservative per-tensor dynamic-axis tracking for symbolic-dim export.

Under ``dynamic_axes`` only some tensor axes are symbolic; the rest keep their
traced constant size. Without knowing which is which, t2n forces *every*
shape-derived value dynamic (see ``_if_dyn_shape_may_remove_resolved_dim``), so
a ``split`` whose sizes come from ``x.shape[-1]`` on a static axis fails to
lower. This pass computes, per tensor, the set of axis indices that are dynamic,
so ``aten::size`` on a provably-static axis can still fold.

Design (why it is sound):

- The returned set OVER-approximates the truly-dynamic axes. An axis is reported
  static (absent) only when provably so, so a genuinely symbolic dimension is
  never baked to a constant.
- The DEFAULT for any op without an explicit rule is "all axes dynamic". In
  particular the elementwise/broadcasting rule (which assumes right-aligned axis
  identity) is applied ONLY to a whitelist of genuine elementwise ops; an
  axis-reordering op (transpose, permute, ...) that is not explicitly handled
  therefore falls back to all-dynamic rather than being mis-mapped.
- Rules are STRUCTURAL: dynamic-ness is decided from op structure (reshape
  target literals, axis arguments, right-aligned broadcasting), never by
  comparing traced shape *values* -- a dynamic axis is often traced as size 1
  (e.g. batch), indistinguishable from a static 1 by value.
"""

import typing as T

from torch_to_nnef.torch_graph.ir_data import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.torch_graph.torch_const import ATEN_LINEAR, ATEN_VIEW_KIND

#: shape + axis-identity preserving (output copies input[0]'s dynamic axes).
_IDENTITY_KINDS = {
    "aten::contiguous",
    "aten::clone",
    "aten::detach",
    "aten::alias",
    "aten::to",
    "aten::type_as",
    "aten::_to_copy",
    "aten::dropout",
}

#: reshape-like kinds resolved from the target-shape spec.
_VIEW_KINDS = {ATEN_VIEW_KIND, "aten::reshape"}

#: matmul-like kinds (contract last of a with second-last of b).
_MATMUL_KINDS = {"aten::matmul", "aten::bmm"}

#: split-like kinds: each output keeps the input axes except the split axis
#: (when the split sizes are compile-time constants).
_SPLIT_KINDS = {
    "aten::split_with_sizes",
    "aten::unsafe_split_with_sizes",
    "aten::split",
    "aten::chunk",
}

#: elementwise / broadcast / keepdim-reduction ops: output axes correspond to
#: inputs by right-aligned (NumPy) broadcasting. ONLY genuine axis-identity ops
#: belong here (never axis reorderers), so the broadcast rule stays sound.
_ELEMENTWISE_KINDS = {
    # arithmetic
    "aten::add",
    "aten::add_",
    "aten::sub",
    "aten::sub_",
    "aten::rsub",
    "aten::mul",
    "aten::mul_",
    "aten::div",
    "aten::div_",
    "aten::pow",
    "aten::pow_",
    "aten::maximum",
    "aten::minimum",
    "aten::fmod",
    # unary math
    "aten::neg",
    "aten::abs",
    "aten::exp",
    "aten::log",
    "aten::log2",
    "aten::sqrt",
    "aten::rsqrt",
    "aten::reciprocal",
    "aten::cos",
    "aten::sin",
    "aten::erf",
    "aten::sign",
    "aten::floor",
    "aten::ceil",
    "aten::round",
    # activations
    "aten::sigmoid",
    "aten::tanh",
    "aten::gelu",
    "aten::relu",
    "aten::relu6",
    "aten::silu",
    "aten::mish",
    "aten::elu",
    "aten::leaky_relu",
    "aten::softplus",
    "aten::hardtanh",
    "aten::hardswish",
    "aten::hardsigmoid",
    "aten::softmax",
    "aten::_softmax",
    "aten::log_softmax",
    "aten::_log_softmax",
    # clamp
    "aten::clamp",
    "aten::clamp_min",
    "aten::clamp_max",
    # keepdim reductions (rank preserved); no-keepdim reduces rank -> full
    "aten::mean",
    "aten::sum",
    "aten::amax",
    "aten::amin",
    "aten::var",
    "aten::std",
    # comparisons / logical / masking (same-shape)
    "aten::gt",
    "aten::lt",
    "aten::ge",
    "aten::le",
    "aten::eq",
    "aten::ne",
    "aten::logical_and",
    "aten::logical_or",
    "aten::logical_not",
    "aten::masked_fill",
    "aten::masked_fill_",
    "aten::where",
}


def _rank(node) -> T.Optional[int]:
    shape = getattr(node, "shape", None)
    return len(shape) if shape is not None else None


def _const_dim(node, rank: int) -> int:
    """Normalize a (possibly negative) constant dim argument to [0, rank)."""
    data = getattr(node, "data", None)
    dim = int(data) if isinstance(data, (int, float)) else 0
    return dim % rank if rank else dim


def _const_int_list(node) -> T.Optional[T.List[int]]:
    """Extract a constant list of ints (e.g. a permute dims arg), or None."""
    if (
        isinstance(node, PythonConstant)
        and isinstance(node.data, (list, tuple))
        and all(isinstance(v, int) for v in node.data)
    ):
        return list(node.data)
    if isinstance(node, FixedTensorList):
        vals = []
        for e in node.data:
            if isinstance(e, PythonConstant) and isinstance(e.data, int):
                vals.append(e.data)
            else:
                return None
        return vals
    return None


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


def _view_rule(op, out, dyn_of, full) -> T.Set[int]:
    """Reshape output dynamic axes from the target-shape spec (structural).

    An output axis is static iff its target-shape element is a non-negative
    integer literal; a ``-1`` (inferred) or a size-derived element is treated as
    dynamic (conservative -- ``-1`` may absorb a symbolic dim).
    """
    target = op.inputs[1] if len(op.inputs) > 1 else None
    orank = _rank(out)
    if not isinstance(target, FixedTensorList) or orank is None:
        return full(out)
    elems = target.data
    if len(elems) != orank:
        return full(out)
    return {i for i, e in enumerate(elems) if not _is_static_literal(e)}


def _matmul_rule(op, out, dyn_of, full) -> T.Set[int]:
    """matmul/bmm: contract a[-1] with b[-2]; batch dims broadcast."""
    a, b = op.inputs[0], op.inputs[1]
    arank, brank, orank = _rank(a), _rank(b), _rank(out)
    if None in (arank, brank, orank) or arank < 2 or brank < 2:
        return full(out)
    res: T.Set[int] = set()
    for ia in dyn_of(a):
        if ia == arank - 1:  # contracted
            continue
        if ia == arank - 2:
            res.add(orank - 2)
        else:  # batch dim (right-aligned)
            res.add(ia + (orank - arank))
    for ib in dyn_of(b):
        if ib == brank - 2:  # contracted
            continue
        if ib == brank - 1:
            res.add(orank - 1)
        else:  # batch dim (right-aligned)
            res.add(ib + (orank - brank))
    return res


def _broadcast_align(op, out, dyn_of, full) -> T.Set[int]:
    """Right-aligned broadcasting rule (structural, over-approximating).

    An output axis is dynamic if any tensor input has a dynamic axis aligned to
    it from the right. Applies only when the output rank is at least every
    input's rank (elementwise / broadcast / keepdim reduction); a rank drop
    (e.g. no-keepdim reduction) falls back to all-dynamic.
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


def _rule_identity(op, out, dyn_of, full) -> T.Set[int]:
    return set(dyn_of(op.inputs[0]))


def _rule_linear(op, out, dyn_of, full) -> T.Set[int]:
    # leading dims preserved; last axis -> out_features (static)
    rank = _rank(out) or 0
    return {a for a in dyn_of(op.inputs[0]) if a < rank - 1}


def _rule_transpose(op, out, dyn_of, full) -> T.Set[int]:
    src = op.inputs[0]
    rank = _rank(src) or 1
    d0, d1 = _const_dim(op.inputs[1], rank), _const_dim(op.inputs[2], rank)
    return {d1 if a == d0 else d0 if a == d1 else a for a in dyn_of(src)}


def _rule_permute(op, out, dyn_of, full) -> T.Set[int]:
    src = op.inputs[0]
    dims = _const_int_list(op.inputs[1])
    rank = _rank(src) or 1
    if dims is None:
        return full(out)
    src_dyn = dyn_of(src)
    return {i for i, d in enumerate(dims) if d % rank in src_dyn}


def _rule_unsqueeze(op, out, dyn_of, full) -> T.Set[int]:
    dim = _const_dim(op.inputs[1], (_rank(out) or 1))
    return {a + 1 if a >= dim else a for a in dyn_of(op.inputs[0])}


def _rule_select(op, out, dyn_of, full) -> T.Set[int]:
    src = op.inputs[0]
    dim = _const_dim(op.inputs[1], (_rank(src) or 1))
    return {a - 1 if a > dim else a for a in dyn_of(src) if a != dim}


def _rule_squeeze(op, out, dyn_of, full) -> T.Set[int]:
    src = op.inputs[0]
    if len(op.inputs) < 2:  # squeeze() over all size-1 axes -> unmapped
        return full(out)
    dim = _const_dim(op.inputs[1], (_rank(src) or 1))
    return {a - 1 if a > dim else a for a in dyn_of(src) if a != dim}


def _rule_index_select(op, out, dyn_of, full) -> T.Set[int]:
    src, dim_node, index = op.inputs[0], op.inputs[1], op.inputs[2]
    if _rank(index) != 1:  # ND index reshapes output -> unmapped
        return full(out)
    dim = _const_dim(dim_node, (_rank(src) or 1))
    res = {a for a in dyn_of(src) if a != dim}
    if dyn_of(index):  # gathered axis follows the (dynamic) index
        res.add(dim)
    return res


def _rule_cat(op, out, dyn_of, full) -> T.Set[int]:
    tensors = op.inputs[0]
    if not isinstance(tensors, FixedTensorList):
        return full(out)
    res: T.Set[int] = set()
    for t in tensors.data:  # same rank; per-axis union (cat dim included)
        res |= dyn_of(t)
    return res


def _rule_stack(op, out, dyn_of, full) -> T.Set[int]:
    tensors = op.inputs[0]
    if not isinstance(tensors, FixedTensorList):
        return full(out)
    orank = _rank(out) or 1
    dim = _const_dim(op.inputs[1], orank) if len(op.inputs) > 1 else 0
    # a new axis of size len(tensors) is inserted at `dim` (static); each
    # input's dynamic axes shift past the inserted axis.
    res: T.Set[int] = set()
    for tnode in tensors.data:
        for a in dyn_of(tnode):
            res.add(a + 1 if a >= dim else a)
    return res


def _rule_flatten(op, out, dyn_of, full) -> T.Set[int]:
    src = op.inputs[0]
    rank = _rank(src)
    if rank is None or len(op.inputs) < 3:
        return full(out)
    start = _const_dim(op.inputs[1], rank)
    end = _const_dim(op.inputs[2], rank)
    if end < start:
        return full(out)
    res = set()
    for a in dyn_of(src):
        if a < start:
            res.add(a)
        elif a <= end:
            res.add(start)  # merged axis
        else:
            res.add(a - (end - start))
    return res


#: exact-kind dispatch (uniform signature ``(op, out, dyn_of, full)``).
_KIND_RULES = {
    ATEN_LINEAR: _rule_linear,
    ATEN_VIEW_KIND: _view_rule,
    "aten::reshape": _view_rule,
    "aten::transpose": _rule_transpose,
    "aten::permute": _rule_permute,
    "aten::unsqueeze": _rule_unsqueeze,
    "aten::select": _rule_select,
    "aten::squeeze": _rule_squeeze,
    "aten::flatten": _rule_flatten,
    "aten::index_select": _rule_index_select,
    "aten::cat": _rule_cat,
    "aten::stack": _rule_stack,
}


def _output_dyn(op, out, dyn_of, full) -> T.Set[int]:
    """Dynamic-axis set of an op's single output (over-approximating)."""
    kind = op.kind
    if kind in _IDENTITY_KINDS:
        return _rule_identity(op, out, dyn_of, full)
    if kind in _MATMUL_KINDS:
        return _matmul_rule(op, out, dyn_of, full)
    if kind in _ELEMENTWISE_KINDS:
        return _broadcast_align(op, out, dyn_of, full)
    rule = _KIND_RULES.get(kind)
    if rule is not None:
        return rule(op, out, dyn_of, full)
    return full(out)  # conservative default: unknown op -> all axes dynamic


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
