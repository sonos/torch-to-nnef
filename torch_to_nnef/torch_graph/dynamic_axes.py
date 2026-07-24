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
from torch_to_nnef.torch_graph.torch_const import (
    ATEN__LOG_SOFTMAX,
    ATEN__SOFTMAX,
    ATEN_ABS,
    ATEN_ADD,
    ATEN_ADD_,
    ATEN_ALIAS,
    ATEN_AMAX,
    ATEN_AMIN,
    ATEN_BMM,
    ATEN_CAT,
    ATEN_CEIL,
    ATEN_CHUNK,
    ATEN_CLAMP,
    ATEN_CLAMP_MAX,
    ATEN_CLAMP_MIN,
    ATEN_CLONE,
    ATEN_CONTIGUOUS_KIND,
    ATEN_COS,
    ATEN_DETACH,
    ATEN_DIV,
    ATEN_DIV_,
    ATEN_DROPOUT,
    ATEN_ELU,
    ATEN_EQ,
    ATEN_ERF,
    ATEN_EXP,
    ATEN_FLATTEN,
    ATEN_FLOOR,
    ATEN_FMOD,
    ATEN_GE,
    ATEN_GELU,
    ATEN_GT,
    ATEN_HARDSIGMOID,
    ATEN_HARDSWISH,
    ATEN_HARDTANH,
    ATEN_INDEX_SELECT,
    ATEN_LE,
    ATEN_LEAKY_RELU,
    ATEN_LINEAR,
    ATEN_LOG,
    ATEN_LOG2,
    ATEN_LOG_SOFTMAX,
    ATEN_LOGICAL_AND,
    ATEN_LOGICAL_NOT,
    ATEN_LOGICAL_OR,
    ATEN_LT,
    ATEN_MASKED_FILL,
    ATEN_MASKED_FILL_,
    ATEN_MATMUL,
    ATEN_MAXIMUM,
    ATEN_MEAN,
    ATEN_MINIMUM,
    ATEN_MISH,
    ATEN_MUL,
    ATEN_MUL_,
    ATEN_NE,
    ATEN_NEG,
    ATEN_PERMUTE,
    ATEN_POW,
    ATEN_POW_,
    ATEN_RECIPROCAL,
    ATEN_RELU,
    ATEN_RELU6,
    ATEN_RESHAPE,
    ATEN_ROUND,
    ATEN_RSQRT,
    ATEN_RSUB,
    ATEN_SELECT,
    ATEN_SIGMOID,
    ATEN_SIGN,
    ATEN_SILU,
    ATEN_SIN,
    ATEN_SOFTMAX,
    ATEN_SOFTPLUS,
    ATEN_SPLIT,
    ATEN_SPLIT_WITH_SIZES,
    ATEN_SQRT,
    ATEN_SQUEEZE,
    ATEN_STACK,
    ATEN_STD,
    ATEN_SUB,
    ATEN_SUB_,
    ATEN_SUM,
    ATEN_TANH,
    ATEN_TO,
    ATEN_TO_COPY,
    ATEN_TRANSPOSE,
    ATEN_TYPE_AS,
    ATEN_UNSAFE_SPLIT_WITH_SIZES,
    ATEN_UNSQUEEZE,
    ATEN_VAR,
    ATEN_VIEW_KIND,
    ATEN_WHERE,
)

#: shape + axis-identity preserving (output copies input[0]'s dynamic axes).
_IDENTITY_KINDS = {
    ATEN_CONTIGUOUS_KIND,
    ATEN_CLONE,
    ATEN_DETACH,
    ATEN_ALIAS,
    ATEN_TO,
    ATEN_TYPE_AS,
    ATEN_TO_COPY,
    ATEN_DROPOUT,
}

#: reshape-like kinds resolved from the target-shape spec.
_VIEW_KINDS = {ATEN_VIEW_KIND, ATEN_RESHAPE}

#: matmul-like kinds (contract last of a with second-last of b).
_MATMUL_KINDS = {ATEN_MATMUL, ATEN_BMM}

#: split-like kinds: each output keeps the input axes except the split axis
#: (when the split sizes are compile-time constants).
_SPLIT_KINDS = {
    ATEN_SPLIT_WITH_SIZES,
    ATEN_UNSAFE_SPLIT_WITH_SIZES,
    ATEN_SPLIT,
    ATEN_CHUNK,
}

#: elementwise / broadcast / keepdim-reduction ops: output axes correspond to
#: inputs by right-aligned (NumPy) broadcasting. ONLY genuine axis-identity ops
#: belong here (never axis reorderers), so the broadcast rule stays sound.
_ELEMENTWISE_KINDS = {
    # arithmetic
    ATEN_ADD,
    ATEN_ADD_,
    ATEN_SUB,
    ATEN_SUB_,
    ATEN_RSUB,
    ATEN_MUL,
    ATEN_MUL_,
    ATEN_DIV,
    ATEN_DIV_,
    ATEN_POW,
    ATEN_POW_,
    ATEN_MAXIMUM,
    ATEN_MINIMUM,
    ATEN_FMOD,
    # unary math
    ATEN_NEG,
    ATEN_ABS,
    ATEN_EXP,
    ATEN_LOG,
    ATEN_LOG2,
    ATEN_SQRT,
    ATEN_RSQRT,
    ATEN_RECIPROCAL,
    ATEN_COS,
    ATEN_SIN,
    ATEN_ERF,
    ATEN_SIGN,
    ATEN_FLOOR,
    ATEN_CEIL,
    ATEN_ROUND,
    # activations
    ATEN_SIGMOID,
    ATEN_TANH,
    ATEN_GELU,
    ATEN_RELU,
    ATEN_RELU6,
    ATEN_SILU,
    ATEN_MISH,
    ATEN_ELU,
    ATEN_LEAKY_RELU,
    ATEN_SOFTPLUS,
    ATEN_HARDTANH,
    ATEN_HARDSWISH,
    ATEN_HARDSIGMOID,
    ATEN_SOFTMAX,
    ATEN__SOFTMAX,
    ATEN_LOG_SOFTMAX,
    ATEN__LOG_SOFTMAX,
    # clamp
    ATEN_CLAMP,
    ATEN_CLAMP_MIN,
    ATEN_CLAMP_MAX,
    # keepdim reductions (rank preserved); no-keepdim reduces rank -> full
    ATEN_MEAN,
    ATEN_SUM,
    ATEN_AMAX,
    ATEN_AMIN,
    ATEN_VAR,
    ATEN_STD,
    # comparisons / logical / masking (same-shape)
    ATEN_GT,
    ATEN_LT,
    ATEN_GE,
    ATEN_LE,
    ATEN_EQ,
    ATEN_NE,
    ATEN_LOGICAL_AND,
    ATEN_LOGICAL_OR,
    ATEN_LOGICAL_NOT,
    ATEN_MASKED_FILL,
    ATEN_MASKED_FILL_,
    ATEN_WHERE,
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
    ATEN_RESHAPE: _view_rule,
    ATEN_TRANSPOSE: _rule_transpose,
    ATEN_PERMUTE: _rule_permute,
    ATEN_UNSQUEEZE: _rule_unsqueeze,
    ATEN_SELECT: _rule_select,
    ATEN_SQUEEZE: _rule_squeeze,
    ATEN_FLATTEN: _rule_flatten,
    ATEN_INDEX_SELECT: _rule_index_select,
    ATEN_CAT: _rule_cat,
    ATEN_STACK: _rule_stack,
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
