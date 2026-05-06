"""Reusable JIT-graph passes for hardening JIT-only models against t2n parsing.

These helpers are useful when the source model arrives as a `torch.jit.JIT`
artifact (e.g. Silero-VAD's `silero_vad.jit`) whose Python source isn't on
the import path. After `torch._C._jit_pass_inline` flattens the graph,
PyTorch's compiled-in dim/shape assertions (notably inside `nn.LSTMCell`
and STFT helpers) leave behind `prim::If` nodes whose only effect on one
branch is to raise an exception. Those branches feed scalar-typed
arithmetic that t2n's tensor-oriented parser cannot represent, so we drop
them.
"""

from __future__ import annotations

import typing as T

import torch

ASSERTION_BLOCK_KINDS: T.Tuple[str, ...] = (
    "prim::Constant",
    "prim::ListConstruct",
    "prim::TupleConstruct",
    "prim::RaiseException",
    "aten::format",
    "aten::__getitem__",
    "aten::__contains__",
    "aten::__not__",
    "aten::dim",
    "aten::eq",
    "aten::ne",
    "aten::Int",
    "aten::str",
    "aten::add",
    "aten::mul",
    "prim::dtype",
    "prim::device",
)


def _walk_nodes(graph_or_block):
    """Yield nodes recursively, descending into prim::If / prim::Loop blocks."""
    for node in graph_or_block.nodes():
        yield node
        for blk in node.blocks():
            yield from _walk_nodes(blk)


def _block_only_raises(block) -> bool:
    """Return True iff the block is purely an assertion / RaiseException.

    A no-op-with-exception block consists only of side-effect-free constant /
    ad-hoc ops preparing the exception, ending in a RaiseException.

    Intentionally conservative: any unrecognized op disqualifies the block
    so we never strip a branch that may have observable side effects on
    the rest of the graph.
    """
    nodes = list(block.nodes())
    if not nodes:
        return False
    if nodes[-1].kind() != "prim::RaiseException":
        return False
    return all(n.kind() in ASSERTION_BLOCK_KINDS for n in nodes)


def _tensor_sizes(value) -> T.Optional[T.List[int]]:
    """Return the concrete `sizes()` for a Tensor SSA value.

    Returns None if any dim is symbolic / unknown.
    """
    ty = value.type()
    if not hasattr(ty, "sizes"):
        return None
    try:
        sizes = ty.sizes()
    except RuntimeError:
        return None
    if sizes is None or any(s is None for s in sizes):
        return None
    return list(sizes)


def _insert_int_constant_before(
    graph: "torch._C.Graph",
    value: int,
    before_node: "torch._C.Node",
) -> "torch._C.Value":
    """Insert a `prim::Constant` int just before `before_node`.

    Uses `Graph.create` + `Node.insertBefore` rather than the higher-level
    `Graph.insertConstant`, which interacts badly with `setInsertPoint`
    when the target lives in a sub-block. Returns the SSA value.
    """
    n = graph.create("prim::Constant")
    n.output().setType(torch._C.IntType.get())
    n.i_("value", int(value))
    n.insertBefore(before_node)
    return n.output()


def _insert_int_list_constant(
    graph: "torch._C.Graph",
    values: T.Sequence[int],
    before_node: "torch._C.Node",
) -> "torch._C.Value":
    """Materialize a `prim::ListConstruct` of int constants.

    Inserts the elements and the list construct just before `before_node`.
    Returns the list SSA value.
    """
    elem_vals = [
        _insert_int_constant_before(graph, v, before_node) for v in values
    ]
    lc = graph.create("prim::ListConstruct", 1)
    lc.output().setType(torch._C.ListType(torch._C.IntType.get()))
    for v in elem_vals:
        lc.addInput(v)
    lc.insertBefore(before_node)
    return lc.output()


_PRIMITIVE_TYPE_KINDS: T.FrozenSet[str] = frozenset(
    {"BoolType", "IntType", "FloatType", "StringType", "NoneType"}
)


def _is_primitive_type(t) -> bool:
    """True for scalar types and lists/optionals of scalar types.

    The fold pass will only walk forward through nodes whose every output
    is a primitive: a `Tensor`-typed output means the size value reaches
    tensor production and must be preserved as a symbolic / dynamic dim.
    """
    kind = t.kind()
    if kind in _PRIMITIVE_TYPE_KINDS:
        return True
    if kind in ("ListType", "OptionalType"):
        return _is_primitive_type(t.getElementType())
    return False


_CONTROL_FLOW_SINKS = {
    # `prim::If` consumes its condition at operand 0; the value never
    # appears in the chosen block's NNEF emission.
    ("prim::If", 0),
    # `prim::Loop(max_trip_count, init_cond, ...)`: both are control-only.
    ("prim::Loop", 0),
    ("prim::Loop", 1),
}


def _reach_only_control_flow(source_value: "torch._C.Value") -> bool:
    """Decide whether folding `source_value` to a constant is shape-safe.

    Walks every forward-reachable use. Returns True iff every reach path
    terminates in a control-flow sink (`prim::If` condition, `prim::Loop`
    trip count / cond, `prim::RaiseException`) without ever crossing a
    node that produces a non-primitive (tensor-typed, ScriptObject, ...)
    output. The latter case means the size value participates in tensor
    production and must remain dynamic so the standard `aten::size`
    handler can route it through `tract_core_shape_of` under
    `inference_target.has_dynamic_axes`.

    Conservative: an unrecognized op kind whose outputs are all
    primitives is followed; an op kind that produces no output and
    isn't `prim::RaiseException` is treated as control-flow-only.
    """
    visited: T.Set[int] = set()
    queue: T.List[torch._C.Value] = [source_value]
    saw_terminal = False
    while queue:
        v = queue.pop()
        if id(v) in visited:
            continue
        visited.add(id(v))
        for use in v.uses():
            user = use.user
            offset = use.offset
            kind = user.kind()
            if (kind, offset) in _CONTROL_FLOW_SINKS:
                saw_terminal = True
                continue
            if kind == "prim::RaiseException":
                # value is being formatted into an exception message
                saw_terminal = True
                continue
            outputs = list(user.outputs())
            if not outputs:
                # No outputs -> pure side-effect node we already handled
                # above for known kinds; treat anything else as a refusal
                # to be safe.
                return False
            if not all(_is_primitive_type(o.type()) for o in outputs):
                # Some output is non-primitive (tensor-typed). The value
                # participates in tensor production -- refuse fold.
                return False
            for o in outputs:
                queue.append(o)
            saw_terminal = saw_terminal  # propagate; we may not have
            # hit a real sink yet but the path is still primitive-only.
    return saw_terminal


def replace_size_calls_with_constants(
    graph: "torch._C.Graph",
    example_inputs: T.Sequence[T.Any],
) -> int:
    """Fold size queries whose values flow only into control flow.

    Reach analysis: walks forward from each candidate source. A source is
    folded only when every reach path terminates in `prim::If` condition,
    `prim::Loop` trip count, or `prim::RaiseException` without ever
    crossing a node that produces a tensor-typed output. Sources whose
    value flows into tensor production (via `aten::reshape`, `aten::view`,
    `aten::expand`, `aten::zeros`, ...) are left alone, so the standard
    `aten::size` handler in `op/aten/other.py` can route them through
    `tract_core_shape_of` under `inference_target.has_dynamic_axes`.

    This makes the pass safe by default for any export target, including
    those declaring dynamic axes: a dim consumed by `aten::view` will not
    be baked into the NNEF graph as a constant.

    Returns the count of size-call nodes folded.
    """
    torch._C._jit_pass_complete_shape_analysis(
        graph, tuple(example_inputs), False
    )

    # Snapshot candidates before mutation; walking nested blocks while
    # destroying nodes invalidates the iterator.
    candidates = [
        n
        for n in _walk_nodes(graph)
        if n.kind() in ("aten::dim", "aten::numel", "aten::len", "aten::size")
    ]

    folded = 0
    for node in candidates:
        # Reach analysis: refuse the fold if any output participates in
        # tensor production.
        if not all(_reach_only_control_flow(out) for out in node.outputs()):
            continue
        kind = node.kind()
        inputs = list(node.inputs())
        new_val: T.Optional["torch._C.Value"] = None
        if kind == "aten::dim":
            sizes = _tensor_sizes(inputs[0])
            if sizes is not None:
                new_val = _insert_int_constant_before(graph, len(sizes), node)
        elif kind == "aten::numel":
            sizes = _tensor_sizes(inputs[0])
            if sizes is not None:
                n = 1
                for s in sizes:
                    n *= int(s)
                new_val = _insert_int_constant_before(graph, n, node)
        elif kind == "aten::len":
            in_ty = inputs[0].type()
            if in_ty.kind() == "TensorType":
                sizes = _tensor_sizes(inputs[0])
                if sizes is not None and len(sizes) > 0:
                    new_val = _insert_int_constant_before(
                        graph, int(sizes[0]), node
                    )
            elif inputs[0].node().kind() == "prim::ListConstruct":
                count = sum(1 for _ in inputs[0].node().inputs())
                new_val = _insert_int_constant_before(graph, count, node)
        elif kind == "aten::size":
            sizes = _tensor_sizes(inputs[0])
            if sizes is None:
                pass
            elif len(inputs) == 1:
                new_val = _insert_int_list_constant(graph, sizes, node)
            elif len(inputs) == 2:
                dim_node = inputs[1].node()
                if dim_node.kind() == "prim::Constant":
                    try:
                        dim_val = dim_node["value"]
                    except RuntimeError:
                        dim_val = None
                    if dim_val is not None:
                        if dim_val < 0:
                            dim_val += len(sizes)
                        if 0 <= dim_val < len(sizes):
                            new_val = _insert_int_constant_before(
                                graph, int(sizes[dim_val]), node
                            )
        if new_val is None:
            continue
        node.output().replaceAllUsesWith(new_val)
        node.destroy()
        folded += 1
    return folded


_SCALAR_BINARY_FOLDERS: T.Mapping[str, T.Callable[[T.Any, T.Any], T.Any]] = {
    "aten::eq": lambda a, b: a == b,
    "aten::ne": lambda a, b: a != b,
    "aten::lt": lambda a, b: a < b,
    "aten::le": lambda a, b: a <= b,
    "aten::gt": lambda a, b: a > b,
    "aten::ge": lambda a, b: a >= b,
}


def _const_value(value):
    """Resolve an SSA value to its compile-time Python constant.

    Returns the Python value when `value` is the output of a
    `prim::Constant`, otherwise the sentinel `_NOT_CONST`.
    """
    n = value.node()
    if n.kind() != "prim::Constant":
        return _NOT_CONST
    out = n.output()
    out_ty = out.type().kind()
    if out_ty == "BoolType":
        return bool(n["value"])
    if out_ty == "IntType":
        return int(n["value"])
    if out_ty == "FloatType":
        return float(n["value"])
    if out_ty == "StringType":
        return str(n["value"])
    if out_ty == "NoneType":
        return None
    return _NOT_CONST


_NOT_CONST = object()


def _emit_python_constant(graph, value, before_node):
    """Insert a `prim::Constant` carrying a Python scalar.

    Supports bool, int, float; placed just before `before_node`.
    """
    n = graph.create("prim::Constant")
    if isinstance(value, bool):
        n.output().setType(torch._C.BoolType.get())
        n.i_("value", int(value))
    elif isinstance(value, int):
        n.output().setType(torch._C.IntType.get())
        n.i_("value", int(value))
    elif isinstance(value, float):
        n.output().setType(torch._C.FloatType.get())
        n.f_("value", float(value))
    else:
        raise NotImplementedError(f"unsupported constant {type(value)}")
    n.insertBefore(before_node)
    return n.output()


def fold_constant_scalar_arithmetic(graph: "torch._C.Graph") -> int:
    """Fold scalar arithmetic on `prim::Constant` operands.

    Walks `aten::eq/ne/lt/le/gt/ge`, `aten::__not__`,
    `aten::__contains__`, and the unary `aten::Bool/Int/Float` casts.
    Standalone replacement for `_jit_pass_constant_propagation`: used
    in the JIT-only export chain to avoid a torch internal assertion
    that fires when the upstream pass walks a graph mixing Phase 1
    inlined submodules and Phase 2 size-fold constants.

    Returns the number of nodes folded.
    """
    folded = 0
    unary_casts = {
        "aten::Bool": bool,
        "aten::Int": int,
        "aten::Float": float,
    }
    candidates = [
        n
        for n in _walk_nodes(graph)
        if n.kind() in _SCALAR_BINARY_FOLDERS
        or n.kind() in ("aten::__not__", "aten::__contains__")
        or n.kind() in unary_casts
    ]
    for node in candidates:
        kind = node.kind()
        inputs = list(node.inputs())
        new_val = None
        if kind in _SCALAR_BINARY_FOLDERS and len(inputs) == 2:
            a = _const_value(inputs[0])
            b = _const_value(inputs[1])
            if a is _NOT_CONST or b is _NOT_CONST:
                continue
            new_val = _emit_python_constant(
                graph, _SCALAR_BINARY_FOLDERS[kind](a, b), node
            )
        elif kind in unary_casts and len(inputs) == 1:
            a = _const_value(inputs[0])
            if a is _NOT_CONST:
                continue
            new_val = _emit_python_constant(graph, unary_casts[kind](a), node)
        elif kind == "aten::__not__" and len(inputs) == 1:
            a = _const_value(inputs[0])
            if a is _NOT_CONST:
                continue
            new_val = _emit_python_constant(graph, not a, node)
        elif kind == "aten::__contains__" and len(inputs) == 2:
            haystack_node = inputs[0].node()
            needle = _const_value(inputs[1])
            if needle is _NOT_CONST:
                continue
            if haystack_node.kind() != "prim::ListConstruct":
                continue
            elems = []
            for el in haystack_node.inputs():
                v = _const_value(el)
                if v is _NOT_CONST:
                    elems = None
                    break
                elems.append(v)
            if elems is None:
                continue
            new_val = _emit_python_constant(graph, needle in elems, node)
        if new_val is None:
            continue
        node.output().replaceAllUsesWith(new_val)
        node.destroy()
        folded += 1
    return folded


def strip_prim_data(graph: "torch._C.Graph") -> int:
    """Replace `prim::data(t)` nodes with their input.

    `prim::data` is the IR form of Tensor `.data` access (detaches from
    autograd). In inference it is a no-op; t2n's parser doesn't have a
    handler for it, so we elide it.
    """
    folded = 0
    for node in list(_walk_nodes(graph)):
        if node.kind() != "prim::data":
            continue
        inputs = list(node.inputs())
        if len(inputs) != 1:
            continue
        node.output().replaceAllUsesWith(inputs[0])
        node.destroy()
        folded += 1
    return folded


def fold_tuple_index_through_tuple_construct(
    graph: "torch._C.Graph",
) -> int:
    """Fold `prim::TupleIndex(tuple_const, k)` into the k-th tuple input.

    JIT artifacts whose Python source builds a tuple at the call site
    (e.g. `return (h, c)`) and consumes it later via positional indexing
    (`pair[0]`, `pair[1]`) leave behind `prim::TupleConstruct ->
    prim::TupleIndex` chains in the inlined graph. t2n's parser already
    knows about `TupleConstruct` and `TupleUnpack`, but `TupleIndex` is
    unsupported. When the index is a static `prim::Constant int` and the
    tuple value is the direct output of a `TupleConstruct`, we rewire
    `TupleIndex`'s output to the tuple's k-th input verbatim, leaving
    the `TupleConstruct` itself in place (DCE removes it later if it
    has no other consumers).

    Returns the count of nodes folded.
    """
    folded = 0
    for node in list(_walk_nodes(graph)):
        if node.kind() != "prim::TupleIndex":
            continue
        inputs = list(node.inputs())
        if len(inputs) != 2:
            continue
        tuple_node = inputs[0].node()
        if tuple_node.kind() != "prim::TupleConstruct":
            continue
        idx_node = inputs[1].node()
        if idx_node.kind() != "prim::Constant":
            continue
        try:
            idx = int(idx_node["value"])
        except (RuntimeError, TypeError):
            continue
        tuple_inputs = list(tuple_node.inputs())
        if not 0 <= idx < len(tuple_inputs):
            continue
        node.output().replaceAllUsesWith(tuple_inputs[idx])
        node.destroy()
        folded += 1
    return folded


def fold_constant_ifs(graph: "torch._C.Graph") -> int:
    """Fold `prim::If` nodes whose condition is a `prim::Constant[bool]`.

    Replaces the If with the chosen block's nodes. Returns the count
    folded.
    """
    folded = 0
    changed = True
    while changed:
        changed = False
        for node in list(_walk_nodes(graph)):
            if node.kind() != "prim::If":
                continue
            cond_node = next(node.inputs()).node()
            if cond_node.kind() != "prim::Constant":
                continue
            try:
                cond = bool(cond_node["value"])
            except (RuntimeError, TypeError):
                continue
            blocks = list(node.blocks())
            if len(blocks) != 2:
                continue
            keep = blocks[0] if cond else blocks[1]
            keep_outs = list(keep.returnNode().inputs())
            node_outs = list(node.outputs())
            if len(keep_outs) != len(node_outs):
                continue
            for old, new in zip(node_outs, keep_outs, strict=True):
                old.replaceAllUsesWith(new)
            for n in list(keep.nodes()):
                n.moveBefore(node)
            node.destroy()
            folded += 1
            changed = True
            break
    return folded


def strip_assertion_ifs(graph: "torch._C.Graph") -> int:
    """Drop `prim::If` nodes whose one branch is purely a `RaiseException`.

    Replace uses of the If's outputs with the non-raising block's outputs,
    then destroy the If. Walks nested blocks (assertion ifs are often
    inside other prim::If branches). Returns the count of stripped nodes.
    """
    changed = True
    total = 0
    while changed:
        changed = False
        for node in list(_walk_nodes(graph)):
            if node.kind() != "prim::If":
                continue
            blocks = list(node.blocks())
            if len(blocks) != 2:
                continue
            true_blk, false_blk = blocks
            if _block_only_raises(true_blk):
                keep = false_blk
            elif _block_only_raises(false_blk):
                keep = true_blk
            else:
                continue
            keep_outs = list(keep.returnNode().inputs())
            node_outs = list(node.outputs())
            if len(keep_outs) != len(node_outs):
                continue
            for old, new in zip(node_outs, keep_outs, strict=True):
                old.replaceAllUsesWith(new)
            for n in list(keep.nodes()):
                n.moveBefore(node)
            node.destroy()
            changed = True
            total += 1
            # Iterators over a parent block may now be invalidated; restart.
            break
    return total
