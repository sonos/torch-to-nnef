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

import contextlib
import typing as T

import torch

from torch_to_nnef.torch_graph.jit_utils import walk_nodes
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_ADD,
    ATEN_BOOL,
    ATEN_CONTAINS,
    ATEN_DIM,
    ATEN_EQ,
    ATEN_FLOAT,
    ATEN_FORMAT,
    ATEN_GE,
    ATEN_GETITEM,
    ATEN_GT,
    ATEN_INT,
    ATEN_LE,
    ATEN_LEN,
    ATEN_LT,
    ATEN_MUL,
    ATEN_NE,
    ATEN_NOT,
    ATEN_NUMEL,
    ATEN_SIZE_KIND,
    ATEN_STR,
    BOOLTYPE_KIND,
    CONSTANT_KIND,
    DATA_KIND,
    DEVICE_KIND,
    DTYPE_KIND,
    FLOATTYPE_KIND,
    IF_KIND,
    INTTYPE_KIND,
    LISTCONSTRUCT_KIND,
    LISTTYPE_KIND,
    LOOP_KIND,
    NONETYPE_KIND,
    OPTIONALTYPE_KIND,
    RAISE_EXCEPTION_KIND,
    STRINGTYPE_KIND,
    TENSORTYPE_KIND,
    TUPLECONSTRUCT_KIND,
    TUPLEINDEX_KIND,
    TUPLEUNPACK_KIND,
)

ASSERTION_BLOCK_KINDS: T.Tuple[str, ...] = (
    CONSTANT_KIND,
    LISTCONSTRUCT_KIND,
    TUPLECONSTRUCT_KIND,
    RAISE_EXCEPTION_KIND,
    ATEN_FORMAT,
    ATEN_GETITEM,
    ATEN_CONTAINS,
    ATEN_NOT,
    ATEN_DIM,
    ATEN_EQ,
    ATEN_NE,
    ATEN_INT,
    ATEN_STR,
    ATEN_ADD,
    ATEN_MUL,
    DTYPE_KIND,
    DEVICE_KIND,
)


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
    if nodes[-1].kind() != RAISE_EXCEPTION_KIND:
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
    graph: torch._C.Graph,
    value: int,
    before_node: torch._C.Node,
) -> torch._C.Value:
    """Insert a `prim::Constant` int just before `before_node`.

    Uses `Graph.create` + `Node.insertBefore` rather than the higher-level
    `Graph.insertConstant`, which interacts badly with `setInsertPoint`
    when the target lives in a sub-block. Returns the SSA value.
    """
    n = graph.create(CONSTANT_KIND)
    n.output().setType(torch._C.IntType.get())
    n.i_("value", int(value))
    n.insertBefore(before_node)
    return n.output()


def _insert_int_list_constant(
    graph: torch._C.Graph,
    values: T.Sequence[int],
    before_node: torch._C.Node,
) -> torch._C.Value:
    """Materialize a `prim::ListConstruct` of int constants.

    Inserts the elements and the list construct just before `before_node`.
    Returns the list SSA value.
    """
    elem_vals = [
        _insert_int_constant_before(graph, v, before_node) for v in values
    ]
    lc = graph.create(LISTCONSTRUCT_KIND, 1)
    lc.output().setType(torch._C.ListType(torch._C.IntType.get()))
    for v in elem_vals:
        lc.addInput(v)
    lc.insertBefore(before_node)
    return lc.output()


_PRIMITIVE_TYPE_KINDS: T.FrozenSet[str] = frozenset(
    {
        BOOLTYPE_KIND,
        INTTYPE_KIND,
        FLOATTYPE_KIND,
        STRINGTYPE_KIND,
        NONETYPE_KIND,
    }
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
    if kind in (LISTTYPE_KIND, OPTIONALTYPE_KIND):
        return _is_primitive_type(t.getElementType())
    return False


_CONTROL_FLOW_SINKS = {
    # `prim::If` consumes its condition at operand 0; the value never
    # appears in the chosen block's NNEF emission.
    (IF_KIND, 0),
    # `prim::Loop(max_trip_count, init_cond, ...)`: both are control-only.
    (LOOP_KIND, 0),
    (LOOP_KIND, 1),
}


def _reach_only_control_flow(source_value: torch._C.Value) -> bool:
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
            if kind == RAISE_EXCEPTION_KIND:
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
            # `saw_terminal` is only flipped True at a real sink; an
            # all-primitive path that has not yet reached one is still
            # eligible to fold once the BFS catches up to a sink.
    return saw_terminal


def replace_size_calls_with_constants(
    graph: torch._C.Graph,
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
    # Older / partially-resolved graphs may carry unresolved submodule
    # references that the shape analyzer rejects. Continue without the
    # propagation in that case; many fold candidates can still be resolved
    # from the graph's pre-existing type annotations.
    with contextlib.suppress(RuntimeError):
        torch._C._jit_pass_complete_shape_analysis(
            graph, tuple(example_inputs), False
        )

    # Snapshot candidates before mutation; walking nested blocks while
    # destroying nodes invalidates the iterator.
    candidates = [
        n
        for n in walk_nodes(graph)
        if n.kind() in (ATEN_DIM, ATEN_NUMEL, ATEN_LEN, ATEN_SIZE_KIND)
    ]

    folded = 0
    for node in candidates:
        # Reach analysis: refuse the fold if any output participates in
        # tensor production.
        if not all(_reach_only_control_flow(out) for out in node.outputs()):
            continue
        new_val = _materialize_size_fold(graph, node)
        if new_val is None:
            continue
        node.output().replaceAllUsesWith(new_val)
        node.destroy()
        folded += 1
    return folded


def _materialize_size_fold(
    graph: torch._C.Graph, node: torch._C.Node
) -> T.Optional[torch._C.Value]:
    """Build the constant that replaces `node`, or None if not foldable."""
    inputs = list(node.inputs())
    sizes = _tensor_sizes(inputs[0]) if inputs else None
    kind = node.kind()
    if kind == ATEN_DIM:
        return (
            _insert_int_constant_before(graph, len(sizes), node)
            if sizes is not None
            else None
        )
    if kind == ATEN_NUMEL:
        if sizes is None:
            return None
        concrete_sizes = T.cast(T.List[int], sizes)
        n = 1
        # pylint: disable-next=not-an-iterable
        for s in concrete_sizes:
            n *= int(s)
        return _insert_int_constant_before(graph, n, node)
    if kind == ATEN_LEN:
        return _fold_len(graph, node, inputs, sizes)
    if kind == ATEN_SIZE_KIND:
        return _fold_size(graph, node, inputs, sizes)
    return None


def _fold_len(graph, node, inputs, sizes):
    in_ty = inputs[0].type()
    if in_ty.kind() == TENSORTYPE_KIND:
        if sizes is not None and len(sizes) > 0:
            return _insert_int_constant_before(graph, int(sizes[0]), node)
        return None
    if inputs[0].node().kind() == LISTCONSTRUCT_KIND:
        count = sum(1 for _ in inputs[0].node().inputs())
        return _insert_int_constant_before(graph, count, node)
    return None


def _fold_size(graph, node, inputs, sizes):
    if sizes is None:
        return None
    if len(inputs) == 1:
        return _insert_int_list_constant(graph, sizes, node)
    dim_val = _resolve_const_dim(inputs, sizes)
    if dim_val is None:
        return None
    return _insert_int_constant_before(graph, int(sizes[dim_val]), node)


def _resolve_const_dim(inputs, sizes) -> T.Optional[int]:
    """Extract a non-negative dim index from `inputs[1]`, or None."""
    if len(inputs) != 2:
        return None
    dim_node = inputs[1].node()
    if dim_node.kind() != CONSTANT_KIND:
        return None
    try:
        dim_val = dim_node["value"]
    except RuntimeError:
        return None
    if dim_val is None:
        return None
    if dim_val < 0:
        dim_val += len(sizes)
    if not 0 <= dim_val < len(sizes):
        return None
    return dim_val


_SCALAR_BINARY_FOLDERS: T.Mapping[str, T.Callable[[T.Any, T.Any], T.Any]] = {
    ATEN_EQ: lambda a, b: a == b,
    ATEN_NE: lambda a, b: a != b,
    ATEN_LT: lambda a, b: a < b,
    ATEN_LE: lambda a, b: a <= b,
    ATEN_GT: lambda a, b: a > b,
    ATEN_GE: lambda a, b: a >= b,
}


_PRIM_CONSTANT_PARSERS: T.Mapping[str, T.Callable[[T.Any], T.Any]] = {
    BOOLTYPE_KIND: bool,
    INTTYPE_KIND: int,
    FLOATTYPE_KIND: float,
    STRINGTYPE_KIND: str,
}


def _const_value(value):
    """Resolve an SSA value to its compile-time Python constant.

    Returns the Python value when `value` is the output of a
    `prim::Constant`, otherwise the sentinel `_NOT_CONST`.
    """
    n = value.node()
    if n.kind() != CONSTANT_KIND:
        return _NOT_CONST
    out_ty = n.output().type().kind()
    if out_ty == NONETYPE_KIND:
        return None
    parser = _PRIM_CONSTANT_PARSERS.get(out_ty)
    return parser(n["value"]) if parser is not None else _NOT_CONST


_NOT_CONST = object()


def _emit_python_constant(graph, value, before_node):
    """Insert a `prim::Constant` carrying a Python scalar.

    Supports bool, int, float; placed just before `before_node`.
    """
    n = graph.create(CONSTANT_KIND)
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


def fold_constant_scalar_arithmetic(graph: torch._C.Graph) -> int:
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
        ATEN_BOOL: bool,
        ATEN_INT: int,
        ATEN_FLOAT: float,
    }
    candidates = [
        n
        for n in walk_nodes(graph)
        if n.kind() in _SCALAR_BINARY_FOLDERS
        or n.kind() in (ATEN_NOT, ATEN_CONTAINS)
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
        elif kind == ATEN_NOT and len(inputs) == 1:
            a = _const_value(inputs[0])
            if a is _NOT_CONST:
                continue
            new_val = _emit_python_constant(graph, not a, node)
        elif kind == ATEN_CONTAINS and len(inputs) == 2:
            haystack_node = inputs[0].node()
            needle = _const_value(inputs[1])
            if needle is _NOT_CONST:
                continue
            if haystack_node.kind() != LISTCONSTRUCT_KIND:
                continue
            elems: T.List[T.Any] = []
            all_const = True
            for el in haystack_node.inputs():
                v = _const_value(el)
                if v is _NOT_CONST:
                    all_const = False
                    break
                elems.append(v)
            if not all_const:
                continue
            new_val = _emit_python_constant(graph, needle in elems, node)
        if new_val is None:
            continue
        node.output().replaceAllUsesWith(new_val)
        node.destroy()
        folded += 1
    return folded


def strip_prim_data(graph: torch._C.Graph) -> int:
    """Replace `prim::data(t)` nodes with their input.

    `prim::data` is the IR form of Tensor `.data` access (detaches from
    autograd). In inference it is a no-op; t2n's parser doesn't have a
    handler for it, so we elide it.
    """
    folded = 0
    for node in list(walk_nodes(graph)):
        if node.kind() != DATA_KIND:
            continue
        inputs = list(node.inputs())
        if len(inputs) != 1:
            continue
        node.output().replaceAllUsesWith(inputs[0])
        node.destroy()
        folded += 1
    return folded


def fold_tuple_index_through_tuple_construct(
    graph: torch._C.Graph,
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
    for node in list(walk_nodes(graph)):
        if node.kind() != TUPLEINDEX_KIND:
            continue
        inputs = list(node.inputs())
        if len(inputs) != 2:
            continue
        tuple_node = inputs[0].node()
        if tuple_node.kind() != TUPLECONSTRUCT_KIND:
            continue
        idx_node = inputs[1].node()
        if idx_node.kind() != CONSTANT_KIND:
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


def fold_tuple_unpack_through_tuple_construct(
    graph: torch._C.Graph,
) -> int:
    """Fold `prim::TupleUnpack(prim::TupleConstruct(...))` into the inputs.

    Sibling of `fold_tuple_index_through_tuple_construct`. JIT artifacts
    that build a tuple at the call site and consume it via destructuring
    assignment (`a, b = my_pair()`) leave behind a
    `prim::TupleConstruct -> prim::TupleUnpack` chain after inlining.
    The k-th unpack output is exactly the k-th construct input; we
    rewire each unpack output verbatim and destroy the unpack. The
    `TupleConstruct` is left in place; DCE removes it if its only
    consumers (the unpack outputs) are now gone.

    Returns the count of `TupleUnpack` nodes folded.
    """
    folded = 0
    for node in list(walk_nodes(graph)):
        if node.kind() != TUPLEUNPACK_KIND:
            continue
        inputs = list(node.inputs())
        if len(inputs) != 1:
            continue
        tuple_node = inputs[0].node()
        if tuple_node.kind() != TUPLECONSTRUCT_KIND:
            continue
        tuple_inputs = list(tuple_node.inputs())
        unpack_outputs = list(node.outputs())
        if len(tuple_inputs) != len(unpack_outputs):
            continue
        for old_out, new_val in zip(unpack_outputs, tuple_inputs, strict=True):
            old_out.replaceAllUsesWith(new_val)
        node.destroy()
        folded += 1
    return folded


def fold_constant_ifs(graph: torch._C.Graph) -> int:
    """Fold `prim::If` nodes whose condition is a `prim::Constant[bool]`.

    Replaces the If with the chosen block's nodes. Returns the count
    folded.
    """
    folded = 0
    changed = True
    while changed:
        changed = False
        for node in list(walk_nodes(graph)):
            if node.kind() != IF_KIND:
                continue
            cond_node = next(node.inputs()).node()
            if cond_node.kind() != CONSTANT_KIND:
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


def strip_assertion_ifs(graph: torch._C.Graph) -> int:
    """Drop `prim::If` nodes whose one branch is purely a `RaiseException`.

    Replace uses of the If's outputs with the non-raising block's outputs,
    then destroy the If. Walks nested blocks (assertion ifs are often
    inside other prim::If branches). Returns the count of stripped nodes.
    """
    changed = True
    total = 0
    while changed:
        changed = False
        for node in list(walk_nodes(graph)):
            if node.kind() != IF_KIND:
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


def _eval_value_by_example(
    graph: torch._C.Graph,
    value: torch._C.Value,
    example_inputs: T.Sequence[T.Any],
) -> T.Any:
    """Run the graph with the given example inputs, returning `value`.

    Temporarily swaps the graph's outputs for `value`, executes via
    `torch._C._jit_interpret_graph`, and restores the original outputs.
    The mutation/restore happens under try/finally so the graph is left
    in its original state even on interpreter failure.
    """
    orig_outputs = list(graph.outputs())
    n_outs = len(orig_outputs)
    for _ in range(n_outs):
        graph.eraseOutput(0)
    graph.registerOutput(value)
    try:
        return torch._C._jit_interpret_graph(graph, tuple(example_inputs))
    finally:
        graph.eraseOutput(0)
        for o in orig_outputs:
            graph.registerOutput(o)


def fold_data_dependent_ifs(
    graph: torch._C.Graph,
    example_inputs: T.Sequence[T.Any],
) -> int:
    """Fold `prim::If` nodes whose condition is data-dependent on the input.

    PyTorch's JIT shape-analysis passes do not propagate shapes through
    `prim::If` nodes that produce tensors, leaving runtime dim/shape
    checks (e.g. `nn.LSTMCell`'s `if input.dim() == 1: ...`) unresolved
    by `replace_size_calls_with_constants` + `fold_constant_ifs`. To
    specialize the graph for the user's example inputs, we evaluate
    each remaining `prim::If`'s condition by running the graph itself
    with the example, observing the chosen branch, and inlining it.

    Only top-level Ifs are folded each pass; nested Ifs surface to the
    top once their parent is removed, so a fixed-point loop catches
    them too.

    Returns the number of Ifs folded.

    Requires `torch._C._jit_interpret_graph`, exposed since torch 1.10.
    The probe is deferred until a candidate If is actually found, so
    callers on older torch with already-clean graphs (no remaining Ifs)
    return 0 without raising. The rest of `torch_to_nnef` still works
    on torch 1.8+, only this one pass needs the newer API.
    """
    top_block = graph.return_node().owningBlock()
    total = 0
    for _ in range(20):
        candidates = [
            n
            for n in graph.nodes()
            if n.kind() == IF_KIND and n.owningBlock() == top_block
        ]
        if candidates and not hasattr(torch._C, "_jit_interpret_graph"):
            raise RuntimeError(
                "fold_data_dependent_ifs requires torch>=1.10 (needs "
                "`torch._C._jit_interpret_graph`). Detected torch=="
                f"{torch.__version__}. Upgrade torch, or skip this pass "
                "and use the rest of the chain (size folds, constant ifs, "
                "etc.)."
            )
        folded = 0
        for if_node in candidates:
            cond_v = next(if_node.inputs())
            try:
                cond_val = _eval_value_by_example(graph, cond_v, example_inputs)
            except (RuntimeError, TypeError):
                # `_jit_interpret_graph` raises RuntimeError on graph
                # state it cannot execute (missing types, unhandled op,
                # malformed inputs). TypeError covers Python-level mismatches
                # (e.g. example arg shape vs graph input).
                continue
            if not isinstance(cond_val, bool):
                continue
            blocks = list(if_node.blocks())
            if len(blocks) != 2:
                continue
            chosen = blocks[0 if cond_val else 1]
            keep_outs = list(chosen.returnNode().inputs())
            node_outs = list(if_node.outputs())
            if len(keep_outs) != len(node_outs):
                continue
            for old, new in zip(node_outs, keep_outs, strict=True):
                old.replaceAllUsesWith(new)
            for n in list(chosen.nodes()):
                n.moveBefore(if_node)
            if_node.destroy()
            folded += 1
        if folded == 0:
            break
        total += folded
    return total
