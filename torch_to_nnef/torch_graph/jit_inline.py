"""Selective inlining of unresolvable JIT submodule calls.

`torch_to_nnef`'s recursive parser identifies the class behind a
`prim::CallMethod` via `importlib.import_module(qualname.module_path)` and
`isinstance(submod, ref_cls)`. JIT artifacts shipped without their training
source (Silero-VAD, FunASR, ...) carry qualified names like
`__torch__.vad.model.vad_annotator.SileroVadBlock` that fail to import.

`inline_unresolvable_submodules(graph, model)` walks the JIT graph and
inlines exactly those calls in place, leaving importable `torch.nn.*`
calls intact so existing module-level extractors (LSTM, GRU, RNN, ...)
still fire.

Companion to `strip_assertion_ifs` in this package: a typical JIT-only
preprocessing chain is `inline_unresolvable_submodules -> _jit_pass_dce
-> strip_assertion_ifs`.
"""

from __future__ import annotations

import importlib
import typing as T

import torch
from torch import nn

GETATTR_KIND = "prim::GetAttr"
CALLMETHOD_KIND = "prim::CallMethod"


def _walk_nodes(graph_or_block) -> T.Iterator["torch._C.Node"]:
    """Yield nodes recursively, descending into prim::If / prim::Loop blocks."""
    for node in graph_or_block.nodes():
        yield node
        for blk in node.blocks():
            yield from _walk_nodes(blk)


def _qualname_module_path(qualname: str) -> str:
    """Return the `importlib`-compatible module path for a JIT qualname.

    Strips `__torch__.` prefix and `___torch_mangle_*` mangling segments,
    then drops the trailing class name. Returns empty string if the
    qualname isn't a class ref.
    """
    if not qualname.startswith("__torch__."):
        return ""
    parts = [
        p
        for p in qualname[len("__torch__.") :].split(".")
        if "___torch_mangle_" not in p
    ]
    if len(parts) < 2:
        return ""
    return ".".join(parts[:-1])


def _is_resolvable_qualname(qualname: str) -> bool:
    mod_path = _qualname_module_path(qualname)
    if not mod_path:
        # Not a class qualname; treat as resolvable (nothing to do).
        return True
    try:
        importlib.import_module(mod_path)
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _resolve_call_target(
    node: "torch._C.Node", root_model: nn.Module
) -> T.Tuple[nn.Module, "torch._C.Graph"]:
    """Resolve a CallMethod node to its target submodule and method graph.

    Walks the GetAttr chain from the CallMethod's first input back to the
    root `self` parameter, traverses the model accordingly, and returns
    `(submodule, method_graph)`.
    """
    method_name = node.s("name")
    first_in = next(node.inputs())

    sequence: T.List[str] = []
    cur = first_in.node()
    while cur.kind() == GETATTR_KIND:
        sequence.append(cur.s("name"))
        cur = next(cur.inputs()).node()
    # `cur` should now be the graph param node for `self`. We don't
    # validate it here because the caller already checked the qualname.
    sequence = sequence[::-1]

    submodule: nn.Module = root_model
    for attr in sequence:
        submodule = getattr(submodule, attr)

    method = getattr(submodule, method_name)
    if not hasattr(method, "graph"):
        raise RuntimeError(
            f"Cannot inline {method_name} on {type(submodule).__name__}: "
            "method has no .graph attribute"
        )
    return submodule, method.graph


def _find_inlineable_callmethod(
    graph: "torch._C.Graph",
) -> T.Optional["torch._C.Node"]:
    for node in _walk_nodes(graph):
        if node.kind() != CALLMETHOD_KIND:
            continue
        first_in = next(node.inputs())
        ty = first_in.type()
        try:
            qualname = ty.qualified_name()
        except (RuntimeError, AttributeError):
            continue
        if _is_resolvable_qualname(qualname):
            continue
        return node
    return None


def _inline_one_callmethod(
    graph: "torch._C.Graph",
    node: "torch._C.Node",
    root_model: nn.Module,
) -> None:
    _, sub_graph = _resolve_call_target(node, root_model)

    # `Graph.insertGraph(sub_graph, vals)` clones sub_graph's body at the
    # current insert point, substituting sub_graph's inputs with `vals`.
    # sub_graph's first input is the submodule's `self`; node's first
    # input is the value referencing that submodule in the parent graph
    # (a GetAttr chain) -- the substitution correctly rewires nested
    # GetAttrs through the parent's path.
    graph.setInsertPoint(node)
    new_outputs = graph.insertGraph(sub_graph, list(node.inputs()))

    for old_out, new_out in zip(node.outputs(), new_outputs, strict=True):
        old_out.replaceAllUsesWith(new_out)
    node.destroy()


def inline_unresolvable_submodules(
    graph: "torch._C.Graph", model: nn.Module
) -> int:
    """Inline every `prim::CallMethod` whose target class is not importable.

    Iterates until fixed point: an inlined body may itself contain further
    CallMethods that also need inlining.

    Returns the count of inlined calls.
    """
    inlined = 0
    # Bound the loop to avoid pathological non-termination on weirdly
    # structured graphs.
    max_passes = 1024
    for _ in range(max_passes):
        target = _find_inlineable_callmethod(graph)
        if target is None:
            return inlined
        _inline_one_callmethod(graph, target, model)
        inlined += 1
    raise RuntimeError(
        f"inline_unresolvable_submodules did not converge after "
        f"{max_passes} passes ({inlined} inlined so far)"
    )
