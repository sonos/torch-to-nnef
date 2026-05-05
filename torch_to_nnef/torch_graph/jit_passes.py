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
