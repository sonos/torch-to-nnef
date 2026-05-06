"""Shared utilities for JIT graph traversal and qualname parsing.

These small helpers are used by `jit_inline`, `jit_passes`, and the
recursive parser in `ir_helpers`. Centralized here to avoid duplication.
"""

from __future__ import annotations

import typing as T

import torch  # noqa: F401  -- kept for type-hint forward refs


def walk_nodes(graph_or_block) -> T.Iterator["torch._C.Node"]:
    """Yield nodes recursively, descending into prim::If / prim::Loop blocks."""
    for node in graph_or_block.nodes():
        yield node
        for blk in node.blocks():
            yield from walk_nodes(blk)


def parse_jit_qualname(qualname: str) -> T.Tuple[str, str]:
    """Return ``(module_path, class_name)`` for a `__torch__.*` qualname.

    Strips the `__torch__.` prefix and `___torch_mangle_*` mangling
    segments; the trailing path component is the class name and the
    rest is the import path. Returns ``("", "")`` when `qualname` is
    not a `__torch__.*` class reference.

    Examples:
        ``"__torch__.vad.model.SileroBlock"`` ->
        ``("vad.model", "SileroBlock")``

        ``"__torch__.torch.nn.modules.rnn.___torch_mangle_8.LSTM"`` ->
        ``("torch.nn.modules.rnn", "LSTM")``
    """
    if not qualname.startswith("__torch__."):
        return "", ""
    parts = [
        p
        for p in qualname[len("__torch__.") :].split(".")
        if "___torch_mangle_" not in p
    ]
    if len(parts) < 2:
        return "", ""
    return ".".join(parts[:-1]), parts[-1]
