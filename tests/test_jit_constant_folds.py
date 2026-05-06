"""Tests for the standalone JIT constant-fold passes.

`fold_constant_scalar_arithmetic`, `fold_constant_ifs`, and
`strip_prim_data` are written so the JIT-only export chain can avoid
`torch._C._jit_pass_constant_propagation`, which has been observed to
trip an internal `setInsertPoint` assertion when applied to a graph that
mixes Phase 1 inlined submodules and Phase 2 size-fold constants.
"""

import torch
from torch import nn

from torch_to_nnef.torch_graph import (
    fold_constant_ifs,
    fold_constant_scalar_arithmetic,
    replace_size_calls_with_constants,
    strip_prim_data,
)


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


def _node_count(g, kind):
    return sum(1 for n in _walk(g) if n.kind() == kind)


class _DimEqIf(nn.Module):
    def forward(self, x):
        if x.dim() == 2:
            return x + 1.0
        return x * 2.0


def test_fold_constant_ifs_picks_true_branch():
    m = torch.jit.script(_DimEqIf())
    x = torch.randn(3, 4)

    # Phase 2 folds dim() to constant; our scalar arithmetic fold then
    # collapses `aten::eq(prim_const_2, prim_const_2)` to `True`.
    replace_size_calls_with_constants(m.graph, [m, x])
    fold_constant_scalar_arithmetic(m.graph)

    folded_ifs = fold_constant_ifs(m.graph)
    torch._C._jit_pass_dce(m.graph)

    assert folded_ifs == 1
    assert _node_count(m.graph, "prim::If") == 0


class _BoolNot(nn.Module):
    def forward(self, x, k: int):
        return x + (1.0 if not bool(k) else 0.0)


def test_fold_scalar_arithmetic_no_crash_on_runtime_inputs():
    """Smoke: the pass walks cleanly on a graph with no constant folds.

    The graph has `aten::Bool` and `aten::__not__` on a runtime int
    input so nothing should fold; just check no crash.
    """
    m = torch.jit.script(_BoolNot())
    folded = fold_constant_scalar_arithmetic(m.graph)
    assert folded >= 0


class _DataAccess(nn.Module):
    def forward(self, x):
        # `x.data` produces a `prim::data` node in the JIT IR.
        return x.data + 1.0


def test_strip_prim_data():
    m = torch.jit.script(_DataAccess())
    assert _node_count(m.graph, "prim::data") == 1
    n_stripped = strip_prim_data(m.graph)
    torch._C._jit_pass_dce(m.graph)
    assert n_stripped == 1
    assert _node_count(m.graph, "prim::data") == 0


def test_fold_chain_on_simple_dim_check():
    """End-to-end mini chain on a dim-check module.

    Order: fold sizes, fold scalar arithmetic, fold constant ifs, dce.
    Result must be a graph with no `prim::If` and an output that
    matches the unmodified module on a 2D input.
    """
    m = torch.jit.script(_DimEqIf())
    m_ref = torch.jit.script(_DimEqIf())
    x = torch.randn(3, 4)

    replace_size_calls_with_constants(m.graph, [m, x])
    fold_constant_scalar_arithmetic(m.graph)
    fold_constant_ifs(m.graph)
    torch._C._jit_pass_dce(m.graph)

    assert _node_count(m.graph, "prim::If") == 0
    with torch.no_grad():
        ref = m_ref(x)
        new = m(x)
    assert torch.allclose(ref, new)
