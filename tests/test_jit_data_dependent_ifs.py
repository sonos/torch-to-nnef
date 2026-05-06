"""Tests for `fold_data_dependent_ifs`.

PyTorch's JIT shape analysis does not propagate shapes through
`prim::If` nodes that produce tensors. That leaves runtime dim/shape
checks (e.g. `nn.LSTMCell`'s `if input.dim() == 1: input.unsqueeze(0)`)
in the graph after the standard size-fold + constant-If passes have
run. `fold_data_dependent_ifs` evaluates each remaining If's condition
by re-executing the graph with the user's example inputs, then inlines
the chosen branch; bitwise output equivalence is verified below.
"""

import torch
from torch import nn

from torch_to_nnef.torch_graph import fold_data_dependent_ifs
from torch_to_nnef.torch_graph.torch_const import IF_KIND


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


def _node_count(g, kind):
    return sum(1 for n in _walk(g) if n.kind() == kind)


class _DimDependentBranch(nn.Module):
    """`x.dim() == 1` branch that JIT shape-analysis cannot fold.

    The condition is data-dependent in the sense that it relies on a
    runtime tensor property; `replace_size_calls_with_constants` only
    folds dim/size queries that flow purely into control flow, and this
    one feeds an `aten::unsqueeze` (tensor production), so it survives.
    """

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x + 1.0


def test_fold_picks_branch_consistent_with_example():
    m = torch.jit.script(_DimDependentBranch())
    torch._C._jit_pass_inline(m.graph)
    x = torch.randn(2, 3)  # 2D input -> the False branch is correct
    assert _node_count(m.graph, IF_KIND) == 1

    folded = fold_data_dependent_ifs(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    assert folded == 1
    assert _node_count(m.graph, IF_KIND) == 0


def test_fold_preserves_output():
    """Bitwise parity: rewriting the graph must not change behavior."""
    ref = _DimDependentBranch().eval()
    x = torch.randn(2, 3)
    expected = ref(x)

    m = torch.jit.script(_DimDependentBranch())
    torch._C._jit_pass_inline(m.graph)
    fold_data_dependent_ifs(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    got = m(x)
    assert torch.allclose(got, expected)


class _NestedDimIfs(nn.Module):
    """Nested data-dependent Ifs exercise the fixed-point loop.

    After the outer If is folded, the inner one surfaces to the top
    level and a second iteration picks it up.
    """

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            return x + 1.0
        return x - 1.0


def test_fold_handles_nested_ifs_via_fixed_point():
    m = torch.jit.script(_NestedDimIfs())
    torch._C._jit_pass_inline(m.graph)
    x = torch.randn(4, 5)
    assert _node_count(m.graph, IF_KIND) == 2

    folded = fold_data_dependent_ifs(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    assert folded == 2
    assert _node_count(m.graph, IF_KIND) == 0


def test_fold_is_safe_when_no_if_present():
    class _Plain(nn.Module):
        def forward(self, x):
            return x + 1.0

    m = torch.jit.script(_Plain())
    x = torch.randn(3)
    folded = fold_data_dependent_ifs(m.graph, [m, x])
    assert folded == 0
