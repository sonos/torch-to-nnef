"""Tests for `fold_tuple_index_through_tuple_construct`.

PyTorch's scripter constant-folds `pair[0]` directly when the
`prim::TupleConstruct` is in the same scope as the consumption, so the
test models route the tuple build through a separate scripted method
that gets inlined via `_jit_pass_inline`. After inlining the pattern
becomes `prim::TupleConstruct -> prim::TupleIndex`, which the pass
under test rewrites.
"""

import pytest
import torch
from torch import nn

from torch_to_nnef.torch_graph import (
    fold_tuple_index_through_tuple_construct,
)
from torch_to_nnef.torch_graph.torch_const import (
    TUPLECONSTRUCT_KIND,
    TUPLEINDEX_KIND,
)
from torch_to_nnef.utils import torch_version

skipif_torch_lt_20 = pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason=(
        "torch < 2.0's `_jit_pass_inline` doesn't expose the "
        "`TupleConstruct -> TupleIndex` pattern the fold targets"
    ),
)


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


def _node_count(g, kind):
    return sum(1 for n in _walk(g) if n.kind() == kind)


class _PairBuilderTwoIndex(nn.Module):
    def helper(self, x, y):
        return (x, y)

    def forward(self, x, y):
        pair = self.helper(x, y)
        return pair[0] + pair[1]


def _scripted_inlined(cls):
    m = torch.jit.script(cls())
    torch._C._jit_pass_inline(m.graph)
    return m


@skipif_torch_lt_20
def test_fold_collapses_tupleindex_into_construct_input():
    m = _scripted_inlined(_PairBuilderTwoIndex)
    assert _node_count(m.graph, TUPLEINDEX_KIND) == 2
    assert _node_count(m.graph, TUPLECONSTRUCT_KIND) == 1

    folded = fold_tuple_index_through_tuple_construct(m.graph)
    torch._C._jit_pass_dce(m.graph)

    assert folded == 2
    assert _node_count(m.graph, TUPLEINDEX_KIND) == 0
    # The TupleConstruct DCEs once its only consumers (the TupleIndex
    # nodes) are gone.
    assert _node_count(m.graph, TUPLECONSTRUCT_KIND) == 0


@skipif_torch_lt_20
def test_fold_preserves_output():
    """Bitwise parity: rewriting the graph must not change behavior."""
    ref = _PairBuilderTwoIndex().eval()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    expected = ref(x, y)

    m = _scripted_inlined(_PairBuilderTwoIndex)
    fold_tuple_index_through_tuple_construct(m.graph)
    torch._C._jit_pass_dce(m.graph)

    got = m(x, y)
    assert torch.allclose(got, expected)


class _RuntimeIndex(nn.Module):
    def helper(self, x, y):
        return (x, y)

    def forward(self, x, y, idx: int):
        # The index is a runtime parameter, not a `prim::Constant`, so
        # the pass must leave the TupleIndex alone.
        pair = self.helper(x, y)
        return pair[idx]


@skipif_torch_lt_20
def test_pass_leaves_runtime_index_alone():
    m = _scripted_inlined(_RuntimeIndex)
    folded = fold_tuple_index_through_tuple_construct(m.graph)
    assert folded == 0
    assert _node_count(m.graph, TUPLEINDEX_KIND) == 1


class _NestedTupleConsumer(nn.Module):
    """Tuple indexing inside `prim::If` exercises the recursive walker.

    Each branch consumes the tuple by a different constant index so the
    pass has work to do inside both blocks.
    """

    def helper(self, x, y):
        return (x, y)

    def forward(self, x, y, take_first: bool):
        pair = self.helper(x, y)
        if take_first:  # noqa: SIM108 -- explicit if/else exercises prim::If
            out = pair[0]
        else:
            out = pair[1]
        return out + 1.0


@skipif_torch_lt_20
def test_fold_walks_into_nested_blocks():
    m = _scripted_inlined(_NestedTupleConsumer)
    assert _node_count(m.graph, TUPLEINDEX_KIND) == 2
    folded = fold_tuple_index_through_tuple_construct(m.graph)
    torch._C._jit_pass_dce(m.graph)
    assert folded == 2
    assert _node_count(m.graph, TUPLEINDEX_KIND) == 0


def test_pass_is_safe_when_no_tuple_index_present():
    class _Plain(nn.Module):
        def forward(self, x):
            return x + 1.0

    m = torch.jit.script(_Plain())
    folded = fold_tuple_index_through_tuple_construct(m.graph)
    assert folded == 0
