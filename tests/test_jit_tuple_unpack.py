"""Tests for `fold_tuple_unpack_through_tuple_construct`.

Mirrors the TupleIndex fold tests: PyTorch's scripter constant-folds
`(a, b) = pair` directly when the `prim::TupleConstruct` is in the
same scope as the consumption, so the test models route the tuple
build through a separate scripted method that gets inlined via
`_jit_pass_inline`. After inlining the pattern becomes
`prim::TupleConstruct -> prim::TupleUnpack`, which the pass under
test rewrites.
"""

import pytest
import torch
from torch import nn

from torch_to_nnef.torch_graph import fold_tuple_unpack_through_tuple_construct
from torch_to_nnef.torch_graph.torch_const import (
    TUPLECONSTRUCT_KIND,
    TUPLEUNPACK_KIND,
)
from torch_to_nnef.utils import torch_version

skipif_torch_lt_20 = pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason=(
        "torch < 2.0's `_jit_pass_inline` doesn't expose the "
        "`TupleConstruct -> TupleUnpack` pattern the fold targets"
    ),
)


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


def _node_count(g, kind):
    return sum(1 for n in _walk(g) if n.kind() == kind)


class _PairBuilderUnpacked(nn.Module):
    def helper(self, x, y):
        return (x, y)

    def forward(self, x, y):
        a, b = self.helper(x, y)
        return a + b


def _scripted_inlined(cls):
    m = torch.jit.script(cls())
    torch._C._jit_pass_inline(m.graph)
    return m


@skipif_torch_lt_20
def test_fold_collapses_tupleunpack_into_construct_inputs():
    m = _scripted_inlined(_PairBuilderUnpacked)
    assert _node_count(m.graph, TUPLEUNPACK_KIND) == 1
    assert _node_count(m.graph, TUPLECONSTRUCT_KIND) == 1

    folded = fold_tuple_unpack_through_tuple_construct(m.graph)
    torch._C._jit_pass_dce(m.graph)

    assert folded == 1
    assert _node_count(m.graph, TUPLEUNPACK_KIND) == 0
    # The TupleConstruct DCEs once its only consumer (the TupleUnpack)
    # is gone.
    assert _node_count(m.graph, TUPLECONSTRUCT_KIND) == 0


@skipif_torch_lt_20
def test_fold_preserves_output():
    """Bitwise parity: rewriting the graph must not change behavior."""
    ref = _PairBuilderUnpacked().eval()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    expected = ref(x, y)

    m = _scripted_inlined(_PairBuilderUnpacked)
    fold_tuple_unpack_through_tuple_construct(m.graph)
    torch._C._jit_pass_dce(m.graph)

    got = m(x, y)
    assert torch.allclose(got, expected)


def test_pass_is_safe_when_no_tuple_unpack_present():
    class _Plain(nn.Module):
        def forward(self, x):
            return x + 1.0

    m = torch.jit.script(_Plain())
    folded = fold_tuple_unpack_through_tuple_construct(m.graph)
    assert folded == 0
