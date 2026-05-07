"""Tests for scalar-typed op outputs in the t2n IR.

`TensorVariable.parse` was extended to accept `IntType` (already supported),
`FloatType`, and `BoolType` SSA values: inlined / constfolded JIT graphs
surface these as op outputs (e.g. `aten::Int`, `aten::Bool`, `aten::div(int,
int) -> float`) and t2n's parser must not refuse them.
"""

import torch
from torch import nn

from torch_to_nnef.torch_graph.ir_data import TensorVariable


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


class _IntDiv(nn.Module):
    def forward(self, x):
        # `len(x) / 2` produces a `FloatType` SSA value via aten::div.
        return x + (len(x) / 2.0)


def test_float_scalar_output_parses():
    m = torch.jit.script(_IntDiv())
    div_outputs = [
        out
        for n in _walk(m.graph)
        if n.kind() == "aten::div"
        for out in n.outputs()
        if out.type().annotation_str == "float"
    ]
    assert div_outputs, "test setup expected an aten::div with float output"

    parsed = TensorVariable.parse(div_outputs[0])
    assert parsed.dtype == torch.float32
    assert parsed.shape == [1]


class _BoolCast(nn.Module):
    def forward(self, x, k: int):
        return x + (1.0 if bool(k) else 0.0)


def test_bool_scalar_output_parses():
    m = torch.jit.script(_BoolCast())
    bool_outputs = [
        out
        for n in _walk(m.graph)
        if n.kind() == "aten::Bool"
        for out in n.outputs()
        if out.type().annotation_str == "bool"
    ]
    assert bool_outputs, "test setup expected an aten::Bool with bool output"

    parsed = TensorVariable.parse(bool_outputs[0])
    assert parsed.dtype == torch.bool
    assert parsed.shape == [1]


def test_int_scalar_output_still_parses():
    """Sanity: the IntType path that already worked must still pass."""

    class _LenQ(nn.Module):
        def forward(self, x):
            return x[: len(x)]

    m = torch.jit.script(_LenQ())
    len_outputs = [
        out
        for n in _walk(m.graph)
        if n.kind() == "aten::len"
        for out in n.outputs()
        if out.type().annotation_str == "int"
    ]
    assert len_outputs

    parsed = TensorVariable.parse(len_outputs[0])
    assert parsed.dtype == torch.int64
    assert parsed.shape == [1]
