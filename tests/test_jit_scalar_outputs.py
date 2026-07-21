"""Tests for scalar-typed op outputs in the t2n IR.

`TensorVariable.parse` was extended to accept `IntType` (already supported),
`FloatType`, and `BoolType` SSA values: inlined / constfolded JIT graphs
surface these as op outputs (e.g. `aten::Int`, `aten::Bool`, `aten::div(int,
int) -> float`) and t2n's parser must not refuse them.
"""

import torch
from torch import nn

from torch_to_nnef.torch_graph.ir_data import TensorVariable
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_BOOL,
    ATEN_DIV,
    ATEN_LEN,
    ATEN_SCALARIMPLICIT,
    NUMBERTYPE_KIND,
)


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
        if n.kind() == ATEN_DIV
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
        if n.kind() == ATEN_BOOL
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
        if n.kind() == ATEN_LEN
        for out in n.outputs()
        if out.type().annotation_str == "int"
    ]
    assert len_outputs

    parsed = TensorVariable.parse(len_outputs[0])
    assert parsed.dtype == torch.int64
    assert parsed.shape == [1]


class _ItemScalar(nn.Module):
    def forward(self, x):
        # `Tensor.item()` is typed `aten::item(...) -> Scalar`, i.e. a bare
        # NumberType SSA value with no `aten::ScalarImplicit` wrapper, the same
        # shape Qwen2.5-VL's vision tower produces when a 0-d tensor
        # (grid_thw.max()) feeds a submodule `forward(seqlen)` -> arange.
        return x + x.max().item()


def _bare_number_value():
    m = torch.jit.script(_ItemScalar())
    for node in _walk(m.graph):
        if node.kind() == ATEN_SCALARIMPLICIT:
            continue
        for out in node.outputs():
            if out.type().kind() == NUMBERTYPE_KIND:
                return out
    return None


def test_bare_number_scalar_value_parses():
    """A bare `Scalar`/NumberType SSA value (no ScalarImplicit) must parse.

    Regression for the Qwen2.5-VL vision export crash: t2n used to raise
    `T2NErrorNotImplemented` here. It now parses as an int64 scalar; the
    recursive-input path fills the concrete dtype/shape from the traced arg.
    """
    value = _bare_number_value()
    assert value is not None, "test setup expected a bare NumberType SSA value"

    parsed = TensorVariable.parse(value)
    assert parsed.dtype == torch.int64
    assert parsed.shape == [1]
