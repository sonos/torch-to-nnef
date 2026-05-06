"""Tests for `replace_size_calls_with_constants`.

The pass uses forward reach analysis: a size source is folded to a
constant only when every reach path terminates in a control-flow sink
(`prim::If` condition, `prim::Loop` trip count, `prim::RaiseException`)
without ever crossing a node that produces a tensor-typed output. Sources
whose value flows into tensor production (via `aten::reshape`, `aten::view`,
`aten::expand`, `aten::zeros`, ...) are deliberately left alone so the
standard `aten::size` handler in `op/aten/other.py` can route them through
`tract_core_shape_of` under `inference_target.has_dynamic_axes`.

The test suite is organized as principled pairs: "control-flow-only ->
fold" and "tensor-shape sink -> refuse".
"""

import torch
from torch import nn

from torch_to_nnef.torch_graph import replace_size_calls_with_constants


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


def _node_count(g, kind):
    return sum(1 for n in _walk(g) if n.kind() == kind)


# --- Control-flow-only sources: SHOULD fold ---------------------------------


class _DimCheckIf(nn.Module):
    def forward(self, x):
        # aten::dim feeds only the prim::If condition: control-flow-only.
        if x.dim() != 2:
            x = x.unsqueeze(0)
        return x + 1.0


def test_dim_in_if_condition_folds():
    m = torch.jit.script(_DimCheckIf())
    x = torch.randn(2, 4)
    assert _node_count(m.graph, "aten::dim") == 1
    assert _node_count(m.graph, "prim::If") == 1

    folded = replace_size_calls_with_constants(m.graph, [m, x])
    torch._C._jit_pass_constant_propagation(m.graph)
    torch._C._jit_pass_dce(m.graph)

    assert folded == 1
    assert _node_count(m.graph, "aten::dim") == 0
    assert _node_count(m.graph, "prim::If") == 0


class _LenInIfCondition(nn.Module):
    def forward(self, state, x):
        # aten::len(state) feeds aten::Bool feeds prim::If condition:
        # all primitive-typed propagators ending in a control-flow sink.
        if bool(len(state)):
            return x + 1.0
        return x


def test_len_in_if_condition_folds():
    m = torch.jit.script(_LenInIfCondition())
    state = torch.zeros(2, 1, 5)
    x = torch.randn(1, 5)
    assert _node_count(m.graph, "aten::len") == 1

    folded = replace_size_calls_with_constants(m.graph, [m, state, x])
    torch._C._jit_pass_constant_propagation(m.graph)
    torch._C._jit_pass_dce(m.graph)

    assert folded == 1
    assert _node_count(m.graph, "aten::len") == 0


# --- Tensor-shape sinks: must NOT fold (dynamic-axes safety) ---------------


class _SizeFedToTensorOp(nn.Module):
    def forward(self, x):
        # aten::size(x, 0) feeds aten::Float feeds aten::add producing a
        # Tensor. Reach analysis must refuse so the size remains symbolic
        # for the standard handler under dynamic_axes.
        b = x.size(0)
        return x + float(b)


def test_size_consumed_by_tensor_op_is_not_folded():
    m = torch.jit.script(_SizeFedToTensorOp())
    x = torch.randn(3, 5)
    assert _node_count(m.graph, "aten::size") == 1

    folded = replace_size_calls_with_constants(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    assert folded == 0
    assert _node_count(m.graph, "aten::size") == 1


class _LenAsSliceBound(nn.Module):
    def forward(self, x):
        # len(x) feeds the slice end argument: aten::slice produces a Tensor.
        n = len(x)
        return x[:n]


def test_len_as_slice_bound_is_not_folded():
    m = torch.jit.script(_LenAsSliceBound())
    x = torch.randn(7, 4)
    assert _node_count(m.graph, "aten::len") == 1

    folded = replace_size_calls_with_constants(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    assert folded == 0
    assert _node_count(m.graph, "aten::len") == 1


class _NumelAsScalar(nn.Module):
    def forward(self, x):
        # numel feeds a tensor add: tensor sink.
        return x + float(x.numel())


def test_numel_as_scalar_is_not_folded():
    m = torch.jit.script(_NumelAsScalar())
    x = torch.randn(2, 3, 4)
    assert _node_count(m.graph, "aten::numel") == 1

    folded = replace_size_calls_with_constants(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    assert folded == 0
    assert _node_count(m.graph, "aten::numel") == 1


class _SizeInReshape(nn.Module):
    def forward(self, x):
        # The classic dynamic-axes case: size(0) appears in a reshape's
        # target shape. Folding it would bake the example batch dim into
        # the NNEF graph and break a `dynamic_axes={"x": {0: "B"}}` export.
        b = x.size(0)
        return x.reshape(b, -1)


def test_size_in_reshape_target_is_not_folded():
    m = torch.jit.script(_SizeInReshape())
    x = torch.randn(3, 5)
    assert _node_count(m.graph, "aten::size") == 1

    folded = replace_size_calls_with_constants(m.graph, [m, x])
    torch._C._jit_pass_dce(m.graph)

    assert folded == 0
    assert _node_count(m.graph, "aten::size") == 1


# --- No-op when nothing to fold --------------------------------------------


def test_pass_is_safe_when_no_size_ops_present():
    class _Plain(nn.Module):
        def forward(self, x):
            return x * 2

    m = torch.jit.script(_Plain())
    x = torch.randn(2, 4)
    folded = replace_size_calls_with_constants(m.graph, [m, x])
    assert folded == 0
