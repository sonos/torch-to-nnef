"""Tests for the `aten::rnn_{tanh,relu}_cell` aten handlers.

`emit_rnn_cell_via_fragment` materializes a single `rnn_tanh_cell` or
`rnn_relu_cell` NNEF fragment call (`op/fragment/rnn_tanh_cell.nnef` /
`op/fragment/rnn_relu_cell.nnef`). Same trace caveat as `gru_cell`: the
aten path only fires for scripted graphs / explicit `_VF.rnn_*_cell`
calls, while trace decomposition routes through `aten::linear + ...`.
"""

import nnef_tools.model
import numpy as np
import pytest
from torch import nn

from torch_to_nnef.op.aten import aten_ops_registry
from torch_to_nnef.op.aten.rnn import emit_rnn_cell_via_fragment


@pytest.mark.parametrize(
    ("aten_name", "nonlinearity", "fragment"),
    [
        ("rnn_tanh_cell", "tanh", "rnn_tanh_cell"),
        ("rnn_relu_cell", "relu", "rnn_relu_cell"),
    ],
)
def test_rnn_cell_is_registered_in_aten_registry(
    aten_name, nonlinearity, fragment
):
    """Sanity: both aten handlers reachable through the standard registry."""
    del nonlinearity, fragment
    fn = aten_ops_registry.get(aten_name)
    assert callable(fn)


def _build_graph(batch: int, in_size: int, hidden: int):
    g = nnef_tools.model.Graph(name="test")
    name_to_tensor = {}
    nnef_dtype = np.float32

    def mk(name, shape):
        t = nnef_tools.model.Tensor(g, name, dtype=nnef_dtype, shape=shape)
        name_to_tensor[name] = t
        return t

    return (
        g,
        name_to_tensor,
        nnef_dtype,
        mk("x", (batch, in_size)),
        mk("h", (batch, hidden)),
    )


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_helper_emits_single_rnn_cell_fragment_call(nonlinearity):
    """Fragment-based shape: one `rnn_{tanh,relu}_cell` op."""
    batch, in_size, hidden = 2, 8, 4
    g, name_to_tensor, nnef_dtype, input_ref, h_ref = _build_graph(
        batch, in_size, hidden
    )

    cell = nn.RNNCell(in_size, hidden, nonlinearity=nonlinearity)
    used = emit_rnn_cell_via_fragment(
        g,
        name_to_tensor,
        base="cell",
        nnef_dtype=nnef_dtype,
        batch_dim=batch,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_ref,
        w_ih=cell.weight_ih.detach(),
        w_hh=cell.weight_hh.detach(),
        b_ih=cell.bias_ih.detach(),
        b_hh=cell.bias_hh.detach(),
        h_new_tv=None,
        nonlinearity=nonlinearity,
    )

    fragment = f"rnn_{nonlinearity}_cell"
    assert used == [fragment]
    op_kinds = [op.type for op in g.operations]
    compute_kinds = [k for k in op_kinds if k != "variable"]
    assert compute_kinds == [fragment]


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_helper_supports_bias_less_cell(nonlinearity):
    """Bias-less variant: helper synthesizes a zero bias internally."""
    batch, in_size, hidden = 1, 4, 4
    g, name_to_tensor, nnef_dtype, input_ref, h_ref = _build_graph(
        batch, in_size, hidden
    )

    cell = nn.RNNCell(in_size, hidden, bias=False, nonlinearity=nonlinearity)
    used = emit_rnn_cell_via_fragment(
        g,
        name_to_tensor,
        base="cell",
        nnef_dtype=nnef_dtype,
        batch_dim=batch,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_ref,
        w_ih=cell.weight_ih.detach(),
        w_hh=cell.weight_hh.detach(),
        b_ih=None,
        b_hh=None,
        h_new_tv=None,
        nonlinearity=nonlinearity,
    )

    fragment = f"rnn_{nonlinearity}_cell"
    assert used == [fragment]
    op_kinds = [op.type for op in g.operations]
    compute_kinds = [k for k in op_kinds if k != "variable"]
    assert compute_kinds == [fragment]
