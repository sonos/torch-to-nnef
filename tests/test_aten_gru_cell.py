"""Tests for the `aten::gru_cell` aten handler.

`emit_gru_cell_via_fragment` materializes a single `gru_cell` NNEF
fragment call (`op/fragment/gru_cell.nnef`). t2n's default trace-based
export decomposes `nn.GRUCell.forward` into `aten::linear + chunk + ...`
rather than emitting `aten::gru_cell`; the aten handler kicks in for
scripted graphs or explicit `_VF.gru_cell(...)` calls. The unit tests
below pin the fragment-based shape.
"""

import nnef_tools.model
import numpy as np
from torch import nn

from torch_to_nnef.op.aten import aten_ops_registry
from torch_to_nnef.op.aten.rnn import emit_gru_cell_via_fragment


def test_gru_cell_is_registered_in_aten_registry():
    """Sanity: aten handler reachable through the standard registry."""
    fn = aten_ops_registry.get("gru_cell")
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


def test_helper_emits_single_gru_cell_fragment_call():
    """Fragment-based shape: one `gru_cell` op, regardless of bias."""
    batch, in_size, hidden = 2, 8, 4
    g, name_to_tensor, nnef_dtype, input_ref, h_ref = _build_graph(
        batch, in_size, hidden
    )

    cell = nn.GRUCell(in_size, hidden)
    used = emit_gru_cell_via_fragment(
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
    )

    assert used == ["gru_cell"]
    op_kinds = [op.type for op in g.operations]
    compute_kinds = [k for k in op_kinds if k != "variable"]
    assert compute_kinds == ["gru_cell"]


def test_fragment_call_carries_correct_slice_bounds():
    """Slice bounds describe the gate boundaries: H, 2H, 3H."""
    batch, in_size, hidden = 1, 6, 7
    g, name_to_tensor, nnef_dtype, input_ref, h_ref = _build_graph(
        batch, in_size, hidden
    )
    cell = nn.GRUCell(in_size, hidden)
    emit_gru_cell_via_fragment(
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
    )
    fragment_ops = [o for o in g.operations if o.type == "gru_cell"]
    assert len(fragment_ops) == 1
    op = fragment_ops[0]
    assert op.attribs["r_end"] == hidden
    assert op.attribs["z_end"] == 2 * hidden
    assert op.attribs["n_end"] == 3 * hidden


def test_helper_supports_bias_less_cell():
    """Bias-less variant: helper materializes zero biases internally."""
    batch, in_size, hidden = 1, 4, 4
    g, name_to_tensor, nnef_dtype, input_ref, h_ref = _build_graph(
        batch, in_size, hidden
    )

    cell = nn.GRUCell(in_size, hidden, bias=False)
    used = emit_gru_cell_via_fragment(
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
    )

    assert used == ["gru_cell"]
    op_kinds = [op.type for op in g.operations]
    compute_kinds = [k for k in op_kinds if k != "variable"]
    assert compute_kinds == ["gru_cell"]
