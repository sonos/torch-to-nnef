"""Tests for the `aten::lstm_cell` aten handler.

Both the aten handler and `LSTMCellExtractor` go through
`op.aten.rnn.emit_lstm_cell_via_fragment`, which materializes the cell
math as a single `lstm_cell` NNEF fragment call (see
`op/fragment/lstm_cell.nnef`). The on-disk shape is one node per cell
step, mirroring the existing `lstm` / `gru` / `rnn` fragment pattern.

t2n's default `torch.jit.trace`-based export decomposes `_VF.lstm_cell`
into `aten::linear + aten::unsafe_chunk + ...` rather than emitting
`aten::lstm_cell`, so the aten handler is exercised indirectly by
`tests/test_lstm_cell.py` (which uses `check_io` against tract through
the module path). Direct unit tests on the helper live below.
"""

import nnef_tools.model
import numpy as np
from torch import nn

from torch_to_nnef.op.aten import aten_ops_registry
from torch_to_nnef.op.aten.rnn import emit_lstm_cell_via_fragment


def test_lstm_cell_is_registered_in_aten_registry():
    """Sanity: the aten handler is reachable through the standard registry."""
    fn = aten_ops_registry.get("lstm_cell")
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
        mk("c", (batch, hidden)),
    )


def test_helper_emits_single_lstm_cell_fragment_call():
    """Fragment-based shape: one `lstm_cell` op, regardless of bias."""
    batch, in_size, hidden = 2, 8, 4
    g, name_to_tensor, nnef_dtype, input_ref, h_ref, c_ref = _build_graph(
        batch, in_size, hidden
    )

    cell = nn.LSTMCell(in_size, hidden)
    used = emit_lstm_cell_via_fragment(
        g,
        name_to_tensor,
        base="cell",
        nnef_dtype=nnef_dtype,
        batch_dim=batch,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_ref,
        c_prev_ref=c_ref,
        w_ih=cell.weight_ih.detach(),
        w_hh=cell.weight_hh.detach(),
        b_ih=cell.bias_ih.detach(),
        b_hh=cell.bias_hh.detach(),
        h_new_tv=None,
        c_new_tv=None,
    )

    assert used == ["lstm_cell"]
    op_kinds = [op.type for op in g.operations]
    compute_kinds = [k for k in op_kinds if k != "variable"]
    # The fragment internalizes everything: no flat matmul / sigmoid /
    # tanh / etc. should appear in the parent graph (only the weight /
    # bias `variable` declarations and the single fragment call).
    assert compute_kinds == ["lstm_cell"]


def test_fragment_call_carries_correct_slice_bounds():
    """Slice bounds describe the gate boundaries.

    `i_end / f_end / g_end / o_end` are H, 2H, 3H, 4H respectively.
    """
    batch, in_size, hidden = 1, 6, 7
    g, name_to_tensor, nnef_dtype, input_ref, h_ref, c_ref = _build_graph(
        batch, in_size, hidden
    )
    cell = nn.LSTMCell(in_size, hidden)
    emit_lstm_cell_via_fragment(
        g,
        name_to_tensor,
        base="cell",
        nnef_dtype=nnef_dtype,
        batch_dim=batch,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_ref,
        c_prev_ref=c_ref,
        w_ih=cell.weight_ih.detach(),
        w_hh=cell.weight_hh.detach(),
        b_ih=cell.bias_ih.detach(),
        b_hh=cell.bias_hh.detach(),
        h_new_tv=None,
        c_new_tv=None,
    )
    fragment_ops = [o for o in g.operations if o.type == "lstm_cell"]
    assert len(fragment_ops) == 1
    op = fragment_ops[0]
    assert op.attribs["i_end"] == hidden
    assert op.attribs["f_end"] == 2 * hidden
    assert op.attribs["g_end"] == 3 * hidden
    assert op.attribs["o_end"] == 4 * hidden


def test_helper_supports_bias_less_cell():
    """Bias-less variant.

    Caller passes None for b_ih / b_hh; helper materializes a zero bias
    internally so the fragment shape stays constant.
    """
    batch, in_size, hidden = 1, 4, 4
    g, name_to_tensor, nnef_dtype, input_ref, h_ref, c_ref = _build_graph(
        batch, in_size, hidden
    )

    cell = nn.LSTMCell(in_size, hidden, bias=False)
    used = emit_lstm_cell_via_fragment(
        g,
        name_to_tensor,
        base="cell",
        nnef_dtype=nnef_dtype,
        batch_dim=batch,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_ref,
        c_prev_ref=c_ref,
        w_ih=cell.weight_ih.detach(),
        w_hh=cell.weight_hh.detach(),
        b_ih=None,
        b_hh=None,
        h_new_tv=None,
        c_new_tv=None,
    )

    assert used == ["lstm_cell"]
    op_kinds = [op.type for op in g.operations]
    compute_kinds = [k for k in op_kinds if k != "variable"]
    assert compute_kinds == ["lstm_cell"]
