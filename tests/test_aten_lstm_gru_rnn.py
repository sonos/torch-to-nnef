"""Tests for the aten-level RNN handlers.

After Phase 6 the canonical NNEF emission lives in `op/aten/rnn.py`. The
module-level extractors are thin shims that delegate to the same free
functions, so existing `test_rnn_advanced.py` byte-equivalence tests
also exercise this path. Here we verify:

- The aten registry exposes `lstm`, `gru`, `rnn_tanh`, `rnn_relu`.
- The adapter classes correctly map an aten op's flat
  `params: Tensor[]` argument back to the named-attribute interface
  the per-variant `tensor_params` reads from.

`torch.jit.trace` decomposes the RNN modules into individual aten ops
that are not the `aten::lstm` / `gru` / `rnn_*` we register, so the aten
handlers fire only on scripted-then-inlined JIT artifacts. End-to-end
parity tests through that path are out of scope here -- the existing
extractor tests already prove byte-identical NNEF emission since both
paths share `emit_rnn_via_fragment`.
"""

import torch
from torch import nn

from torch_to_nnef.op.aten import aten_ops_registry
from torch_to_nnef.op.aten.rnn import (
    _GRUAtenAdapter,
    _LSTMAtenAdapter,
    _RNNAtenAdapter,
)


def test_aten_rnn_handlers_registered():
    for name in ("lstm", "gru", "rnn_tanh", "rnn_relu"):
        fn = aten_ops_registry.get(name)
        assert callable(fn), name


def test_lstm_adapter_unidirectional_with_biases():
    """Layout: per layer-direction `[w_ih, w_hh, b_ih, b_hh]`.

    Single-direction, 2-layer with biases: 8 tensors total.
    """
    in_size, hidden, num_layers = 5, 7, 2
    cell = nn.LSTM(in_size, hidden, num_layers=num_layers, bias=True)
    flat = []
    for layer in range(num_layers):
        flat.extend(
            [
                getattr(cell, f"weight_ih_l{layer}").detach(),
                getattr(cell, f"weight_hh_l{layer}").detach(),
                getattr(cell, f"bias_ih_l{layer}").detach(),
                getattr(cell, f"bias_hh_l{layer}").detach(),
            ]
        )
    adapter = _LSTMAtenAdapter(
        params_tensors=flat,
        has_biases=True,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=False,
        base_name="t",
    )
    assert adapter.hidden_size == hidden
    assert adapter.num_layers == num_layers
    assert adapter.bidirectional is False
    for layer in range(num_layers):
        assert torch.equal(
            getattr(adapter, f"weight_ih_l{layer}"),
            getattr(cell, f"weight_ih_l{layer}"),
        )
        assert torch.equal(
            getattr(adapter, f"bias_hh_l{layer}"),
            getattr(cell, f"bias_hh_l{layer}"),
        )


def test_lstm_adapter_bidirectional_no_biases():
    """No biases: 2 tensors per direction. Bidirectional doubles per layer."""
    in_size, hidden, num_layers = 4, 6, 1
    cell = nn.LSTM(
        in_size, hidden, num_layers=num_layers, bias=False, bidirectional=True
    )
    flat = [
        cell.weight_ih_l0.detach(),
        cell.weight_hh_l0.detach(),
        cell.weight_ih_l0_reverse.detach(),
        cell.weight_hh_l0_reverse.detach(),
    ]
    adapter = _LSTMAtenAdapter(
        params_tensors=flat,
        has_biases=False,
        num_layers=num_layers,
        bidirectional=True,
        batch_first=False,
        base_name="t",
    )
    assert adapter.hidden_size == hidden
    assert adapter.bidirectional is True
    assert torch.equal(adapter.weight_ih_l0, cell.weight_ih_l0)
    assert torch.equal(adapter.weight_hh_l0_reverse, cell.weight_hh_l0_reverse)
    assert not hasattr(adapter, "bias_ih_l0")


def test_gru_adapter_recovers_hidden_size():
    """GRU `weight_ih_l0` is `(3*H, I)` so adapter divides by 3."""
    in_size, hidden = 4, 9
    cell = nn.GRU(in_size, hidden, num_layers=1, bias=True)
    flat = [
        cell.weight_ih_l0.detach(),
        cell.weight_hh_l0.detach(),
        cell.bias_ih_l0.detach(),
        cell.bias_hh_l0.detach(),
    ]
    adapter = _GRUAtenAdapter(
        params_tensors=flat,
        has_biases=True,
        num_layers=1,
        bidirectional=False,
        batch_first=False,
        base_name="t",
    )
    assert adapter.hidden_size == hidden
    assert adapter.weight_ih_l0.shape == (3 * hidden, in_size)


def test_rnn_adapter_recovers_hidden_size_and_nonlinearity():
    """RNN `weight_ih_l0` is `(H, I)`."""
    in_size, hidden = 4, 11
    cell = nn.RNN(in_size, hidden, num_layers=1, bias=True, nonlinearity="relu")
    flat = [
        cell.weight_ih_l0.detach(),
        cell.weight_hh_l0.detach(),
        cell.bias_ih_l0.detach(),
        cell.bias_hh_l0.detach(),
    ]
    adapter = _RNNAtenAdapter(
        params_tensors=flat,
        has_biases=True,
        num_layers=1,
        bidirectional=False,
        batch_first=False,
        base_name="t",
        nonlinearity="relu",
    )
    assert adapter.hidden_size == hidden
    assert adapter.nonlinearity == "relu"
    assert adapter.weight_ih_l0.shape == (hidden, in_size)
