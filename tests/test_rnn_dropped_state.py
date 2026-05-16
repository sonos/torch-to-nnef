"""Exports where the caller discards the RNN hidden-state output.

`nn.GRU` and `nn.LSTM` both return a tuple. Real-world models often use only
the sequence output and drop the hidden-state(s) at the call site
(``out, _ = gru(x)``). Under `torch.jit.trace` (PyTorch >= 2.0) the submodule's
IR graph still exposes the full output tuple, while the upstream
`prim::CallMethod` declares only the bound output -- a `len(gouts) > len(
provided_outputs)` mismatch in
`torch_to_nnef.op.custom_extractors.base._extract_outputs`. The extractor
must synthesise placeholders for the un-bound slots instead of raising.

These tests exercise the dropped-state path end-to-end via tract `check_io`.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class GRUDropState(nn.Module):
    """`nn.GRU` wrapper that discards `h_n` (the common production shape)."""

    def __init__(self, nin: int, nout: int, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(nin, nout, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class LSTMDropState(nn.Module):
    """`nn.LSTM` wrapper that discards `(h_n, c_n)`."""

    def __init__(self, nin: int, nout: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(nin, nout, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class GRUDropStateBatchFirst(nn.Module):
    """Like `GRUDropState`, but `batch_first=True` (DFN3-style)."""

    def __init__(self, nin: int, nout: int) -> None:
        super().__init__()
        self.gru = nn.GRU(nin, nout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class GRUImplicitInitThenReshape(nn.Module):
    """GRU with implicit (zero) init state, then a static `view`.

    This is the DFN3 shape: the GRU runs without an explicit `h_0`, so
    `_translate_state_variable_load_and_prep` materialises a default
    init and tiles it to the input's batch. A downstream `.view(-1)`
    (or any reshape with a static target) is the first thing that
    breaks if the batch dim is emitted as a symbolic
    (`tract_core_shape_of`) value: tract's constraint solver sees the
    flattened size as a symbolic expression and rejects the reshape
    against the baked static target.
    """

    def __init__(self, nin: int, nout: int) -> None:
        super().__init__()
        self.gru = nn.GRU(nin, nout, batch_first=True)
        self.nout = nout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # implicit zero init
        # Take the last timestep, then reshape to a static (B, nout).
        last = out[:, -1, :]
        return last.view(-1, self.nout)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_gru_drops_hidden_state(inference_target):
    """`nn.GRU` -> tuple is destructured, h_n is discarded."""
    seqlen, batch, nin, nout = 7, 3, 5, 4
    module = GRUDropState(nin, nout)
    x = torch.rand(seqlen, batch, nin)
    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {"input": {0: "S", 1: "B"}}
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_lstm_drops_hidden_state(inference_target):
    """`nn.LSTM` -> tuple is destructured, `(h_n, c_n)` is discarded."""
    seqlen, batch, nin, nout = 7, 3, 5, 4
    module = LSTMDropState(nin, nout)
    x = torch.rand(seqlen, batch, nin)
    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {"input": {0: "S", 1: "B"}}
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_gru_drops_state_batch_first(inference_target):
    """`nn.GRU(batch_first=True)`, drop h_n. Mirrors the DFN3 GRU shape."""
    batch, seqlen, nin, nout = 3, 11, 6, 4
    module = GRUDropStateBatchFirst(nin, nout)
    x = torch.rand(batch, seqlen, nin)
    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {"input": {0: "B", 1: "S"}}
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_gru_drops_state_multilayer(inference_target):
    """Multi-layer GRU with dropped state.

    Verifies the layer-stacking path survives the placeholder synthesis
    (each layer still has its own internal output tuple).
    """
    seqlen, batch, nin, nout = 5, 2, 4, 3
    module = GRUDropState(nin, nout, num_layers=2)
    x = torch.rand(seqlen, batch, nin)
    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {"input": {0: "S", 1: "B"}}
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_gru_implicit_init_then_static_reshape(inference_target):
    """Static-axes GRU with implicit init + downstream static reshape.

    Regression for the DFN3 export failure: when the user does not pass
    an explicit `h_0`, the GRU/LSTM/RNN extractor synthesises one and
    `_translate_state_variable_load_and_prep` tiles it to the input's
    batch. The original implementation emitted a
    `tract_core_shape_of -> slice -> squeeze -> tile` chain
    unconditionally, which made tract treat the resulting batch dim as
    symbolic even when no dynamic axes were requested. The symbolic dim
    then propagated through the GRU output, the slice, and into the
    `.view(-1, nout)` reshape, which has a static target -- tract's
    constraint solver rejected the reshape because the symbolic
    flattened size could not be unified with the baked target. The fix
    is to use a static `tile` repeats vector (the input's actual batch
    size) when `inference_target.has_dynamic_axes` is False.

    Note: NO `dynamic_axes` set on the inference target. Triggering the
    bug requires the static path.
    """
    batch, seqlen, nin, nout = 2, 5, 4, 3
    module = GRUImplicitInitThenReshape(nin, nout)
    x = torch.rand(batch, seqlen, nin)
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=deepcopy(inference_target),
    )
