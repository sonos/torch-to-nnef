import torch
from torch import nn

from .utils import check_model_io_test


class LSTMWrapper(nn.Module):
    def __init__(self, nin, nout, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(nin, nout, num_layers=n_layers)

    def forward(self, x, c, h):
        states = (h, c)
        y, (hnew, cnew) = self.lstm(x, states)
        return y, hnew, cnew


def test_manage_lstm_states():
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = LSTMWrapper(inputs, outputs)
    x = torch.rand(seqlen, batch, inputs)

    check_model_io_test(
        model=module,
        test_input=(
            x,
            torch.rand(1, batch, outputs),
            torch.rand(1, batch, outputs),
        ),
        input_names=["input", "input_state_1", "input_state_2"],
        output_names=["output", "output_state_1", "output_state_2"],
        dynamic_axes={
            "input": {0: "S", 1: "B"},
            "input_state_1": {1: "B"},
            "input_state_2": {1: "B"},
            "output": {0: "S", 1: "B"},
            "output_state_1": {1: "B"},
            "output_state_2": {1: "B"},
        },
    )


def test_manage_lstm_states_multi_layers():
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = LSTMWrapper(inputs, outputs, 3)
    x = torch.rand(seqlen, batch, inputs)

    check_model_io_test(
        model=module,
        test_input=(
            x,
            torch.rand(3, batch, outputs),
            torch.rand(3, batch, outputs),
        ),
        input_names=["input", "input_state_1", "input_state_2"],
        output_names=["output", "output_state_1", "output_state_2"],
        dynamic_axes={
            "input": {0: "S", 1: "B"},
            "input_state_1": {1: "B"},
            "input_state_2": {1: "B"},
            "output": {0: "S", 1: "B"},
            "output_state_1": {1: "B"},
            "output_state_2": {1: "B"},
        },
    )


def _test_mono_states_rnn(cls):
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = cls(inputs, outputs)
    x = torch.rand(seqlen, batch, inputs)

    check_model_io_test(
        model=module,
        test_input=(x, torch.rand(1, batch, outputs)),
        input_names=["input", "input_state_1"],
        output_names=["output", "output_state_1"],
        dynamic_axes={
            "input": {0: "S", 1: "B"},
            "input_state_1": {1: "B"},
            "output": {0: "S", 1: "B"},
            "output_state_1": {1: "B"},
        },
    )


def test_manage_gru_states():
    _test_mono_states_rnn(nn.GRU)


def test_manage_rnn_states():
    _test_mono_states_rnn(nn.RNN)
