from copy import deepcopy

import pytest
import torch
from torch import nn

from torch_to_nnef.utils import torch_version

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test

skipif_sub_torch2 = pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason="torch version need to be >= 1.12.0 to use OffloadedTensor",
)


class LSTMWrapper(nn.Module):
    def __init__(self, nin, nout, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(nin, nout, num_layers=n_layers)

    def forward(self, x, c, h):
        states = (h, c)
        y, (hnew, cnew) = self.lstm(x, states)
        return y, hnew, cnew


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_manage_lstm_states(inference_target):
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = LSTMWrapper(inputs, outputs)
    x = torch.rand(seqlen, batch, inputs)

    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {
        "input": {0: "S", 1: "B"},
        "input_state_1": {1: "B"},
        "input_state_2": {1: "B"},
    }
    check_model_io_test(
        model=module,
        test_input=(
            x,
            torch.rand(1, batch, outputs),
            torch.rand(1, batch, outputs),
        ),
        input_names=["input", "input_state_1", "input_state_2"],
        output_names=["output", "output_state_1", "output_state_2"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_manage_lstm_states_multi_layers(inference_target):
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = LSTMWrapper(inputs, outputs, 3)
    x = torch.rand(seqlen, batch, inputs)

    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {
        "input": {0: "S", 1: "B"},
        "input_state_1": {1: "B"},
        "input_state_2": {1: "B"},
    }

    check_model_io_test(
        model=module,
        test_input=(
            x,
            torch.rand(3, batch, outputs),
            torch.rand(3, batch, outputs),
        ),
        input_names=["input", "input_state_1", "input_state_2"],
        output_names=["output", "output_state_1", "output_state_2"],
        inference_target=inference_target,
    )


def _test_mono_states_rnn(cls, inference_target):
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = cls(inputs, outputs)
    x = torch.rand(seqlen, batch, inputs)

    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {
        "input": {0: "S", 1: "B"},
        "input_state_1": {1: "B"},
    }

    check_model_io_test(
        model=module,
        test_input=(x, torch.rand(1, batch, outputs)),
        input_names=["input", "input_state_1"],
        output_names=["output", "output_state_1"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_manage_gru_states(inference_target):
    _test_mono_states_rnn(nn.GRU, inference_target)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_manage_rnn_states(inference_target):
    _test_mono_states_rnn(nn.RNN, inference_target)


class PickHt(nn.Module):
    def __init__(self, nin, nout, n_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            nin,
            nout,
            batch_first=batch_first,
            num_layers=n_layers,
        )

    def forward(self, x):
        _, (ht, _) = self.lstm(x)
        return ht.squeeze(0)


@pytest.mark.parametrize(
    "inference_target", TRACT_INFERENCES_TO_TESTS_APPROX[:1]
)
def test_pick_ht(inference_target):
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = PickHt(inputs, outputs, 1)

    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {
        "i1": {1: "S", 0: "B"},
    }

    check_model_io_test(
        model=module,
        test_input=(torch.rand(batch, seqlen, inputs),),
        input_names=["i1"],
        output_names=["output"],
        inference_target=inference_target,
    )


class EncoderJoin(nn.Module):
    def __init__(self, nin, nout, n_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            nin,
            nout,
            batch_first=batch_first,
            num_layers=n_layers,
        )

    def forward(self, x, y):
        _, (x1, _) = self.lstm(x)
        _, (x2, _) = self.lstm(y)
        return torch.cat([x1[-1], x2[-1]], dim=1)


@skipif_sub_torch2
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_reused_lstm_on_2_inputs(inference_target):
    seqlen = 10
    batch = 16
    inputs = 2
    outputs = 4

    module = EncoderJoin(inputs, outputs, 2)

    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {
        "i1": {1: "S", 0: "B"},
        "i2": {1: "C", 0: "B"},
    }

    check_model_io_test(
        model=module,
        test_input=(
            torch.rand(batch, seqlen, inputs),
            torch.rand(batch, seqlen, inputs),
        ),
        input_names=["i1", "i2"],
        output_names=["output"],
        inference_target=inference_target,
    )
