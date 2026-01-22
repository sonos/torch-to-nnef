import torch
from torch import nn

from torch_to_nnef import TractNNEF
from tests.utils import check_model_io_test


class LSTMWithState(nn.Module):
    """
    LSTM wrapper that accepts hidden state as a list or tuple.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, state=None):
        output, new_state = self.lstm(x, state)
        return output, new_state


class LSTMStatePostOpModel(nn.Module):
    """
    Non-regression model:
    - LSTM state passed as list [h, c]
    - LSTM output consumed by another module
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = LSTMWithState(input_size, hidden_size)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        state = [h, c]
        output, (h_new, c_new) = self.rnn(x, state=state)
        output = self.proj(output)
        return output, h_new, c_new


def test_lstm_state_list_with_post_op_non_regression():
    """
    Non-regression test for LSTM state handling during tracing.

    Ensures that passing LSTM state as a list [h, c] works correctly
    even when the LSTM output is consumed by another module.
    """
    batch_size = 4
    seq_len = 1
    input_size = 6
    hidden_size = 8
    output_size = 10
    num_layers = 1

    model = LSTMStatePostOpModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    x = torch.randn(batch_size, seq_len, input_size)
    h = torch.randn(num_layers, batch_size, hidden_size)
    c = torch.randn(num_layers, batch_size, hidden_size)

    target = TractNNEF(
        version="0.21.13",
        check_io=False,
        dynamic_axes={},
    )

    check_model_io_test(
        model=model,
        test_input=(x, h, c),
        input_names=("x", "h", "c"),
        output_names=("y", "h_new", "c_new"),
        inference_target=target,
    )
