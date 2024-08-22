import torch
from torch import nn

from tests.utils import check_model_io_test


class NLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._module = nn.LSTM(*args, **kwargs)

    def forward(self, x):
        y, _ = self._module.forward(x)
        return y


def atest_model_with_explicit_forward_call_to_catch_module():
    model = NLSTM(input_size=10, hidden_size=30)
    test_input = torch.rand(1, 30, 3)
    check_model_io_test(model=model, test_input=test_input)
