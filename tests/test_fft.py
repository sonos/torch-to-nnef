import pytest
import torch
from torch import nn

from tests.utils import check_model_io_test, id_tests
from torch_to_nnef.tract import tract_version_lower_than


class MyFFT(nn.Module):
    def forward(self, x):
        x = torch.fft.fft(x)
        x = torch.view_as_real(x)
        return x


INPUT_AND_MODELS = []

if not tract_version_lower_than("0.19.0"):
    INPUT_AND_MODELS += [
        (torch.FloatTensor([[1, 2, 4, 5], [4, 3, 2, 9]]), MyFFT())
    ]


@pytest.mark.parametrize(
    "test_input,model", INPUT_AND_MODELS, ids=id_tests(INPUT_AND_MODELS)
)
def test_complex_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
