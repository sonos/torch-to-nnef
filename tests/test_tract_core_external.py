"""Tests case where input of graph are of specific dtypes."""

import os

import pytest
import torch
from torch import nn

from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))


class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


INPUT_AND_MODELS = [
    (torch.randint(0, 4, (2, 4), dtype=torch.int32), Mul()),
    (torch.rand((2, 4), dtype=torch.float64), Mul()),
    # (torch.rand((2, 4), dtype=torch.float16), Mul()), # tract strange error with npz format
]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_tricky_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
