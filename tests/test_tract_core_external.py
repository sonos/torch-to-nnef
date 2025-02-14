"""Tests case where input of graph are of specific dtypes."""

import os

import pytest
import torch
from torch import nn

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))


class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)

test_suite.add(torch.randint(0, 4, (2, 4), dtype=torch.int32), Mul())
test_suite.add(torch.rand((2, 4), dtype=torch.float64), Mul())
# (torch.rand((2, 4), dtype=torch.float16), Mul()), # tract does not support npz f16 format


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_externals_export(id, test_input, model, inference_target):
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
