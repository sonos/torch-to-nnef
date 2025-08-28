"""Tests simple accumulator option."""

import os
from copy import deepcopy

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

tract_latest = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
tract_latest.force_linear_accumulation_in_f32 = True

test_suite = TestSuiteInferenceExactnessBuilder([tract_latest])

if tract_latest.version >= "0.21.11":
    mod = nn.Linear(3, 4)
    mod = mod.half()
    inp = torch.arange(6).reshape(1, 2, 3).half()
    try:
        with torch.inference_mode():
            mod(inp)
        test_suite.add(inp, mod)
    except RuntimeError as exp:
        print("failed to add the test because torch:", exp)


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_linear_accumulate_f32_export(id, test_input, model, inference_target):
    """Test simple aten PyTorch core."""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
