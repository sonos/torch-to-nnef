"""Integer clamp bounds must retain the floating input's precision."""

import pytest
import torch

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class ClampLargeIntegerBound(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        # 2**24 + 1 is exact in float64, but rounds down in float32.
        # Subtraction makes that rounding visible to approximate IO checks.
        if self.op == "clamp_min":
            y = x.clamp_min(16777217)
        elif self.op == "clamp_max":
            y = x.clamp_max(16777217)
        else:
            y = x.clamp(16777217, 16777219)
        return y - 16777216.0


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
@pytest.mark.parametrize("op", ["clamp", "clamp_min", "clamp_max"])
def test_clamp_float64_integer_bound_precision(op, inference_target):
    check_model_io_test(
        ClampLargeIntegerBound(op),
        torch.tensor([0.0, 16777218.0, 16777220.0], dtype=torch.float64),
        inference_target,
    )
