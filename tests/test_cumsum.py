import pytest
import torch

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from tests.wrapper import UnaryPrimitive


def _skip_if_not_tract(inf):
    # Only test on Tract targets; Khronos does not implement cumsum fragment
    from torch_to_nnef.inference_target import TractNNEF

    return isinstance(inf, TractNNEF)


test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


# Float tensor cases across typical axes
_inp = torch.arange(24).reshape(2, 3, 4).float()
for axis in [0, 1, 2, -1]:
    test_suite.add(
        (_inp,),
        UnaryPrimitive(lambda x, a=axis: torch.cumsum(x, dim=a)),
        inference_conditions=_skip_if_not_tract,
    )


# Integer tensor cases across typical axes
_inp_i = torch.arange(24).reshape(2, 3, 4)
for axis in [0, 1, 2, -1]:
    test_suite.add(
        (_inp_i,),
        UnaryPrimitive(lambda x, a=axis: torch.cumsum(x, dim=a)),
        inference_conditions=_skip_if_not_tract,
    )


# cumprod across typical axes. Use a small-magnitude float input so the
# running product doesn't blow up.
_inp_p = torch.tensor(
    [
        [[0.5, 1.5, -0.5, 2.0], [0.8, 1.2, 0.9, -1.1], [1.0, 0.7, 1.3, 0.95]],
        [
            [1.1, 0.9, 1.05, 0.85],
            [-1.2, 0.95, 1.15, 1.0],
            [0.6, 1.4, 0.7, 1.25],
        ],
    ],
)
for axis in [0, 1, 2, -1]:
    test_suite.add(
        (_inp_p,),
        UnaryPrimitive(lambda x, a=axis: torch.cumprod(x, dim=a)),
        inference_conditions=_skip_if_not_tract,
    )


class _CumMaxModel(torch.nn.Module):
    """Return both values+indices so the test framework checks both."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.cummax(x, dim=self.dim)
        return out.values, out.indices


class _CumMinModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.cummin(x, dim=self.dim)
        return out.values, out.indices


# cummax / cummin across typical axes. Use a varied input that has
# multiple plateaus so tie-breaking exercises (cummax/cummin keep the
# first occurrence on equal values).
_inp_m = torch.tensor(
    [
        [
            [0.5, 1.5, -0.5, 2.0],
            [2.0, 1.2, 0.9, -1.1],
            [1.0, 0.7, 1.3, 0.95],
        ],
        [
            [1.1, 0.9, 1.05, 0.85],
            [-1.2, 0.95, 1.15, 1.0],
            [0.6, 1.4, 0.7, 1.25],
        ],
    ],
)
for axis in [0, 1, 2, -1]:
    test_suite.add(
        (_inp_m,),
        _CumMaxModel(dim=axis),
        inference_conditions=_skip_if_not_tract,
    )
    test_suite.add(
        (_inp_m,),
        _CumMinModel(dim=axis),
        inference_conditions=_skip_if_not_tract,
    )


@pytest.mark.parametrize(
    "_id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_cumsum_export(_id, test_input, model, inference_target):
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
