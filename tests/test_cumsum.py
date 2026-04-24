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


@pytest.mark.parametrize(
    "_id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_cumsum_export(_id, test_input, model, inference_target):
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
