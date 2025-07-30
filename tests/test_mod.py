import pytest

from tests.mod.nemo_featurizer import FilterbankFeatures
import torch

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from torch_to_nnef.inference_target import TractNNEF

test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


def cond_tract_gt_0_20_7(i) -> bool:
    return isinstance(i, TractNNEF) and i.version > "0.20.7"


def add_test(*args):
    global test_suite
    test_suite.add(*args, inference_conditions=cond_tract_gt_0_20_7)


add_test(
    (torch.rand(1, 16000), torch.tensor([16000])),
    FilterbankFeatures(dither=False),
)


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_complex_and_fft_export(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
