"""Tests dynamic_axes."""

import os

import pytest
import torch

from tests.test_primitive import TorchFnPrimitive
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.op.helper import (
    DTYPES_EXPECTED_IMPLICIT_CAST_ORDER,
    IMPLICIT_CAST_SUPPORTED_OPS,
)

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))


def is_tract(i):
    return isinstance(i, TractNNEF)


test_suite = TestSuiteInferenceExactnessBuilder(TRACT_INFERENCES_TO_TESTS)

_base_tensor = torch.arange(6).reshape(2, 3)
base_tensor = _base_tensor[:]
for op in IMPLICIT_CAST_SUPPORTED_OPS:
    for idx, dtype in enumerate(DTYPES_EXPECTED_IMPLICIT_CAST_ORDER):
        for other_dtype in DTYPES_EXPECTED_IMPLICIT_CAST_ORDER[idx:]:
            if dtype == other_dtype == torch.bool:
                continue
            if (
                dtype == torch.bool or other_dtype == torch.bool
            ) and op == "sub":
                continue
            if op == "div":
                base_tensor = _base_tensor[:] + 1
            test_suite.add(
                (base_tensor.to(dtype), base_tensor.to(other_dtype)),
                TorchFnPrimitive(op),
                inference_conditions=is_tract,
            )
            if op == "div":
                base_tensor = _base_tensor[:]


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_dynamic_axes_exports(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
