from copy import deepcopy

import pytest
import torch

from tests.mod.nemo_featurizer import FilterbankFeatures
from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from torch_to_nnef.inference_target import TractNNEF

test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


def cond_tract_gt_0_21_13(i) -> bool:
    return isinstance(i, TractNNEF) and i.version > "0.21.13"


def add_test(*args, inference_modifier=None):
    global test_suite
    test_suite.add(
        *args,
        inference_conditions=cond_tract_gt_0_21_13,
        inference_modifier=inference_modifier,
    )


def inference_stream(it):
    it = deepcopy(it)
    it.dynamic_axes = {"input_0": {0: "B", 1: "S"}, "input_1": {0: "B"}}
    return it


add_test(
    (torch.rand(1, 16000), torch.tensor([16000])),
    FilterbankFeatures(dither=False, pad_to=0),
    inference_modifier=inference_stream,
)


def check_non_concretized_tract_interp(inference_target, export_path):
    if not isinstance(inference_target, TractNNEF):
        return
    inference_target.tract_cli.run(
        [
            str(export_path.absolute()),
            "--nnef-tract-core",
            "--nnef-tract-pulse",
            "-O",
            "dump",
            # "--profile",
            # "--set",
            # "B=1",
            # "--set",
            # "S=16000",
            # "--allow-random-input",
        ],
        quiet=True,
    )


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_complex_and_fft_export(id, test_input, model, inference_target):
    """Test simple models"""
    custom_extensions = None
    if isinstance(model, FilterbankFeatures):
        symb = set(
            [
                v
                for o in inference_target.dynamic_axes.values()
                for v in o.values()
            ]
        )
        if "B" in symb and "S" in symb:
            custom_extensions = [
                "tract_assert S > 1",
                "tract_assert B >= 1",
            ]
    check_model_io_test(
        model=model,
        test_input=test_input,
        inference_target=inference_target,
        custom_extensions=custom_extensions,
        callback_post_export=check_non_concretized_tract_interp,
    )
