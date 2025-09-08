import os
from functools import partial

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_to_nnef.llm_tract.config import LlamaSlugs
from torch_to_nnef.llm_tract.models.base import (
    TRANSFORMERS_VERSION,
    BaseCausalWithDynCacheAndTriu,
)
from torch_to_nnef.utils import torch_version

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    change_dynamic_axes,
    check_model_io_test,
    set_seed,
    transformers_tract_export_test_condition,
)

set_seed(int(os.environ.get("SEED", 25)))

test_suite = TestSuiteInferenceExactnessBuilder(
    [
        _
        for _ in TRACT_INFERENCES_TO_TESTS_APPROX
        if transformers_tract_export_test_condition(_)
    ]
)


if torch_version() > "1.13.0":
    # working exports
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    DEFAULT_MODEL_SLUG = os.environ.get("LLAMA_SLUG", LlamaSlugs.DUMMY.value)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_SLUG)
    causal_llama = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_SLUG)
    striped_model = BaseCausalWithDynCacheAndTriu(causal_llama)
    inputs = tokenizer("Hello, I am happy", return_tensors="pt")

    S = 10
    test_suite.add(
        (
            # can only be 1 @ export time since regressive model
            inputs.input_ids[:, :1],
            # kv cache
            torch.rand((1, 2, S, 4)),
            torch.rand((1, 2, S, 4)),
        ),
        striped_model,
        inference_modifier=partial(
            change_dynamic_axes,
            dynamic_axes={
                "input_0": {1: "S"},
                "input_1": {2: "P"},
                "input_2": {2: "P"},
            },
        ),
    )


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_llama_export(id, test_input, model, inference_target):
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
