import os
from functools import partial

import pytest
import torch
from tests.utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    change_dynamic_axes,
    check_model_io_test,
    set_seed,
    transformers_tract_export_test_condition,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_to_nnef.utils import torch_version
from torch_to_nnef_llm.config import HFConfigHelper, LlamaSlugs
from torch_to_nnef_llm.exporter import LLMExporter
from torch_to_nnef_llm.models.base import BaseCausal

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
    model_infos = HFConfigHelper(causal_llama.config)
    striped_model = BaseCausal(
        causal_llama,
        handler=model_infos.handler,
        with_dyn_cache=model_infos.handler.with_dyn_cache,
    )
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


@pytest.mark.skipif(
    torch_version() <= "1.13.0", reason="export needs torch > 1.13.0"
)
@pytest.mark.parametrize("num_logits_to_keep", [1, 3])
@pytest.mark.parametrize(
    "inference_target",
    TRACT_INFERENCES_TO_TESTS_APPROX,
    ids=[str(_) for _ in TRACT_INFERENCES_TO_TESTS_APPROX],
)
def test_llama_dynamic_logits_to_keep(num_logits_to_keep, inference_target):
    """`logits_to_keep` exposed as a runtime input gathers the last k rows.

    One export then serves prefill (pass 1) and speculative decode (pass k+1);
    the scalar drives a tract_core_range + tract_core_gather, so the kept row
    count is decided at run time instead of baked at export.
    """
    dyn_model = BaseCausal(
        causal_llama,
        handler=model_infos.handler,
        with_dyn_cache=model_infos.handler.with_dyn_cache,
        num_logits_to_keep="dynamic",
    )
    k = num_logits_to_keep
    past_s = 10
    # need at least k positions in the sequence to keep k rows
    ids = tokenizer("Hello, I am happy today and", return_tensors="pt")
    test_input = (
        ids.input_ids[:, :k],
        torch.rand((1, 2, past_s, 4)),
        torch.rand((1, 2, past_s, 4)),
        torch.tensor(k, dtype=torch.int64),
    )
    target = change_dynamic_axes(
        inference_target,
        dynamic_axes={
            "input_ids": {1: "S"},
            "in_cache_key_0": {2: "P"},
            "in_cache_value_0": {2: "P"},
        },
    )
    check_model_io_test(
        model=dyn_model,
        test_input=test_input,
        inference_target=target,
        input_names=[
            "input_ids",
            "in_cache_key_0",
            "in_cache_value_0",
            "logits_to_keep",
        ],
        output_names=["logits", "out_cache_key_0", "out_cache_value_0"],
    )


@pytest.mark.skipif(
    torch_version() <= "1.13.0", reason="export needs torch > 1.13.0"
)
def test_llama_dynamic_logits_to_keep_export_spec():
    """The LLMExporter spec path adds logits_to_keep as a plain input.

    Guards the real CLI export path (generate_inputs_io_names_and_dynaxes),
    which check_model_io_test bypasses: the extra scalar must be an input with
    no matching output and no dynamic axis.
    """
    exporter = LLMExporter(
        causal_llama, tokenizer, num_logits_to_keep="dynamic"
    )
    inputs, input_names, output_names, dynamic_axes = (
        exporter.generate_inputs_io_names_and_dynaxes()
    )
    assert input_names[-1] == "logits_to_keep"
    assert len(inputs) == len(input_names) == len(output_names) + 1
    assert "logits_to_keep" not in output_names
    assert "logits_to_keep" not in dynamic_axes
