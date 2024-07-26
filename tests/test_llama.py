import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.export_llama import LlamaSLugs, SuperBasicCausal
from torch_to_nnef.tract import tract_version

from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))
INPUT_AND_MODELS = []


# working exports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_MODEL_SLUG = os.environ.get("LLAMA_SLUG", LlamaSLugs.DUMMY.value)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_SLUG)
causal_llama = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_SLUG)
striped_model = SuperBasicCausal(causal_llama)
inputs = tokenizer("Hello, I am happy", return_tensors="pt")
if tract_version() > "0.21.5":  # prior bug in tract
    S = 10

    INPUT_AND_MODELS += [
        (
            (
                # can only be 1 @ export time since regressive model
                inputs.input_ids[:, :1],
                # kv cache
                torch.rand((1, 2, S, 4)),
                torch.rand((1, 2, S, 4)),
            ),
            striped_model,
            {"input_0": {1: "S"}, "input_1": {2: "P"}, "input_2": {2: "P"}},
        )
    ]


@pytest.mark.parametrize("test_input,model,dynamic_axes", INPUT_AND_MODELS)
def test_model_export(test_input, model, dynamic_axes):
    """Test simple models"""
    check_model_io_test(
        model=model,
        test_input=test_input,
        dynamic_axes=dynamic_axes,
    )
