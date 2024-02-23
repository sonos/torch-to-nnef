import os
from enum import Enum

import pytest
import torch
from transformers import (  # LlamaConfig,; LlamaModel,; LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import (  # LlamaRotaryEmbedding,
    LlamaDecoderLayer,
)

from torch_to_nnef.tract import tract_version_greater_than

from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))
INPUT_AND_MODELS = []


class StripedModel(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        # return self.model.generate(input_ids, max_length=30)
        transformers_outputs = self.model.model(
            input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        return transformers_outputs
        # transformers_outputs contains 'logits', 'past_key_values'
        # hidden_states = transformers_outputs[0]
        # return hidden_states
        return transformers_outputs.logits


class StripedDecodingLayer(torch.nn.Module):
    def __init__(
        self,
        model: LlamaDecoderLayer,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        super().__init__()
        self.model = model
        self.attention_mask = attention_mask
        self.position_ids = position_ids

    def forward(self, hidden_states):
        return self.model(
            hidden_states,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            past_key_value=None,  # opt-in
            output_attentions=False,
            use_cache=False,
        )


class Llama2SLugs(str, Enum):
    DUMMY = "yujiepan/llama-2-tiny-random"
    TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# working exports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_MODEL_SLUG = os.environ.get("LLAMA_SLUG", Llama2SLugs.DUMMY.value)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_SLUG)
causal_llama = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_SLUG)
striped_model = StripedModel(causal_llama)
inputs = tokenizer("Hello, I am happy", return_tensors="pt")
if tract_version_greater_than("0.19.0"):
    INPUT_AND_MODELS += [
        (
            tuple(
                inputs.input_ids.unsqueeze(0),
            ),
            striped_model,
        )
    ]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_model_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
