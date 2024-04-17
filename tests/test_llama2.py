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

from torch_to_nnef.tract import tract_version

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
        #
        return transformers_outputs.logits


class AttentionMaskModel(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        from transformers.models.llama.modeling_llama import (
            _prepare_4d_causal_attention_mask,
        )

        inputs_embeds = self.model.model.embed_tokens(input_ids)
        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0
        attention_mask = _prepare_4d_causal_attention_mask(
            None,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
        return attention_mask


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
if tract_version() >= "0.21.4":  # prior bug in tract
    # works locally with tract 0.21.3 but seems to need triu export in CI tests ...
    INPUT_AND_MODELS += [
        (
            tuple(
                inputs.input_ids.unsqueeze(0),
            ),
            striped_model,
            {"input_0": {1: "S"}},
        )
    ]
if tract_version() >= "0.21.4":  # prior bug in tract
    # works locally with tract 0.21.3 but seems to need triu export in CI tests ...
    try:
        INPUT_AND_MODELS += [
            (
                tuple(
                    inputs.input_ids.unsqueeze(0),
                ),
                AttentionMaskModel(causal_llama),
                {"input_0": {1: "S"}},
            )
        ]
    except ImportError as exp:
        print(exp)


@pytest.mark.parametrize("test_input,model,dynamic_axes", INPUT_AND_MODELS)
def test_model_export(test_input, model, dynamic_axes):
    """Test simple models"""
    check_model_io_test(
        model=model,
        test_input=test_input,
        dynamic_axes=dynamic_axes,
    )
