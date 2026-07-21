from types import SimpleNamespace

import torch

from torch_to_nnef_llm.models.handlers import Qwen35MoeArchitectureHandler
from torch_to_nnef_llm.models.handlers.registry import get_handler


class FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        del text, return_tensors
        return SimpleNamespace(
            input_ids=torch.arange(16, dtype=torch.long).reshape(1, 16)
        )


class FakeQwen35Config(SimpleNamespace):
    def get_text_config(self, decoder=False):
        del decoder
        return self


class FakeConfigHelper:
    def __init__(self):
        self.decoder_conf = FakeQwen35Config(
            layer_types=["full_attention", "linear_attention"],
            num_hidden_layers=2,
            num_key_value_heads=2,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
            max_position_embeddings=128,
        )

    def get_num_kv_heads(self, layer_idx):
        del layer_idx
        return self.decoder_conf.num_key_value_heads

    def get_head_dim(self):
        return self.decoder_conf.head_dim


def test_qwen35_moe_handler_is_registered():
    assert get_handler("qwen3_5_moe_text") is Qwen35MoeArchitectureHandler


def test_qwen35_moe_input_spec_names_hybrid_cache_tensors():
    handler = Qwen35MoeArchitectureHandler()
    spec = handler.build_input_spec(
        tokenizer=FakeTokenizer(),
        config_helper=FakeConfigHelper(),
        inputs_dtype=torch.float32,
        sample_text="hello",
        n_input_tokens=3,
        n_past_input_tokens=5,
    )

    assert spec.input_names == [
        "input_ids",
        "in_cache_key_0",
        "in_cache_value_0",
        "in_cache_conv_1",
        "in_cache_recurrent_1",
    ]
    assert spec.output_names == [
        "outputs",
        "out_cache_key_0",
        "out_cache_value_0",
        "out_cache_conv_1",
        "out_cache_recurrent_1",
    ]
    assert tuple(spec.inputs[1].shape) == (1, 2, 5, 16)
    assert tuple(spec.inputs[3].shape) == (1, 192, 4)
    assert tuple(spec.inputs[4].shape) == (1, 4, 16, 16)
    assert spec.dynamic_axes["in_cache_key_0"] == {2: "P"}
    assert "in_cache_conv_1" not in spec.dynamic_axes


def test_qwen35_moe_forward_inputs_rebuild_transformers_hybrid_cache():
    handler = Qwen35MoeArchitectureHandler()
    config_helper = FakeConfigHelper()
    spec = handler.build_input_spec(
        tokenizer=FakeTokenizer(),
        config_helper=config_helper,
        inputs_dtype=torch.float32,
        sample_text="hello",
        n_input_tokens=3,
        n_past_input_tokens=5,
    )
    wrapper = SimpleNamespace(
        model=SimpleNamespace(config=config_helper.decoder_conf),
        force_causal_mask=True,
    )

    state_context = handler.build_forward_inputs(
        inputs=spec.inputs,
        wrapper=wrapper,
    )
    cache = state_context.model_inputs["past_key_values"]

    assert [type(layer).__name__ for layer in cache.layers] == [
        "DynamicLayer",
        "LinearAttentionLayer",
    ]
    assert tuple(cache.layers[0].keys.shape) == (1, 2, 5, 16)
    assert tuple(cache.layers[0].values.shape) == (1, 2, 5, 16)
    assert tuple(cache.layers[1].conv_states.shape) == (1, 192, 4)
    assert tuple(cache.layers[1].recurrent_states.shape) == (1, 4, 16, 16)
    assert cache.has_previous_state(1)
    assert tuple(state_context.model_inputs["attention_mask"].shape) == (
        1,
        1,
        3,
        8,
    )
