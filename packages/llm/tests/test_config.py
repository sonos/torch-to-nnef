from types import SimpleNamespace

from torch_to_nnef_llm.config import HFConfigHelper


def _llama_conf(**kwargs):
    values = {
        "model_type": "llama",
        "max_position_embeddings": 128,
        "hidden_size": 1024,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 2,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_head_dim_falls_back_when_config_value_is_none():
    helper = HFConfigHelper(_llama_conf(head_dim=None))

    assert helper.get_head_dim() == 32


def test_head_dim_prefers_explicit_config_value():
    helper = HFConfigHelper(_llama_conf(head_dim=128))

    assert helper.get_head_dim() == 128
