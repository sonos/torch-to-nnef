from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_to_nnef_llm.models.base import BaseCausal
from torch_to_nnef_llm.models.handlers import Qwen3VLArchitectureHandler


class FakeQwen3VLInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope_deltas = torch.tensor([[99]], dtype=torch.long)

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        mm_token_type_ids=None,
    ):
        del image_grid_thw, video_grid_thw, attention_mask, mm_token_type_ids
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long).view(
            1, 1, seq_length
        )
        position_ids = position_ids.repeat(3, batch_size, 1)
        rope_deltas = torch.full((batch_size, 1), 7, dtype=torch.long)
        return position_ids, rope_deltas


class FakeQwen3VLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            image_token_id=8,
            video_token_id=9,
            text_config=SimpleNamespace(
                max_position_embeddings=128,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=4,
            ),
        )
        self.model = FakeQwen3VLInnerModel()
        self.embeddings = nn.Embedding(16, 4)
        self.last_kwargs = None

    def get_input_embeddings(self):
        return self.embeddings

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        batch_size, seq_length = kwargs["inputs_embeds"].shape[:2]
        return {
            "logits": torch.zeros(batch_size, seq_length, 4),
        }


class CapturingQwen3VLArchitectureHandler(Qwen3VLArchitectureHandler):
    def __init__(self):
        super().__init__()
        self.last_state = None

    def build_forward_outputs(self, *, state_context, **kwargs):
        self.last_state = dict(state_context.state)
        return super().build_forward_outputs(
            state_context=state_context,
            **kwargs,
        )


def test_qwen3_vl_get_auto_model_class_uses_conditional_generation():
    class FakeTransformers:
        Qwen3VLForConditionalGeneration = object()

    assert (
        Qwen3VLArchitectureHandler.get_auto_model_class(FakeTransformers)
        is FakeTransformers.Qwen3VLForConditionalGeneration
    )


def test_qwen3_vl_inject_token_features_rejects_shape_mismatch():
    inputs_embeds = torch.zeros(1, 3, 4)
    token_mask = torch.tensor([[False, True, True]])
    features = torch.ones(1, 4)

    with pytest.raises(
        ValueError,
        match="feature/slot count mismatch: got 1 feature\\(s\\) "
        "for 2 placeholder slot\\(s\\) in input_ids",
    ):
        Qwen3VLArchitectureHandler._inject_token_features(
            inputs_embeds=inputs_embeds,
            token_mask=token_mask,
            features=features,
        )


def test_qwen3_vl_wrapper_uses_handler_prepared_model_inputs():
    model = FakeQwen3VLModel()
    handler = CapturingQwen3VLArchitectureHandler()
    wrapper = BaseCausal(model, handler=handler)

    input_ids = torch.tensor([[1, 8, 8]], dtype=torch.long)
    cache_key = torch.rand(1, 1, 2, 4)
    cache_value = torch.rand(1, 1, 2, 4)
    image_embeddings = torch.rand(2, 4)
    video_embeddings = torch.zeros(0, 4)
    image_grid_thw = torch.tensor([[1, 2, 4]], dtype=torch.long)
    video_grid_thw = torch.zeros(0, 3, dtype=torch.long)
    rope_deltas_state = torch.zeros(0, 1, dtype=torch.long)
    previous_rope_deltas = model.model.rope_deltas.clone()

    outputs = wrapper(
        input_ids,
        cache_key,
        cache_value,
        image_embeddings,
        video_embeddings,
        image_grid_thw,
        video_grid_thw,
        rope_deltas_state,
    )

    model_kwargs = model.last_kwargs
    assert model_kwargs is not None
    assert tuple(model_kwargs["inputs_embeds"].shape) == (1, 3, 4)
    assert tuple(model_kwargs["attention_mask"].shape) == (1, 1, 3, 5)
    assert tuple(model_kwargs["position_ids"].shape) == (3, 1, 3)
    assert torch.equal(model.model.rope_deltas, previous_rope_deltas)
    assert handler.last_state is not None
    assert set(handler.last_state) >= {
        "image_embeddings",
        "video_embeddings",
        "image_grid_thw",
        "video_grid_thw",
        "prev_rope_deltas",
        "last_rope_deltas",
    }
    assert torch.equal(outputs[-5], image_embeddings)
    assert torch.equal(outputs[-4], video_embeddings)
    assert torch.equal(outputs[-3], image_grid_thw)
    assert torch.equal(outputs[-2], video_grid_thw)
    assert torch.equal(outputs[-1], torch.full((1, 1), 7, dtype=torch.long))
