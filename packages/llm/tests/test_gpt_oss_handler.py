from types import SimpleNamespace

from torch import nn

from torch_to_nnef_llm.models.handlers import GptOssArchitectureHandler
from torch_to_nnef_llm.models.handlers.registry import get_handler


class FakeGptOssExpert(nn.Module):
    def __init__(self, experts_implementation):
        super().__init__()
        self.config = SimpleNamespace(
            _experts_implementation=experts_implementation
        )


class FakeGptOssModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_experts_implementation="grouped_mm")
        self.grouped = FakeGptOssExpert("grouped_mm")
        self.eager = FakeGptOssExpert("eager")


def test_gpt_oss_handler_registered():
    assert get_handler("gpt_oss") is GptOssArchitectureHandler


def test_gpt_oss_handler_uses_traceable_experts_implementation():
    model = FakeGptOssModel()

    GptOssArchitectureHandler().prepare_model_for_export(model)

    assert model.config._experts_implementation == "batched_mm"
    assert model.grouped.config._experts_implementation == "batched_mm"
    assert model.eager.config._experts_implementation == "eager"
