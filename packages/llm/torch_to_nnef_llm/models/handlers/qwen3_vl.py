from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class Qwen3VLArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Qwen3-VL models."""

    ARCH_NAMES = ("qwen3_vl",)

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Qwen3VLForConditionalGeneration
