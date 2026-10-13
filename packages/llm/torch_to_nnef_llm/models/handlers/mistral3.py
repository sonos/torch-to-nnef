from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class Mistral3TextArchitectureHandler(DefaultArchitectureHandler):
    """Load the Mistral3 wrapper for text-only export, without vision inputs."""

    ARCH_NAMES = ("mistral3",)

    @staticmethod
    def get_auto_model_class(transformers):
        return transformers.Mistral3ForConditionalGeneration
