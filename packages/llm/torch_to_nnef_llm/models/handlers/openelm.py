from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class OpenELMArchitectureHandler(DefaultArchitectureHandler):
    """Handler for OpenELM models."""

    ARCH_NAMES = ("openelm",)
    with_dyn_cache = False
