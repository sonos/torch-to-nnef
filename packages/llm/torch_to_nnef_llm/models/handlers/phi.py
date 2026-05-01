from torch_to_nnef_llm.models.base import BaseCausalWithDynCacheAndTriu

from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class PhiArchitectureHandler(DefaultArchitectureHandler):
    """Handler for Phi-family models."""

    ARCH_NAMES = ("phi",)

    @staticmethod
    def get_wrapper_class():
        return BaseCausalWithDynCacheAndTriu
