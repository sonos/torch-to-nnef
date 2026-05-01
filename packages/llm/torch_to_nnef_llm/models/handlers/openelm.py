from functools import partial

from torch_to_nnef_llm.models.base import BaseCausal

from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class OpenELMArchitectureHandler(DefaultArchitectureHandler):
    """Handler for OpenELM models."""

    ARCH_NAMES = ("openelm",)

    @staticmethod
    def get_wrapper_class():
        return partial(BaseCausal, with_dyn_cache=False)
