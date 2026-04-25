from functools import partial

from torch_to_nnef_llm.models.base import BaseCausal

from .base import ArchitectureHandler


class OpenELMArchitectureHandler(ArchitectureHandler):
    """Handler for OpenELM models."""

    ARCH_NAMES = ("openelm",)

    @staticmethod
    def get_wrapper_class():
        return partial(BaseCausal, with_dyn_cache=False)
