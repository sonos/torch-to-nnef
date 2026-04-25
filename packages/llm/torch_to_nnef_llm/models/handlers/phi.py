from torch_to_nnef_llm.models.base import BaseCausalWithDynCacheAndTriu

from .base import ArchitectureHandler


class PhiArchitectureHandler(ArchitectureHandler):
    """Handler for Phi-family models"""

    ARCH_NAMES = ("phi",)

    @staticmethod
    def get_wrapper_class():
        return BaseCausalWithDynCacheAndTriu
