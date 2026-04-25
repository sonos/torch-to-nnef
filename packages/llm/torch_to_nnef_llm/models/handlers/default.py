from torch_to_nnef_llm.models.base import BaseCausal

from .base import ArchitectureHandler

class DefaultArchitectureHandler(ArchitectureHandler):
    """Fallback handler for standard causal decoder models"""

    ARCH_NAMES = ("default",)

    @staticmethod
    def get_wrapper_class():
        return BaseCausal
