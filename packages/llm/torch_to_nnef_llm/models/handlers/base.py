import typing as T
from abc import ABC, abstractmethod

class ArchitectureHandler(ABC):
    """Base type for architecture-specific export behavior"""

    ARCH_NAMES: T.Tuple[str, ...] = ()

    @staticmethod
    @abstractmethod
    def get_wrapper_class():
        """Return the wrapper class or factory for this architecture"""
