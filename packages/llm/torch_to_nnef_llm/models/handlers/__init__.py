from typing import Dict, Type

from .base import ArchitectureHandler
from .default import DefaultArchitectureHandler
from .openelm import OpenELMArchitectureHandler
from .phi import PhiArchitectureHandler

_HANDLER_REGISTRY: Dict[str, Type[ArchitectureHandler]] = {
    arch_name: handler
    for handler in (
        DefaultArchitectureHandler,
        OpenELMArchitectureHandler,
        PhiArchitectureHandler,
    )
    for arch_name in handler.ARCH_NAMES
}

def get_handler(model_type: str) -> Type[ArchitectureHandler]:
    """Return the registered handler for a model_type"""
    return _HANDLER_REGISTRY.get(model_type, DefaultArchitectureHandler)

__all__ = [
    "ArchitectureHandler",
    "DefaultArchitectureHandler",
    "OpenELMArchitectureHandler",
    "PhiArchitectureHandler",
    "get_handler",
]
