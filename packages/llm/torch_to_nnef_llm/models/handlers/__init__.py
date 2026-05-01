from typing import Dict, Type

from torch_to_nnef.exceptions import T2NErrorConsistency

from .base import ArchitectureHandler
from .default import DefaultArchitectureHandler
from .openelm import OpenELMArchitectureHandler
from .phi import PhiArchitectureHandler

_HANDLER_REGISTRY: Dict[str, Type[ArchitectureHandler]] = {}


def register_handler(
    handler_cls: Type[ArchitectureHandler],
) -> Type[ArchitectureHandler]:
    """Register an architecture handler class for its declared names."""
    for arch_name in handler_cls.ARCH_NAMES:
        if arch_name in _HANDLER_REGISTRY:
            raise T2NErrorConsistency(
                f"duplicate handler for {arch_name!r}: "
                f"{handler_cls.__name__} vs "
                f"{_HANDLER_REGISTRY[arch_name].__name__}"
            )
        _HANDLER_REGISTRY[arch_name] = handler_cls
    return handler_cls


for cls in (
    DefaultArchitectureHandler,
    OpenELMArchitectureHandler,
    PhiArchitectureHandler,
):
    register_handler(cls)


def get_handler(model_type: str) -> Type[ArchitectureHandler]:
    """Return the registered handler for a model type."""
    return _HANDLER_REGISTRY.get(model_type, DefaultArchitectureHandler)


__all__ = [
    "ArchitectureHandler",
    "DefaultArchitectureHandler",
    "OpenELMArchitectureHandler",
    "PhiArchitectureHandler",
    "get_handler",
    "register_handler",
]
