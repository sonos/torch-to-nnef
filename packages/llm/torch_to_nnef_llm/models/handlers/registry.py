from typing import Dict, Type

from torch_to_nnef.exceptions import T2NErrorConsistency, T2NErrorMisuse

from .base import ArchitectureHandler

_HANDLER_REGISTRY: Dict[str, Type[ArchitectureHandler]] = {}


def register_handler(
    handler_cls: Type[ArchitectureHandler],
) -> Type[ArchitectureHandler]:
    """Register an architecture handler class for its declared names."""
    if not issubclass(handler_cls, ArchitectureHandler):
        raise T2NErrorMisuse(
            f"{handler_cls!r} must inherit from ArchitectureHandler"
        )
    if not handler_cls.ARCH_NAMES:
        raise T2NErrorMisuse(
            f"{handler_cls.__name__} must define at least one ARCH_NAMES entry"
        )
    for arch_name in handler_cls.ARCH_NAMES:
        if arch_name in _HANDLER_REGISTRY:
            raise T2NErrorConsistency(
                f"duplicate handler for {arch_name!r}: "
                f"{handler_cls.__name__} vs "
                f"{_HANDLER_REGISTRY[arch_name].__name__}"
            )
        _HANDLER_REGISTRY[arch_name] = handler_cls
    return handler_cls


def get_handler(model_type: str) -> Type[ArchitectureHandler]:
    """Return the registered handler for a model type."""
    return _HANDLER_REGISTRY.get(model_type, _HANDLER_REGISTRY["default"])
