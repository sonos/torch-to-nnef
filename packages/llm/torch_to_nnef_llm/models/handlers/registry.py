from typing import Dict, List, Type

from torch_to_nnef.exceptions import T2NErrorConsistency, T2NErrorMisuse

from .base import ArchitectureHandler, EncoderHandler

_HANDLER_REGISTRY: Dict[str, Type[ArchitectureHandler]] = {}

#: Encoder handlers keyed by ``config.model_type``. A single multimodal model
#: type may register more than one encoder handler (e.g. a vision tower and an
#: audio tower), so values are lists.
_ENCODER_REGISTRY: Dict[str, List[Type[EncoderHandler]]] = {}


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


def register_encoder_handler(
    handler_cls: Type[EncoderHandler],
) -> Type[EncoderHandler]:
    """Register a modality-encoder handler for its declared model types."""
    if not issubclass(handler_cls, EncoderHandler):
        raise T2NErrorMisuse(
            f"{handler_cls!r} must inherit from EncoderHandler"
        )
    if not handler_cls.ARCH_NAMES:
        raise T2NErrorMisuse(
            f"{handler_cls.__name__} must define at least one ARCH_NAMES entry"
        )
    for arch_name in handler_cls.ARCH_NAMES:
        registered = _ENCODER_REGISTRY.setdefault(arch_name, [])
        if handler_cls in registered:
            raise T2NErrorConsistency(
                f"encoder handler {handler_cls.__name__} already registered "
                f"for {arch_name!r}"
            )
        if any(h.MODALITY == handler_cls.MODALITY for h in registered):
            raise T2NErrorConsistency(
                f"duplicate encoder handler for modality "
                f"{handler_cls.MODALITY!r} of {arch_name!r}: "
                f"{handler_cls.__name__} vs "
                f"{[h.__name__ for h in registered]}"
            )
        registered.append(handler_cls)
    return handler_cls


def get_encoder_handlers(model_type: str) -> List[Type[EncoderHandler]]:
    """Return registered encoder handlers for a model type (may be empty)."""
    return list(_ENCODER_REGISTRY.get(model_type, []))


def is_multimodal(model_type: str) -> bool:
    """Whether any encoder handler is registered for this model type."""
    return bool(_ENCODER_REGISTRY.get(model_type))
