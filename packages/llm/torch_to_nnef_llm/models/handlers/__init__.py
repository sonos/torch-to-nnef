from .base import (
    ArchitectureHandler,
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    MultiModalArchitectureHandler,
    StateContext,
)
from .default import DefaultArchitectureHandler
from .gemma3_vl import (
    Gemma3ArchitectureHandler,
    Gemma3VisionEncoderHandler,
)
from .gemma4_vl import (
    Gemma4ArchitectureHandler,
    Gemma4VideoEncoderHandler,
    Gemma4VisionEncoderHandler,
)
from .gpt_oss import GptOssArchitectureHandler
from .idefics3_vl import (
    Idefics3ArchitectureHandler,
    Idefics3VisionEncoderHandler,
)
from .openelm import OpenELMArchitectureHandler
from .phi import PhiArchitectureHandler
from .qwen2_5_vl import (
    Qwen25VLArchitectureHandler,
    Qwen25VLVisionEncoderHandler,
)
from .qwen3_vl import (
    Qwen3VLArchitectureHandler,
    Qwen3VLVisionEncoderHandler,
)
from .qwen3_next import Qwen3NextArchitectureHandler
from .qwen35_moe import Qwen35MoeArchitectureHandler
from .registry import (
    get_encoder_handlers,
    get_handler,
    is_multimodal,
    register_encoder_handler,
    register_handler,
)
from .voxtral import (
    VoxtralArchitectureHandler,
    VoxtralAudioEncoderHandler,
)

__all__ = [
    "ArchitectureHandler",
    "DefaultArchitectureHandler",
    "EmbeddingContract",
    "EncoderHandler",
    "Gemma3ArchitectureHandler",
    "Gemma3VisionEncoderHandler",
    "Gemma4ArchitectureHandler",
    "Gemma4VideoEncoderHandler",
    "Gemma4VisionEncoderHandler",
    "GptOssArchitectureHandler",
    "IOSpec",
    "Idefics3ArchitectureHandler",
    "Idefics3VisionEncoderHandler",
    "MultiModalArchitectureHandler",
    "OpenELMArchitectureHandler",
    "PhiArchitectureHandler",
    "Qwen25VLArchitectureHandler",
    "Qwen25VLVisionEncoderHandler",
    "Qwen35MoeArchitectureHandler",
    "Qwen3NextArchitectureHandler",
    "Qwen3VLArchitectureHandler",
    "Qwen3VLVisionEncoderHandler",
    "StateContext",
    "VoxtralArchitectureHandler",
    "VoxtralAudioEncoderHandler",
    "get_encoder_handlers",
    "get_handler",
    "is_multimodal",
    "register_encoder_handler",
    "register_handler",
]
