"""Optional type utilities for LLM runtime type checking.

Help with dependency injection for transformers, huggingface-hub, and peft.
"""

from typing import TYPE_CHECKING, Any, Protocol, Union

from torch_to_nnef.utils import Injected
from typing_extensions import TypeAlias


class _TransformersCacheUtilsProto(Protocol):
    class DynamicCache:
        pass


class _TransformersProto(Protocol):
    class AutoTokenizer:
        pass

    class AutoModelForCausalLM:
        pass


if TYPE_CHECKING:
    import huggingface_hub
    import peft
    import transformers
    import transformers.cache_utils as transformers_cache_utils
    import transformers.utils as transformers_utils
else:
    transformers = _TransformersProto
    transformers_utils = Any
    transformers_cache_utils = _TransformersCacheUtilsProto
    huggingface_hub = Any
    peft = Any


TransformersModule: TypeAlias = transformers
TransformersUtilsModule: TypeAlias = transformers_utils
TransformersCacheUtils: TypeAlias = transformers_cache_utils
HuggingFaceHubModule: TypeAlias = huggingface_hub
PeftModule: TypeAlias = peft


InjectedTransformersModule: TypeAlias = Union[TransformersModule, Injected]
InjectedTransformersUtilsModule: TypeAlias = Union[
    TransformersUtilsModule, Injected
]
InjectedTransformersCacheUtilsModule: TypeAlias = Union[
    TransformersCacheUtils, Injected
]
InjectedHuggingFaceHubModule: TypeAlias = Union[HuggingFaceHubModule, Injected]
InjectedPeftModule: TypeAlias = Union[PeftModule, Injected]
