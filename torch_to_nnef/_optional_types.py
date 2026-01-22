"""Optional type utilities for runtime type checking.

Help with dependency injection.

"""

from typing import TYPE_CHECKING, Any, TypeAlias, Protocol
from torch_to_nnef.utils import Injected


class _TransformersCacheUtilsProto(Protocol):
    class DynamicCache:
        pass


class _TransformersProto(Protocol):
    class AutoTokenizer:
        pass

    class AutoModelForCausalLM:
        pass


if TYPE_CHECKING:
    import transformers
    import transformers.utils as transformers_utils
    import transformers.cache_utils as transformers_cache_utils
    import huggingface_hub
    import peft
    import omegaconf
    import pytorch_lightning
    import nemo
else:
    transformers = _TransformersProto
    transformers_utils = Any
    transformers_cache_utils = _TransformersCacheUtilsProto
    huggingface_hub = Any
    peft = Any
    nemo = Any
    omegaconf = Any
    pytorch_lightning = Any


TransformersModule: TypeAlias = transformers
TransformersUtilsModule: TypeAlias = transformers_utils
TransformersCacheUtils: TypeAlias = transformers_cache_utils
HuggingFaceHubModule: TypeAlias = huggingface_hub
PeftModule: TypeAlias = peft
NemoModule: TypeAlias = nemo
OmegaConfModule: TypeAlias = omegaconf
LightningModule: TypeAlias = pytorch_lightning


InjectedTransformersModule: TypeAlias = TransformersModule | Injected
InjectedTransformersUtilsModule: TypeAlias = TransformersUtilsModule | Injected
InjectedTransformersCacheUtilsModule: TypeAlias = (
    TransformersCacheUtils | Injected
)
InjectedHuggingFaceHubModule: TypeAlias = HuggingFaceHubModule | Injected
InjectedPeftModule: TypeAlias = PeftModule | Injected
InjectedNemoModule: TypeAlias = NemoModule | Injected
InjectedOmegaConfModule: TypeAlias = OmegaConfModule | Injected
InjectedLightningModule: TypeAlias = LightningModule | Injected
