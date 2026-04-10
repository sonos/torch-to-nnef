"""Optional type utilities for NeMo runtime type checking.

Help with dependency injection for nemo, omegaconf, pytorch-lightning,
questionary, torchaudio, and huggingface-hub.
"""

from typing import TYPE_CHECKING, Any, Union

from typing_extensions import TypeAlias

from torch_to_nnef.utils import Injected

if TYPE_CHECKING:
    import huggingface_hub
    import nemo
    import omegaconf
    import pytorch_lightning
    import questionary
    import torchaudio
else:
    huggingface_hub = Any
    nemo = Any
    omegaconf = Any
    pytorch_lightning = Any
    questionary = Any
    torchaudio = Any


HuggingFaceHubModule: TypeAlias = huggingface_hub
NemoModule: TypeAlias = nemo
OmegaConfModule: TypeAlias = omegaconf
LightningModule: TypeAlias = pytorch_lightning
QuestionaryModule: TypeAlias = questionary
TorchaudioModule: TypeAlias = torchaudio


InjectedHuggingFaceHubModule: TypeAlias = Union[HuggingFaceHubModule, Injected]
InjectedNemoModule: TypeAlias = Union[NemoModule, Injected]
InjectedOmegaConfModule: TypeAlias = Union[OmegaConfModule, Injected]
InjectedLightningModule: TypeAlias = Union[LightningModule, Injected]
InjectedQuestionaryModule: TypeAlias = Union[QuestionaryModule, Injected]
InjectedTorchaudioModule: TypeAlias = Union[TorchaudioModule, Injected]
