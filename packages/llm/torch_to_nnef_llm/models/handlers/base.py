from dataclasses import dataclass
import typing as T
from abc import ABC, abstractmethod

import torch

@dataclass
class InputSpec:
    """Defines the exported input/output signature for one decoder graph"""

    inputs: T.Tuple[torch.Tensor, ...]
    input_names: T.List[str]
    output_names: T.List[str]
    dynamic_axes: T.Dict[str, T.Dict[int, str]]

class ArchitectureHandler(ABC):
    """Base type for architecture-specific export behavior"""

    ARCH_NAMES: T.Tuple[str, ...] = ()

    @staticmethod
    @abstractmethod
    def get_wrapper_class():
        """Return the wrapper class or factory for this architecture"""

    @abstractmethod
    def build_input_spec(
        self,
        *,
        tokenizer,
        config_helper,
        inputs_dtype: torch.dtype,
        sample_text: str,
        n_input_tokens: int,
        n_past_input_tokens: int,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
    ) -> InputSpec:
        """Build exported inputs plus names/dynamic axes for the decoder"""

    @abstractmethod
    def prepare_inputs_for_model(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> T.Dict[str, T.Any]:
        """Convert exported inputs into kwargs expected by the HF model"""
