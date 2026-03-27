"""NVIDIA NeMo export helpers (TractNNEF-focused).

This package splits the original monolithic implementation into
submodules while keeping the same public API.

Public re-exports preserved for compatibility:
- constants: PARAKEET_V3_SLUG, PARAKEET_110M_SLUG, NEMOTRON_0_6B
- functions/classes: iter_export_params_for_generic_nemo_asr_model,
  export_nemo_asr_model, main

Programmatic (non-CLI) API:
- export_nemo_from_model, NemoExportConfig
"""

from torch_to_nnef.nemo_tract.cli import main  # CLI entry-point
from torch_to_nnef.nemo_tract.config import NemoExportConfig
from torch_to_nnef.nemo_tract.entry import export_nemo_from_model
from torch_to_nnef.nemo_tract.export import (
    export_nemo_asr_model,
    iter_export_params_for_generic_nemo_asr_model,
)
from torch_to_nnef.nemo_tract.model_loader import (
    NEMOTRON_0_6B,
    PARAKEET_110M_SLUG,
    PARAKEET_V3_SLUG,
)

__all__ = [
    "PARAKEET_V3_SLUG",
    "PARAKEET_110M_SLUG",
    "NEMOTRON_0_6B",
    "iter_export_params_for_generic_nemo_asr_model",
    "export_nemo_asr_model",
    "main",
    "export_nemo_from_model",
    "NemoExportConfig",
]
