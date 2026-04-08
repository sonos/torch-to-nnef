"""NeMo ASR export to NNEF via torch-to-nnef.

Public re-exports preserved for compatibility:
- constants: PARAKEET_V3_SLUG, PARAKEET_110M_SLUG, NEMOTRON_0_6B
- functions/classes: iter_export_params_for_generic_nemo_asr_model,
  export_nemo_asr_model, main

Programmatic (non-CLI) API:
- export_nemo_from_model, NemoExportConfig
"""

from torch_to_nnef_nemo.cli import main  # CLI entry-point
from torch_to_nnef_nemo.config import NemoExportConfig
from torch_to_nnef_nemo.export import (
    export_nemo_asr_model,
    export_nemo_from_model,
    iter_export_params_for_generic_nemo_asr_model,
)
from torch_to_nnef_nemo.model_loader import (
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
    "export_nemo_from_model",
    "NemoExportConfig",
    "main",
]
