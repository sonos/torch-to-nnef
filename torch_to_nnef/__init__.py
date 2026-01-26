"""Top-level package for torch_to_nnef."""

__author__ = """Julien Balian"""
__email__ = "julien.balian@sonos.com"
__version__ = "0.20.3"

from torch_to_nnef.export import (
    export_model_to_nnef,
    export_tensors_from_disk_to_nnef,
    export_tensors_to_nnef,
)
from torch_to_nnef.inference_target import KhronosNNEF, TractNNEF
from torch_to_nnef.utils import SemanticVersion

VERSION = SemanticVersion.from_str(__version__)

__all__ = [
    "export_model_to_nnef",
    "export_tensors_to_nnef",
    "export_tensors_from_disk_to_nnef",
    "TractNNEF",
    "KhronosNNEF",
    "__author__",
    "__email__",
    "__version__",
    "VERSION",
]
