"""Top-level package for torch_to_nnef."""

__author__ = """Julien Balian"""
__email__ = "julien.balian@sonos.com"
__version__ = "0.13.16"

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import KhronosNNEF, TractNNEF

__all__ = [
    "export_model_to_nnef",
    "TractNNEF",
    "KhronosNNEF",
    "__author__",
    "__email__",
    "__version__",
]
