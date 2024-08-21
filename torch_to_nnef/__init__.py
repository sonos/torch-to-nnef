"""Top-level package for torch_to_nnef."""

__author__ = """Julien Balian"""
__email__ = "julien.balian@sonos.com"
__version__ = "0.12.3"

from torch_to_nnef.export import export_model_to_nnef

__all__ = ["export_model_to_nnef", "__author__", "__email__", "__version__"]
