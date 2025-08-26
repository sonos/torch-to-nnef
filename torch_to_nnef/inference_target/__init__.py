"""Targeted inference engine.

We mainly focus our effort to best support SONOS 'tract' inference engine.

Stricter Khronos NNEF specification mode also exist
but is less extensively tested.


"""

from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.khronos import KhronosNNEF
from torch_to_nnef.inference_target.tract import TractNNEF

__all__ = ["InferenceTarget", "KhronosNNEF", "TractNNEF"]
