"""Advanced QTensor (<= 8bits) with complex quant scheme non torch native"""

from torch_to_nnef.qtensor.base import (
    QTensor,
    QTensorBasic,
    QTensorBasicExtractor,
    QZPScalePerChannel,
    QZPScalePerGroup,
    QZPScaleScalar,
)
from torch_to_nnef.qtensor.qsep import QTensorSepParamsWithPack

__all__ = [
    "QTensor",
    "QTensorBasic",
    "QTensorBasicExtractor",
    "QTensorSepParamsWithPack",
    "QZPScalePerChannel",
    "QZPScalePerGroup",
    "QZPScaleScalar",
]

try:  # noqa: C901
    from torch_to_nnef.qtensor.gguf import QTensorGGUF, QTensorGGUFExtractor

    __all__ += ["QTensorGGUF", "QTensorGGUFExtractor"]
except ImportError as exp:
    # feature gate: gguf_dtype
    print(exp)
