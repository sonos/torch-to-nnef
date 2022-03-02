import typing as T

import numpy as np
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.op.base import ModuleInfoExtractor


__all__ = ["QuantizeExtractor"]


class QuantizeExtractor(ModuleInfoExtractor):

    JIT_CLASS_NAME = "__torch__.torch.nn.quantized.modules.Quantize"

    def extract_operations(self, name_to_tensor: T.Dict[str, NTensor]):
        output_tensor = self._add_idem_type_shape_output_tensor(
            name_to_tensor, force_dtype=np.int8
        )
        # TODO
        # Need 3 distinct operations
        # round((x/scale)+zero_point)
        NOperation(
            graph=self.g,
            type="quantize",
            name=f"{self.node.export_name}_",
            inputs=name_to_tensor[self.node.export_inputs[1]],
            outputs=output_tensor,
        )
