import typing as T

from nnef_tools.model import Tensor as NTensor
from nnef_tools.model import Operation as NOperation

from torch_to_nnef.op.base import ModuleInfoExtractor

__all__ = ["ReLUExtractor"]


class ReLUExtractor(ModuleInfoExtractor):
    JIT_CLASS_NAME = "__torch__.torch.nn.modules.activation.ReLU"

    def extract_operations(
        self, name_to_tensor: T.Dict[str, NTensor], input_name: T.Optional[str]
    ):
        if input_name is None:
            input_name = self.node.export_inputs[1]
        output_tensor = self._add_idem_type_shape_output_tensor(name_to_tensor)
        NOperation(
            graph=self.g,
            type="relu",
            name=f"{self.node.export_name}_relu",
            inputs=name_to_tensor[input_name],
            outputs=output_tensor,
        )
