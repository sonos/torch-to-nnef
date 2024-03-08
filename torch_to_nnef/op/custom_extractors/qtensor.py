# from nnef_tools.model import Tensor as NTensor
# from torch import nn

from torch_to_nnef.exceptions import StrictNNEFSpecError
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor import QTensor


class QTensorExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensor

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        nnef_spec_strict: bool,
        **kwargs,
    ):
        if nnef_spec_strict:
            raise StrictNNEFSpecError(
                "Impossible to export QTensor with NNEF spec compliance activated"
            )

        # qtensor = node.op_ref
        # TODO: export based on Code DOC
        __import__("ipdb").set_trace()
