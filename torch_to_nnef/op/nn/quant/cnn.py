import typing as T

import numpy as np
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.op.nn.activation import ReLUExtractor
from torch_to_nnef.op.base import ModuleInfoExtractor

__all__ = ["QConvReLU1dExtractor"]


class QConvReLU1dExtractor(ModuleInfoExtractor):
    JIT_CLASS_NAME = (
        "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU1d"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nnef_weight_ref = None
        self.nnef_bias_ref = None

    def extract_extra_tensor_from_module(
        self, name_to_tensors: T.Dict[str, NTensor]
    ):
        self.nnef_weight_ref = self.add_quantized_tensor_to_ngraph(
            self.mod.weight(), "weight", name_to_tensors
        )
        bias = self.mod.bias()
        if bias is not None:
            # Warning ! Pytorch does not quantize bias at training time
            # https://github.com/pytorch/pytorch/issues/28952
            # since no clear expectation, we use same scale & zero_point as
            # weight
            qbias = self._quant_from_ref(bias, self.mod.weight())
            self.nnef_bias_ref = self.add_quantized_tensor_to_ngraph(
                qbias, "bias", name_to_tensors
            )
        return name_to_tensors

    def extract_operations(self, name_to_tensor):
        out_tensor_name = f"{self.node.export_name}_conv"
        output_tensor = NTensor(
            graph=self.g,
            name=out_tensor_name,
            dtype=np.int8,
            shape=tuple(self.node.tensor_size)
            if self.node.tensor_size
            else None,
        )
        name_to_tensor[out_tensor_name] = output_tensor

        weight_var = NOperation(
            graph=self.g,
            type="variable",
            name=f"{self.node.export_name}_weight_var",
            inputs=None,
            outputs=self.nnef_weight_ref,
            attribs={
                "label": self.nnef_weight_ref.name,
                "shape": list(self.nnef_weight_ref.shape),
                "dtype": self.nnef_weight_ref.dtype,
            },
        )
        bias_var = NOperation(
            graph=self.g,
            type="variable",
            name=f"{self.node.export_name}_bias_var",
            inputs=None,
            outputs=self.nnef_bias_ref,
            attribs={
                "label": self.nnef_bias_ref.name,
                "shape": list(self.nnef_bias_ref.shape),
                "dtype": self.nnef_weight_ref.dtype,
            },
        )
        NOperation(
            graph=self.g,
            type="conv",
            name=f"{self.node.export_name}_op",
            inputs=(
                name_to_tensor[self.node.export_inputs[1]],
                weight_var.output,
                bias_var.output,
            ),
            outputs=output_tensor,
            attribs={
                "dilation": list(self.mod.dilation),
                "padding": [(self.mod.padding[0], 0)],
                "stride": list(self.mod.stride),
                "groups": self.mod.groups,
                "border": "constant",
            },
        )
        ReLUExtractor.extract_operations(
            self, name_to_tensor, input_name=output_tensor.name
        )
