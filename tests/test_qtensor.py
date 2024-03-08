import typing as T

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_to_nnef.qtensor import QTensor, TargetDType

from .utils import check_model_io_test


class WeightInputedConv1d(nn.Module):
    def __init__(self, conv_nn: nn.Conv1d):
        super().__init__()
        for attr in [
            "bias",
            "padding_mode",
            "stride",
            "dilation",
            "padding",
            "groups",
            "_reversed_padding_repeated_twice",
        ]:
            setattr(self, attr, getattr(conv_nn, attr))

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: T.Optional[Tensor]
    ):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: Tensor, weight: Tensor) -> Tensor:
        return self._conv_forward(input, weight, self.bias)


class QWeightedOp(nn.Module):
    def __init__(self, mod: nn.Module, weight_mod: QTensor):
        super().__init__()
        self.mod = mod
        self.weight_mod = weight_mod

    def forward(self, x: Tensor) -> Tensor:
        return self.mod(x, self.weight_mod())


def test_quantize_with_q_tensor():
    """Test simple models"""
    engine = "qnnpack"
    test_input = torch.rand(10, 2, 1)
    model = nn.Conv1d(2, 1, 1, stride=1, bias=False).eval()
    qconfig = torch.quantization.get_default_qconfig(engine)
    torch.backends.quantized.engine = engine
    model.qconfig = qconfig

    model_fp32_prepared = torch.quantization.prepare(
        torch.quantization.QuantWrapper(model).eval()
    )
    model_fp32_prepared(test_input)
    q_model = torch.quantization.convert(model_fp32_prepared).eval()
    weight = q_model.module.weight()
    q_weight = QTensor.from_torch_qtensor(
        weight, target_dtype=TargetDType(torch.float32)
    )

    model = QWeightedOp(WeightInputedConv1d(model), q_weight)
    check_model_io_test(model=model, test_input=test_input)
    pass
