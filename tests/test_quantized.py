"""Tests export quantized models."""
import os

import pytest
import torch
from torch import nn

from .utils import _test_check_model_io, set_seed  # noqa: E402

#
# in some case bug may happen with random at specific seed
# you can alway do this to brute force find failure mode
# in your bash
# $ for i in {0..100}; do  echo $i; SEED=$i DEBUG=1 Q8=1 py.test || echo $i >> failed_seed.log; done;
set_seed(int(os.environ.get("SEED", 25)))


class WithQuantDeQuant(torch.quantization.QuantWrapper):
    @classmethod
    def quantize_model_and_stub(
        cls, model, input_shape, representative_data=None
    ):
        model = cls(model)
        # pylint: disable-next=attribute-defined-outside-init
        model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        model_qat = torch.quantization.prepare_qat(model)
        model_qat.train()
        scale = 1.0
        offset = 0.0
        if representative_data is not None:
            dmax = representative_data.max()
            dmin = representative_data.min()
            drange = (dmax - dmin).abs()
            # add 10% safe margin
            dmax += drange / 100 * 200
            dmin -= drange / 100 * 200
            scale = (dmax - dmin).abs()
            offset = dmin
        for _ in range(100):
            model_qat(torch.rand(input_shape) * scale + offset)
        model_q8 = torch.quantization.convert(model_qat.eval())
        return model_q8

    def forward(self, x):
        x = self.quant(x)
        x = self.module(x)
        return x


# Test with quantization
INPUT_AND_MODELS = []
if os.environ.get("Q8"):
    # my_shape = [1, 2, 2]
    my_shape = [1, 256, 10]
    reduce_r = 1
    for si in my_shape:
        reduce_r *= si
    INPUT_AND_MODELS += [
        (
            torch.arange(reduce_r).reshape(*my_shape).float(),
            WithQuantDeQuant.quantize_model_and_stub(
                mod,
                input_shape=my_shape,
                representative_data=torch.arange(reduce_r)
                .reshape(*my_shape)
                .float(),
            ),
        )
        for mod in [
            nn.Sequential(
                nn.Conv1d(
                    my_shape[1], 1, min(my_shape[2], 3), stride=1, bias=False
                ),
                # nn.intrinsic.ConvBnReLU1d(
                # nn.Conv1d(10, 20, 3),
                # nn.BatchNorm1d(20),
                # nn.ReLU(),
                # ),
                # nn.intrinsic.ConvBnReLU1d(
                # nn.Conv1d(20, 15, 5, stride=2), nn.BatchNorm1d(15), nn.ReLU()
                # ),
                # nn.intrinsic.ConvBnReLU1d(
                # nn.Conv1d(15, 50, 7, stride=3, padding=3),
                # nn.BatchNorm1d(50),
                # nn.ReLU(),
                # ),
            ),
        ]
    ]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_quantize_export(test_input, model):
    """Test simple models"""
    _test_check_model_io(model=model, test_input=test_input)
