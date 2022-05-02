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
set_seed(int(os.environ.get("SEED", 2)))  # 3 fail


class WithQuantDeQuant(torch.quantization.QuantWrapper):
    @classmethod
    def quantize_model_and_stub(
        cls, model, input_shape, representative_data, safe_margin_percents=200
    ):
        model = cls(model)
        # pylint: disable-next=attribute-defined-outside-init
        model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        model_qat = torch.quantization.prepare_qat(model)
        model_qat.train()

        dmax = representative_data.max()
        dmin = representative_data.min()
        drange = (dmax - dmin).abs()

        # add safe margin %
        dmax += drange / 100 * safe_margin_percents
        dmin -= drange / 100 * safe_margin_percents
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


def build_test_tup(mod, shape=(1, 2, 1), safe_margin_percents=200):
    reduce_r = 1
    for si in shape:
        reduce_r *= si
    return (
        torch.arange(reduce_r).reshape(*shape).float(),
        WithQuantDeQuant.quantize_model_and_stub(
            mod,
            input_shape=shape,
            representative_data=torch.arange(reduce_r).reshape(*shape).float(),
            safe_margin_percents=safe_margin_percents,
        ),
    )


# SEED selected so that it works.
INPUT_AND_MODELS += [
    build_test_tup(mod, shape=(1, 2, 1))
    for mod in [
        nn.Sequential(nn.Conv1d(2, 1, 1, stride=1, bias=False)),
    ]
]

INPUT_AND_MODELS += [
    build_test_tup(mod, shape=(1, 3, 4))
    for mod in [
        nn.intrinsic.ConvBnReLU1d(
            nn.Conv1d(3, 1, kernel_size=3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        ),
    ]
]


INPUT_AND_MODELS += [
    build_test_tup(mod, shape=(1, 2))
    for mod in [
        nn.Linear(2, 1, bias=False),
        nn.Linear(2, 1, bias=True),
        nn.intrinsic.LinearReLU(nn.Linear(2, 1), nn.ReLU()),
    ]
]

INPUT_AND_MODELS += [
    build_test_tup(mod, shape=(1, 2, 3, 4))
    for mod in [
        nn.Conv2d(2, 2, kernel_size=(2, 3), bias=False),
        # nn.intrinsic.ConvBnReLU2d(
        # nn.Conv2d(2, 2, kernel_size=(2, 3), bias=False),
        # nn.BatchNorm2d(2),
        # nn.ReLU(),
        # ),
    ]
]


# INPUT_AND_MODELS += [
# (torch.rand(1, 3, 256, 256), mod)
# for mod in [
# # vision_mdl.quantization.alexnet(pretrained=True, quantize=True)
# # vision_mdl.quantization.resnet50(pretrained=True, quantize=True)
# ]
# ]

# torch.nn.quantizable.LSTM??

# To Support ?
# torch.nn.quantizable.MultiheadAttention??

# INPUT_AND_MODELS += [
# build_test_tup(mod, shape=(1, 3, 4))
# for mod in [nn.intrinsic.ConvBnReLU2d()]
# ]

# torch.nn.quantizable.LSTM??

# to support ?
# torch.nn.quantizable.MultiheadAttention??


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_quantize_export(test_input, model):
    """Test simple models"""
    _test_check_model_io(model=model, test_input=test_input)
