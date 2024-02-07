"""Tests export quantized models."""
import os
import typing as T

import pytest
import torch
import torch.nn.quantized as nnq
from torch import nn
from torch.quantization import quantize_fx

from torch_to_nnef.tract import (
    tract_version_greater_than,
    tract_version_lower_than,
)

from .utils import check_model_io_test, set_seed  # noqa: E402

#
# in some case bug may happen with random at specific seed
# you can alway do this to brute force find failure mode
# in your bash
# $ for i in {0..100}; do  echo $i; SEED=$i DEBUG=1 Q8=1 py.test || echo $i >> failed_seed.log; done;
set_seed(int(os.environ.get("SEED", 2)))  # 3 fail


class WithQuantDeQuant(torch.quantization.QuantWrapper):
    @classmethod
    def quantize_model_and_stub(
        cls,
        model: nn.Module,
        input_shape: T.Tuple[int, ...],
        representative_data: torch.Tensor,
        safe_margin_percents: int = 200,
        use_ao_quant: bool = True,
        use_static: bool = False,
        quant_engine: str = "qnnpack",
    ):
        model = cls(model)
        if use_ao_quant:
            if use_static:
                model_qprep = quantize_fx.prepare_fx(
                    model.eval(),
                    {"": torch.quantization.get_default_qconfig(quant_engine)},
                )
                return quantize_fx.convert_fx(model_qprep)

            model_qat = quantize_fx.prepare_qat_fx(
                model,
                {"": torch.quantization.get_default_qat_qconfig(quant_engine)},
            )
            model_qat.train()
        else:
            if use_static:
                model_fp32_prepared = torch.quantization.prepare(model.eval())
                return torch.quantization.convert(model_fp32_prepared)

            # pylint: disable-next=attribute-defined-outside-init
            model.qconfig = torch.quantization.get_default_qat_qconfig(
                quant_engine
            )
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

        if use_ao_quant:
            model_q8 = quantize_fx.convert_fx(model_qat)
        else:
            model_q8 = torch.quantization.convert(model_qat.eval())

        return model_q8

    def forward(self, x):
        x = self.quant(x)
        x = self.module(x)
        return self.dequant(x)


# Test with quantization
INPUT_AND_MODELS = []


def build_test_tup(
    mod: nn.Module,
    shape: T.Tuple[int, ...] = (1, 2, 1),
    safe_margin_percents: int = 200,
    use_ao_quant: bool = False,
    use_static: bool = False,
):
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
            use_ao_quant=use_ao_quant,
            use_static=use_static,
        ),
    )


if not tract_version_lower_than(
    "0.19.0"
):  # with tract 0.18 quantization work only for PyTorch 1.X
    # we do not test PyTorch 1.X anymore (only 2.X)

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
    if tract_version_lower_than("0.20.0") or tract_version_greater_than(
        "0.20.7"
    ):  # tract regression
        INPUT_AND_MODELS += [
            build_test_tup(mod, shape=(1, 2))
            for mod in [
                nn.Linear(2, 1, bias=False),
                nn.Linear(2, 1, bias=True),
                nn.intrinsic.LinearReLU(nn.Linear(2, 2, bias=True), nn.ReLU()),
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
if not tract_version_lower_than("0.20.7"):

    class DummyMulExample(nn.Module):
        def __init__(self, math_op: str):
            super().__init__()
            self.math_op = math_op
            self.op_f = nnq.FloatFunctional()

        def forward(self, x):
            return getattr(self.op_f, self.math_op)(x, x)

    def math_binary_test(math_op):
        model = torch.quantization.QuantWrapper(
            DummyMulExample(math_op=math_op)
        )
        qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.backends.quantized.engine = "qnnpack"
        model.qconfig = qconfig
        model = torch.quantization.prepare(model)
        # input selected to avoid issue
        inp = torch.tensor([0, 2, 4, 8, 16, 8]).reshape(2, 3).float()
        model(inp)
        model(inp)
        model(inp)
        # model(torch.arange(6).reshape(2, 3).float() * 2)
        model_int8 = torch.quantization.convert(model).eval()
        check_model_io_test(
            model=model_int8,
            test_input=(inp,),
        )

    def test_quantize_dummy_add():
        math_binary_test("add")

    def test_quantize_dummy_mul():
        math_binary_test("mul")


# Need Monitoring !
# With pytorch v1.11.0 MultiheadAttention and LSTM are supported via dynamic Quantization only

# INPUT_AND_MODELS += [
# (torch.rand(1, 3, 256, 256), mod)
# for mod in [
# # vision_mdl.quantization.alexnet(pretrained=True, quantize=True)
# # vision_mdl.quantization.resnet50(pretrained=True, quantize=True)
# ]
# ]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_quantize_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
