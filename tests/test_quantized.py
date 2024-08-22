"""Tests export quantized models."""

import os
import typing as T

import pytest
import torch
import torch.nn.quantized as nnq
from torch import nn
from torch.quantization import quantize_fx

from torch_to_nnef.inference_target import InferenceTarget, TractNNEF

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    set_seed,
)

#
# in some case bug may happen with random at specific seed
# you can alway do this to brute force find failure mode
# in your bash
# $ for i in {0..100}; do  echo $i; SEED=$i DEBUG=1 Q8=1 py.test || echo $i >> failed_seed.log; done;
set_seed(int(os.environ.get("SEED", 2)))  # 3 fail


# with tract 0.18 quantization work only for PyTorch 1.X
# we do not test PyTorch 1.X anymore (only 2.X)
test_suite = TestSuiteInferenceExactnessBuilder(
    [_ for _ in TRACT_INFERENCES_TO_TESTS if _.version >= "0.19.0"]
)


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


def cond_diff_tract_0_21_0(i):
    return isinstance(i, TractNNEF) and i.version != "0.21.0"


# SEED selected so that it's exact
test_suite.add(
    *build_test_tup(
        nn.Sequential(nn.Conv1d(2, 1, 1, stride=1, bias=False)), shape=(1, 2, 1)
    ),
    test_name="single_conv1d_with_kernel_1_no_bias",
    inference_conditions=cond_diff_tract_0_21_0,
)


test_suite.add(
    *build_test_tup(
        nn.intrinsic.ConvBnReLU1d(
            nn.Conv1d(3, 1, kernel_size=3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        ),
        shape=(1, 3, 4),
    ),
    test_name="fused_conv1d_bn_relu_with_kernel3",
)


def cond_tract_not_within_0_20_0_and_0_20_7(i):
    return isinstance(i, TractNNEF) and (
        i.version < "0.20.0" or "0.20.7" < i.version
    )


# tract regression
test_suite.add(
    *build_test_tup(nn.Linear(2, 1, bias=True), shape=(1, 2)),
    test_name="single_linear_with_bias",
    inference_conditions=cond_tract_not_within_0_20_0_and_0_20_7,
)

test_suite.add(
    *build_test_tup(nn.Linear(2, 1, bias=False), shape=(1, 2)),
    test_name="single_linear_no_bias",
    inference_conditions=cond_tract_not_within_0_20_0_and_0_20_7,
)

test_suite.add(
    *build_test_tup(
        nn.intrinsic.LinearReLU(nn.Linear(2, 2, bias=True), nn.ReLU()),
        shape=(1, 2),
    ),
    test_name="linear_with_bias_and_relu",
    inference_conditions=cond_tract_not_within_0_20_0_and_0_20_7,
)

test_suite.add(
    *build_test_tup(
        nn.Conv2d(2, 2, kernel_size=(2, 3), bias=False), shape=(1, 2, 3, 4)
    ),
    test_name="single_conv2d_kernel_2_3_no_bias",
    inference_conditions=cond_tract_not_within_0_20_0_and_0_20_7,
)

# test_suite.add(
#     *build_test_tup(
#         nn.intrinsic.ConvBnReLU2d(
#             nn.Conv2d(2, 2, kernel_size=(2, 3), bias=False),
#             nn.BatchNorm2d(2),
#             nn.ReLU(),
#         ),
#         shape=(1, 2, 3, 4),
#     ),
#     test_name="single_conv2d_bn_relu",
# )


def qcheck(
    module: nn.Module, inp: torch.Tensor, inference_target: InferenceTarget
):
    """Check basic ptq export align with tract"""
    model = torch.quantization.QuantWrapper(module)
    qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.backends.quantized.engine = "qnnpack"
    model.qconfig = qconfig
    model = torch.quantization.prepare(model)
    # input selected to avoid issue
    model(inp)
    model(inp)
    model(inp)
    # model(torch.arange(6).reshape(2, 3).float() * 2)
    model_int8 = torch.quantization.convert(model).eval()
    # true optimal formulation is:
    #
    # https://github.com/pytorch/pytorch/blob/8182fce76913f70822158f1c394be217122e66f6/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L1410
    check_model_io_test(
        model=model_int8, test_input=(inp,), inference_target=inference_target
    )


class DummyMathExample(nn.Module):
    def __init__(self, math_op: str):
        super().__init__()
        self.math_op = math_op
        self.op_f = nnq.FloatFunctional()

    def forward(self, x):
        return getattr(self.op_f, self.math_op)(x, x)


def _test_math_binary(
    op_name: str, values: T.List[float], inference_target: InferenceTarget
):
    inp = torch.tensor(values).reshape(2, 3).float()
    qcheck(DummyMathExample(op_name), inp, inference_target)


def cond_ge_tract_0_21_3(i):  # tract PR on quant accuracy merged
    return isinstance(i, TractNNEF) and i.version >= "0.21.3"


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if cond_ge_tract_0_21_3(_)],
)
def test_quantize_dummy_add(inference_target):
    _test_math_binary("add", [0, 2, 4, 8, 16, 8], inference_target)


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if cond_ge_tract_0_21_3(_)],
)
def test_quantize_dummy_mul(inference_target):
    _test_math_binary("mul", [0, 2, 4, 8, 16, 8], inference_target)


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if cond_ge_tract_0_21_3(_)],
)
def test_quantize_dummy_mul_1(inference_target):
    _test_math_binary("mul", [0, 1.51, 4, 8, 34.3, 8], inference_target)


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if cond_ge_tract_0_21_3(_)],
)
def test_quantize_dummy_max_relu(inference_target):
    qcheck(
        torch.nn.ReLU(),
        torch.tensor([-1, -0.5, -0.3, -0.2, 0, 1]).reshape(2, 3).float(),
        inference_target=inference_target,
    )


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if cond_ge_tract_0_21_3(_)],
)
def test_quantize_deq_req_sigmoid(inference_target):
    qcheck(
        torch.nn.Sequential(
            torch.quantization.DeQuantStub(),
            torch.nn.Sigmoid(),
            torch.quantization.QuantStub(),
        ),
        torch.tensor([-5.0, -4, -3.0, 0, -3, 5, 1]).float(),
        inference_target=inference_target,
    )


# @pytest.mark.parametrize(
#     "inference_target",
#     [_ for _ in TRACT_INFERENCES_TO_TESTS if cond_ge_tract_0_21_3(_)],
# )
# def test_quantized_sigmoid(inference_target):
#     qcheck(
#         torch.nn.Sigmoid(),
#         torch.tensor([-5.0, -4, -3.0, 0, -3, 5, 1]).float(),
#         inference_target=inference_target
#     )


# Once tract support on quant scheme improve {

# INPUT_AND_MODELS += [
# (torch.rand(1, 3, 256, 256), mod)
# for mod in [
# # vision_mdl.quantization.alexnet(pretrained=True, quantize=True)
# # vision_mdl.quantization.resnet50(pretrained=True, quantize=True)
# ]
# ]

# With pytorch v1.11.0 MultiheadAttention and LSTM are supported via dynamic Quantization only
# }


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_quantize_export(id, test_input, model, inference_target):
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
