from copy import deepcopy
import platform
import time
from datetime import datetime

import pytest
import torch
from torch import nn

from torch_to_nnef.qtensor.base import (
    U8Compressor,
    qscale_per_group_f16_min_max_calibration,
)
from torch_to_nnef.qtensor.qtract import (
    QTensorTractScaleOnly,
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.inference_target.tract import TractCheckTolerance

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TRACT_INFERENCES_TO_TESTS_EXACT,
    check_model_io_test,
)


def test_quantize_with_tract_q4_0_and_manipulate_tensor():
    original_weight = torch.arange(64).reshape(2, 32).float()
    q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
    q_tensor.u8_blob  # check access works
    new_q_tensor = q_tensor.to(torch.float32)
    new_q_tensor.u8_blob  # check access works

    class Test(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(new_q_tensor, requires_grad=False)

        def forward(self, x):
            return x @ self.weight

    mod = Test()
    assert mod.weight.u8_blob.dtype == torch.uint8
    # mod.weight.data.u8_blob

    inp_tensor = torch.rand(4, 2)
    out = mod(inp_tensor)
    assert type(out) is type(inp_tensor)  # avoid propagation of qtype


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_basic(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = 0.0
        model.weight[0:5, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.decompress()
        diff = (original_weight - deq_weights).abs()
        assert diff.sum() == 0

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.sum() == 0
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_controled(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.decompress()
        diff = (original_weight - deq_weights).abs()
        assert diff.mean() < 0.01, diff.mean()

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.mean() < 0.01, diff.mean()
        if "arm" in platform.uname().machine.lower():
            inference_target = deepcopy(inference_target)
            inference_target.check_io_tolerance = TractCheckTolerance.SUPER

        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_rounding2(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        nd = 32
        test_input = torch.rand(nd).float().reshape(1, nd)
        model = nn.Linear(nd, 2, bias=False).eval()
        original_weight = model.weight
        model.weight[:, :] = 0.0
        model.weight[0, 0:5] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.decompress()
        diff = (original_weight - deq_weights).abs()
        assert diff.mean() < 0.03, diff.mean()

        model.weight = nn.Parameter(q_tensor, requires_grad=False)

        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_arange(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        if "arm" in platform.uname().machine.lower():
            inference_target = deepcopy(inference_target)
            inference_target.check_io_tolerance = TractCheckTolerance.SUPER

        test_input = torch.arange(960).float().reshape(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = torch.arange(16 * 96).float().reshape(16, 96)
        original_weight = model.weight

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        # can safely check io since all values controled
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )


class DummyU8Compressor(U8Compressor):
    def __init__(self):
        self.compress_times = []
        self.decompress_times = []

    def compress(self, u8_tensor) -> torch.Tensor:
        time.sleep(0.01)
        self.compress_times.append(datetime.now())
        return u8_tensor

    def decompress(self, u8_tensor) -> torch.Tensor:
        time.sleep(0.01)
        self.decompress_times.append(datetime.now())
        return u8_tensor


def test_u8_compressors():
    fp_tensor = torch.rand(2, 32)
    with torch.no_grad():
        q_scheme = qscale_per_group_f16_min_max_calibration(
            fp_tensor, n_bits=4, group_size=32, percentile=1
        )
        dummy_compressor1 = DummyU8Compressor()
        dummy_compressor2 = DummyU8Compressor()
        qtensor = QTensorTractScaleOnly(
            fp_tensor,
            qscheme=q_scheme,
            dequant_to_dtype=fp_tensor.dtype,
            u8_compressors=[dummy_compressor1, dummy_compressor2],
        )
        # compressed
        assert len(dummy_compressor1.compress_times) == 1
        assert len(dummy_compressor2.compress_times) == 1
        assert (
            dummy_compressor1.compress_times[0]
            < dummy_compressor2.compress_times[0]
        )
        # decompressed
        assert len(dummy_compressor1.decompress_times) == 1
        assert len(dummy_compressor2.decompress_times) == 1
        assert (
            dummy_compressor1.decompress_times[0]
            > dummy_compressor2.decompress_times[0]
        )
        # test
        torch.matmul(torch.rand(1, 2), qtensor)
        assert len(dummy_compressor1.compress_times) == 1
        assert len(dummy_compressor2.compress_times) == 1
        # decompressed
        assert len(dummy_compressor1.decompress_times) == 2
        assert len(dummy_compressor2.decompress_times) == 2
        assert (
            dummy_compressor1.decompress_times[1]
            > dummy_compressor2.decompress_times[1]
        )


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_assign_to(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.arange(960).float().reshape(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = torch.arange(16 * 96).float().reshape(16, 96)
        original_weight = model.weight

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version >= "0.21.11"],
)
def test_quantize_with_tract_q4_0_embedding(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        x = 6
        y = 32
        model = nn.Embedding(x, y).eval()
        original_weight = (torch.arange(x * y).reshape(x, y).float() * 2).half()

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )
