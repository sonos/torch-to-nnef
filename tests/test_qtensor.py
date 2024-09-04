import pytest
import torch
from torch import nn

from torch_to_nnef.qtensor.qtract import (
    fp_to_tract_q4_0_with_min_max_calibration,
)

from .utils import TRACT_INFERENCES_TO_TESTS, check_model_io_test


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
    assert type(out) == type(inp_tensor)  # avoid propagation of qtype


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if _.version > "0.21.6"],
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
        deq_weights = q_tensor.to_torch_float_tensor()
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
    [_ for _ in TRACT_INFERENCES_TO_TESTS if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_classic(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.to_torch_float_tensor()
        diff = (original_weight - deq_weights).abs()
        assert diff.mean() < 0.01, diff.mean()

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.mean() < 0.01, diff.mean()
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_arange(inference_target):
    """basic quantization values"""
    with torch.no_grad():
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
