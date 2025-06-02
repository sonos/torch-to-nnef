from pathlib import Path
import tempfile
import pytest
import torch

from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant.qtract import (
    fp_to_tract_q4_0_with_min_max_calibration,
)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_offload_tensor_export_with_tract_and_conv2d(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        k = 3
        y = 8
        in_size = 128
        multiplier = 2
        chan_size = k * multiplier
        test_input = torch.arange(in_size * k * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, k, chan_size)
        ker = (k, k)
        model = torch.nn.Conv2d(in_size, y, kernel_size=ker).eval()

        original_weight = (
            torch.arange(in_size * y * k * k).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        with tempfile.TemporaryDirectory() as td:
            offloaded_value = OffloadedTensor.from_real_tensor(
                original_weight, "my_offloaded_weight", offload_dir=Path(td)
            )
            # offloaded_value = original_weight
            model.weight = torch.nn.Parameter(
                offloaded_value, requires_grad=False
            )
            check_model_io_test(
                model=model,
                test_input=test_input,
                inference_target=inference_target,
            )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_offload_qtensor_export(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        k = 3
        y = 8
        in_size = 128
        multiplier = 2
        chan_size = k * multiplier
        test_input = torch.arange(in_size * k * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, k, chan_size)
        ker = (k, k)
        model = torch.nn.Conv2d(in_size, y, kernel_size=ker).eval()

        original_weight = (
            torch.arange(in_size * y * k * k).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        with tempfile.TemporaryDirectory() as td:
            q_tensor = fp_to_tract_q4_0_with_min_max_calibration(
                original_weight
            )
            offloaded_value = OffloadedTensor.from_real_tensor(
                q_tensor, "my_offloaded_weight", offload_dir=Path(td)
            )
            # offloaded_value = original_weight
            model.weight = torch.nn.Parameter(
                offloaded_value, requires_grad=False
            )
            check_model_io_test(
                model=model,
                test_input=test_input,
                inference_target=inference_target,
            )
