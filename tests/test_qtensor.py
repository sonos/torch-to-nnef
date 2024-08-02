import pytest
import torch
from torch import nn

from torch_to_nnef.qtensor import replace_nn_ops
from torch_to_nnef.qtensor.qtract import QTensorTractScaleOnly
from torch_to_nnef.tract import tract_version

from .utils import check_model_io_test

if tract_version() < "0.21.6":
    pytest.skip(allow_module_level=True)


def test_quantize_with_tract_q4_0_basic():
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = 0.0
        model.weight[0:5, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = QTensorTractScaleOnly.build_q4_0_from_min_max_calibration(
            original_weight
        )
        deq_weights = q_tensor.to_torch_float_tensor()
        diff = (original_weight - deq_weights).abs()
        assert diff.sum() == 0

        model = replace_nn_ops(model, q_tensor)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.sum() == 0
        check_model_io_test(model=model, test_input=test_input)


def test_quantize_with_tract_q4_0_classic():
    """basic quantization values"""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = QTensorTractScaleOnly.build_q4_0_from_min_max_calibration(
            original_weight
        )
        deq_weights = q_tensor.to_torch_float_tensor()
        diff = (original_weight - deq_weights).abs()
        assert diff.mean() < 0.005, diff.mean()

        model = replace_nn_ops(model, q_tensor)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.mean() < 0.005, diff.mean()
        check_model_io_test(model=model, test_input=test_input)
