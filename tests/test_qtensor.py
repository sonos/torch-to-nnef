import torch
from torch import nn

from torch_to_nnef.qtensor import QTensor, TargetDType, replace_nn_ops

from .utils import check_model_io_test


def test_quantize_with_q_tensor_basic():
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
    model = replace_nn_ops(model, q_weight)
    check_model_io_test(model=model, test_input=test_input)


def test_quantize_with_q_tensor_per_channel():
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
    q_weight.qscheme = q_weight.qscheme.to_zpscale_per_channel(
        q_weight.packed_torch_tensor
    )
    model = replace_nn_ops(model, q_weight)
    check_model_io_test(model=model, test_input=test_input)
