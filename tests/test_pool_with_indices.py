import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.utils import check_model_io_test
from torch_to_nnef.inference_target.tract import TractNNEF


class MaxPool2dWithIndices(nn.Module):
    def forward(self, x):
        y, idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        return y, idx


class MaxPool3dWithIndices(nn.Module):
    def forward(self, x):
        y, idx = F.max_pool3d(x, kernel_size=2, stride=2, return_indices=True)
        return y, idx


def test_max_pool2d_with_indices_export():
    model = MaxPool2dWithIndices().eval()
    data = torch.rand(1, 3, 8, 8)
    # Use Tract to validate indices output
    inference_target = TractNNEF.latest()
    check_model_io_test(model, data, inference_target)


def test_max_pool3d_with_indices_export():
    model = MaxPool3dWithIndices().eval()
    data = torch.rand(1, 2, 8, 8, 8)
    inference_target = TractNNEF.latest()
    check_model_io_test(model, data, inference_target)
