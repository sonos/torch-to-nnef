"""Tests case where shape is defined by operations within graph."""

import os

import pytest
import torch
from torch import nn

from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))


class DynamicDoubleBatchRank(nn.Module):
    """Build view based on several dynamical ops

    It fails for now as we do not yet have a proper tracing/solver for chain
    of generated constant. It will be needed as view at inference time need to
    have fixed dimensions.

    """

    def forward(self, x):
        return x.view(x.shape[0] * 2, x.shape[1] // 2, x.shape[2])


class MultiOut(nn.Module):
    def forward(self, x):
        x1, x2 = x.split([5, 5], dim=1)
        return x1, x2


class SelectNotFirstOutput(nn.Module):
    """Select 2nd Ouput comming from another module.

    note this worked when operation directly applied to same module.
    We test this as it used to bug as some point.

    By default jit.trace generate at this level a graph with node like this:

    %17 : Tensor = prim::CallMethod[name="forward"](%multi_out, %x)

    There is no indication that %17 is in reality x2 ?

    """

    def __init__(self):
        super().__init__()
        self.multi_out = MultiOut()

    def forward(self, x):
        _, x2 = self.multi_out(x)
        return x2


class LostDimPad(nn.Module):
    """
    Simplified part with parser issue in swin rolling attention mechanism
    """

    def __init__(self):
        self.window_size = [1, 2]
        super().__init__()

    def forward(self, inp):
        # B, H, W, C
        _, _, W, _ = inp.shape
        pad_r = (
            self.window_size[1] - W % self.window_size[1]
        ) % self.window_size[1]
        x = nn.functional.pad(inp, (0, 0, 0, pad_r, 0, 0))
        return x


INPUT_AND_MODELS = [
    # to solve this we will need to expand fully graph constant at torch_graph module level
    (torch.rand(5, 10, 4), DynamicDoubleBatchRank()),
    (torch.rand(5, 10, 4), SelectNotFirstOutput()),
    (torch.rand(1, 3, 16, 16), LostDimPad()),
]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_tricky_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
