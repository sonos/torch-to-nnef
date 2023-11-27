"""Tests dynamic_axes."""
import os
from functools import partial

import pytest
import torch
from torch import nn
from torchaudio import models as audio_mdl

from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))

INPUT_AND_MODELS = [
    (torch.rand(1, 1, 100, 64), {2: "S"}, model)
    for model in [
        audio_mdl.DeepSpeech(64, n_hidden=256),
    ]
]


class MimicShapeOut(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        xsh = x.shape
        return self.op(xsh, dtype=torch.float32)


INPUT_AND_MODELS += [
    (torch.rand(1, 2, 3), {2: "S"}, MimicShapeOut(torch.ones)),
    (torch.rand(1, 10, 3), {2: "S"}, MimicShapeOut(torch.ones)),
    (
        torch.rand(1, 4, 3),
        {2: "S"},
        MimicShapeOut(partial(torch.full, fill_value=5)),
    ),
]


@pytest.mark.parametrize("test_input,dyn_shapes,model", INPUT_AND_MODELS)
def test_tricky_export(test_input, dyn_shapes, model):
    """Test simple models"""
    check_model_io_test(
        model=model,
        test_input=test_input,
        dynamic_axes={"input_0": dyn_shapes},
    )
