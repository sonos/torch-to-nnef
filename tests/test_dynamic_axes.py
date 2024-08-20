"""Tests dynamic_axes."""

import os
from functools import partial

import pytest
import torch
from torch import nn
from torchaudio import models as audio_mdl

from torch_to_nnef.tract import tract_version

from .test_primitive import TorchFnPrimitive
from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))

INPUT_AND_MODELS = []
INPUT_AND_MODELS += [
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


class LambdaOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x)


INPUT_AND_MODELS += [
    (torch.rand(1, 2, 3), {2: "S"}, MimicShapeOut(torch.ones)),
    (torch.rand(1, 10, 3), {2: "S"}, MimicShapeOut(torch.ones)),
    (
        torch.rand(1, 4, 3),
        {2: "S"},
        MimicShapeOut(partial(torch.full, fill_value=5)),
    ),
]

if "0.21.5" < tract_version():
    INPUT_AND_MODELS += [
        (
            torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]),
            {2: "S"},
            TorchFnPrimitive(
                "repeat_interleave", opt_kwargs={"repeats": 3, "dim": 2}
            ),
        ),
    ]
    INPUT_AND_MODELS += [
        (
            # shape 1, 2, 1, 4
            torch.tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]]]).float(),
            {3: "S"},
            LambdaOp(lambda x: x[..., : x.shape[-1] // 2]),
        ),
    ]

# INPUT_AND_MODELS = [(torch.rand(2, 1), {2: "S"}, LambdaOp(lambda x: x[:, -3:]))]
# # TODO: should in such case have a max(inshape, 1000) before slice
# INPUT_AND_MODELS = [(torch.rand(2, 1), {2: "S"}, LambdaOp(lambda x: x[:, :1000]))]


@pytest.mark.parametrize("test_input,dyn_shapes,model", INPUT_AND_MODELS)
def test_tricky_export(test_input, dyn_shapes, model):
    """Test simple models"""
    check_model_io_test(
        model=model,
        test_input=test_input,
        dynamic_axes={"input_0": dyn_shapes},
    )
