"""Tests dynamic_axes."""

import os
from functools import partial

import pytest
import torch
from torch import nn
from torchaudio import models as audio_mdl

from torch_to_nnef.inference_target import TractNNEF

from .test_primitive import TorchFnPrimitive
from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS,
    TestSuiteInferenceExactnessBuilder,
    change_dynamic_axes,
    check_model_io_test,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))


test_suite = TestSuiteInferenceExactnessBuilder(TRACT_INFERENCES_TO_TESTS)


dyn_stream_axis2 = {"input_0": {2: "S"}}
dyn_stream_axis3 = {"input_0": {3: "S"}}

test_suite.add(
    torch.rand(1, 1, 100, 64),
    audio_mdl.DeepSpeech(64, n_hidden=256),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)


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


test_suite.add(
    torch.rand(1, 2, 3),
    MimicShapeOut(torch.ones),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

test_suite.add(
    torch.rand(1, 10, 3),
    MimicShapeOut(torch.ones),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

test_suite.add(
    torch.rand(1, 4, 3),
    MimicShapeOut(partial(torch.full, fill_value=5)),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)


def ge_tract_0_21_5(i):
    return isinstance(i, TractNNEF) and i.version > "0.21.5"


test_suite.add(
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]),
    TorchFnPrimitive("repeat_interleave", opt_kwargs={"repeats": 3, "dim": 2}),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

test_suite.add(
    torch.tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]]]).float(),
    LambdaOp(lambda x: x[..., : x.shape[-1] // 2]),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis3
    ),
)

test_suite.add(
    torch.tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]]]).float(),
    LambdaOp(lambda x: x[..., x.shape[-1] // 2 :]),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis3
    ),
)
test_suite.add(
    torch.rand(2, 1),
    LambdaOp(lambda x: x[:, -3:]),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

test_suite.add(
    torch.rand(2, 1),
    LambdaOp(lambda x: x[:, :1000]),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_dynamic_axes_exports(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
