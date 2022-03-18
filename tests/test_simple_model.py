"""Tests simple models."""

import os
import subprocess
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision import models as vision_mdl

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.tract import (
    build_io,
    debug_dumper_pytorch_to_onnx_to_nnef,
    tract_assert_io,
)

INPUT_AND_MODELS = []


class UnaryPrimitive(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x)


class BinaryPrimitive(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x1, x2):
        return self.op(x1, x2)


class TensorFnPrimitive(nn.Module):
    def __init__(self, op, kwargs, args=None):
        super().__init__()
        self.op = op
        self.args = args or tuple()
        self.kwargs = kwargs

    def forward(self, x):
        return getattr(x, self.op)(*self.args, **self.kwargs)


class WithQuantDeQuant(torch.quantization.QuantWrapper):
    @classmethod
    def quantize_model_and_stub(cls, model):
        model = cls(model)
        # pylint: disable-next=attribute-defined-outside-init
        model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        model_qat = torch.quantization.prepare_qat(model)
        model_qat.train()
        for _ in range(10):
            model_qat(torch.rand(1, 10, 100))
        model_q8 = torch.quantization.convert(model_qat.eval())
        return model_q8

    def forward(self, x):
        x = self.quant(x)
        x = self.module(x)
        return x


class ListInputPrim(nn.Module):
    def __init__(self, op, dims):
        super().__init__()
        self.y = torch.rand(*dims)
        self.op = op

    def forward(self, x):
        return self.op([x, self.y], dim=1)


def nnef_split(value, axis, ratios):
    assert value.shape[axis] % sum(ratios) == 0

    multiplier = value.shape[axis] // sum(ratios)
    sections = [ratio * multiplier for ratio in ratios]
    return torch.split(value, split_size_or_sections=sections, dim=axis)


# Base unary operations
_condition_1 = condition = torch.eye(5, 4).to(torch.bool)
_input0 = torch.zeros(5, 4)
INPUT_AND_MODELS = [
    (torch.arange(20).reshape(5, 4).float(), UnaryPrimitive(op))
    for op in [
        torch.sin,
        torch.cos,
        torch.exp,
        torch.log,
        torch.abs,
        torch.sign,
        torch.neg,
        torch.floor,
        torch.ceil,
        torch.round,
        torch.sqrt,
        torch.rsqrt,
        torch.log2,
        # unimplemented tract {
        # torch.tan,
        # torch.asin,
        # torch.acos,
        # torch.atan,
        # torch.sinh,
        # torch.cosh,
        # torch.tanh,
        # torch.asinh,
        # torch.acosh,
        # torch.atanh,
        # torch.reciprocal,
        # torch.clone,
        # partial(nn.functional.pad, pad=(0, 1), mode="replicate"),
        # }
        partial(torch.pow, exponent=2.0),
        partial(torch.pow, exponent=-2.0),
        # tract need reversed NNEF transpose.axes=[] order than spec
        partial(torch.transpose, dim0=1, dim1=0),
        # tract need reversed NNEF transpose.axes=[] order than spec
        partial(torch.permute, dims=[1, 0]),
        partial(torch.reshape, shape=(2, 5, 2)),
        partial(torch.unsqueeze, dim=1),
        partial(nn.functional.pad, pad=(1, 0), mode="reflect"),
        # lambda x: torch.where(
        # _condition_1,
        # input=_input0,
        # other=x,
        # ),
    ]
]
INPUT_AND_MODELS += [
    (torch.rand(13, 10, 1), UnaryPrimitive(op))
    for op in [
        partial(torch.squeeze, dim=2),
        TensorFnPrimitive("mean", {"dim": 1}),
        TensorFnPrimitive("mean", {"dim": 1, "keepdim": True}),
        TensorFnPrimitive("sum", {"dim": 1}),
        TensorFnPrimitive("max", {"dim": 1}),
        TensorFnPrimitive("min", {"dim": 1}),
        TensorFnPrimitive("argmax", {"dim": 1}),
        TensorFnPrimitive("argmin", {"dim": 1}),
        # TensorFnPrimitive(
        # "repeat", kwargs={}, args=([1, 2, 1],)
        # ),  # missing an s in repeat export to nnef since tract is false
        partial(
            nn.functional.pad,
            pad=[0, 0, 0, 0, 0, 1],
            mode="constant",
            value=0.0,
        ),
        partial(
            nn.functional.pad,
            pad=[2, 0, 0, 1],
            mode="constant",
            value=2.0,
        ),
    ]
]


INPUT_AND_MODELS += [
    (
        torch.tensor(
            [[True, False, True], [True, True, True], [False, False, False]]
        ),
        UnaryPrimitive(op),
    )
    for op in [
        # TensorFnPrimitive("any", {"dim": 1}),
        # TensorFnPrimitive("all", {"dim": 1}),
    ]
]

# _binary
INPUT_AND_MODELS += [
    ((torch.rand(13, 10), torch.rand(13, 10)), BinaryPrimitive(op))
    for op in [
        torch.min,
        torch.max,
        torch.sub,
        torch.add,
        torch.mul,
        torch.div,
        torch.pow,
        torch.less,
        torch.eq,
        torch.ne,
        torch.greater,
        torch.less_equal,
        torch.greater_equal,
    ]
]

INPUT_AND_MODELS += [
    ((torch.rand(13, 10), torch.rand(13, 10).T), BinaryPrimitive(op))
    for op in [
        # torch.matmul,  # tract not same results ??
    ]
]

INPUT_AND_MODELS += [
    (torch.tensor([True, False, True]), UnaryPrimitive(torch.bitwise_not))
]
INPUT_AND_MODELS += [
    (
        (torch.tensor([True, False, True]), torch.tensor([True, False, False])),
        BinaryPrimitive(op),
    )
    for op in [
        (lambda x, y: x & y),  # and
        (lambda x, y: x | y),  # or
    ]
]

# tract do not handle cast op
# INPUT_AND_MODELS += [
# (
# torch.tensor([True, False, True]),
# TensorFnPrimitive("to", {"dtype": torch.bool}),
# )
# ]


# Base Layers
INPUT_AND_MODELS += [
    (torch.rand(13, 10), layer)
    for layer in [
        nn.Linear(10, 20, bias=False),
        nn.Linear(10, 32),
    ]
]

INPUT_AND_MODELS += [
    (torch.rand(1, 3, 256, 256), layer)
    for layer in [
        nn.Conv2d(
            3,
            64,
            (3, 7),
        ),
        nn.Flatten(start_dim=1, end_dim=2),
        nn.Dropout(),
        nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False
        ),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=4, padding=0, dilation=2),
        nn.AdaptiveAvgPool2d(32),
    ]
]

INPUT_AND_MODELS += [
    (torch.rand(1, 10, 100), layer)
    for layer in [
        nn.Conv1d(10, 20, 3),
        nn.Conv1d(10, 20, 3, stride=2),
        nn.Conv1d(10, 20, 3, groups=10),
        nn.Conv1d(10, 20, 3, bias=False),
        nn.Conv1d(10, 20, 3, padding=3),
        nn.BatchNorm1d(10, eps=0, momentum=0.1),
        nn.MaxPool1d(10, stride=3, padding=2, dilation=1),
        # nn.AvgPool1d(10), # Not same results between tract and Pytorch
        nn.ConvTranspose1d(10, 20, 3),
        nn.ConvTranspose1d(10, 20, 3, padding=2, dilation=4),
        # nn.LSTM(100, 5),
        # nn.GRU(100, 5),
    ]
]

# Activations
INPUT_AND_MODELS += [
    (torch.rand(1, 10, 100), activation)
    for activation in [
        nn.ELU(),
        nn.LeakyReLU(),
        nn.PReLU(),
        nn.ReLU(),
        nn.Sigmoid(),
        nn.Tanh(),
        nn.Softmax(1),
        nn.Softplus(),
        nn.GELU(),
        nn.SELU(),
        nn.SiLU(),
    ]
]

# Test composition is expanded correctly
INPUT_AND_MODELS += [
    (
        torch.rand(1, 10, 100),
        nn.Sequential(
            nn.Sequential(nn.Conv1d(10, 20, 3)),
            nn.Conv1d(20, 30, 5),
            nn.Conv1d(30, 50, 1),
        ),
    ),
]

INPUT_AND_MODELS += [
    (torch.rand(13, 10, 1), op)
    for op in [
        ListInputPrim(torch.cat, (13, 10, 1)),
        # ListInputPrim(torch.stack, (13, 10, 1)), # not implemented in tract
    ]
]


INPUT_AND_MODELS += [
    (torch.rand(13, 10, 1), UnaryPrimitive(op))
    for op in [
        # internal cpp failure for now
        # partial(torch.unbind, axis=1),
        # partial(nnef_split, axis=1, ratios=[3, 3, 4]),
        #
        lambda x: torch.max(x, dim=1, keepdim=True)[0],
        lambda x: torch.min(x, dim=1, keepdim=False)[0],
    ]
]


# Test classical vision models
if os.environ.get("MODELS"):
    INPUT_AND_MODELS += [
        (
            torch.rand(1, 3, 224, 224),
            vision_mdl.alexnet(pretrained=True, progress=False),
        ),
    ]
    INPUT_AND_MODELS += [
        (
            torch.rand(1, 3, 256, 256),
            model,
        )
        for model in [
            vision_mdl.resnet50(pretrained=True, progress=False),
            # vision_mdl.regnet_y_8gf(
            # pretrained=True
            # ),  # works - similar to resnet
            vision_mdl.mnasnet1_0(
                pretrained=True, progress=False
            ),  # works - nas similar to resnet
            vision_mdl.efficientnet_b0(pretrained=True, progress=False),
        ]
    ]


# Test with quantization
if os.environ.get("Q8"):
    INPUT_AND_MODELS = [
        (torch.rand(1, 10, 100), WithQuantDeQuant.quantize_model_and_stub(mod))
        for mod in [
            nn.Sequential(
                nn.Conv1d(10, 20, 3, bias=False),
                # nn.intrinsic.ConvBnReLU1d(
                # nn.Conv1d(10, 20, 3, bias=False),
                # nn.BatchNorm1d(20),
                # nn.ReLU(),
                # ),
                # nn.intrinsic.ConvBnReLU1d(
                # nn.Conv1d(20, 15, 5, stride=2), nn.BatchNorm1d(15), nn.ReLU()
                # ),
                # nn.intrinsic.ConvBnReLU1d(
                # nn.Conv1d(15, 50, 7, stride=3, padding=3),
                # nn.BatchNorm1d(50),
                # nn.ReLU(),
                # ),
            ),
        ]
    ]


def test_should_fail_since_false_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_input = torch.rand(1, 10, 100)
        model = nn.Sequential(nn.Conv1d(10, 20, 3))
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        test_output = model(test_input)
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )

        np.savez(
            io_npz_path,
            input=test_input.detach().numpy(),
            output=test_output.detach().numpy()
            + 1,  # <-- here we artificially add 1 to make it FAIL
        )
        assert not tract_assert_io(
            export_path.with_suffix(".nnef.tgz"), io_npz_path
        ), f"SHOULD fail tract io check with {model}"


# INPUT_AND_MODELS = [
# (torch.rand(1, 10, 100), layer)
# for layer in [
# nn.LSTM(100, 5),
# ]
# ]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_model_export(test_input, model):
    """Test simple models"""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        model = model.eval()

        input_names, output_names = build_io(
            model, test_input, io_npz_path=io_npz_path
        )
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
        )
        real_export_path = export_path.with_suffix(".nnef.tgz")
        assert real_export_path.exists()
        try:

            assert tract_assert_io(
                real_export_path, io_npz_path
            ), f"failed tract io check with {model}"
        except AssertionError as exp:
            if not os.environ.get("DEBUG", False):
                raise exp
            exp_path = (
                Path.cwd()
                / "failed_tests"
                / datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
            )
            exp_path.mkdir(parents=True, exist_ok=True)
            subprocess.check_output(
                f"cd {exp_path} && rm -rf ./* && cp {real_export_path} {exp_path}/model.nnef.tgz "
                f"&& tar -xvzf {real_export_path} && cp {io_npz_path} {exp_path}/io.npz",
                shell=True,
            )
            debug_dumper_pytorch_to_onnx_to_nnef(
                model, test_input, target_folder=exp_path / "tract"
            )
            raise exp
