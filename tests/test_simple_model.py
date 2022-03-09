"""Tests simple models."""

from functools import partial
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime


import pytest

import numpy as np
import torch
from torch import nn

from torchvision import models as vision_mdl
from torch_to_nnef.export import export_model_to_nnef

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
    def __init__(self, op, kwargs):
        super().__init__()
        self.op = op
        self.kwargs = kwargs

    def forward(self, x):
        return getattr(x, self.op)(**self.kwargs)


class WithQuantDeQuant(torch.quantization.QuantWrapper):
    @classmethod
    def quantize_model_and_stub(cls, model):
        model = cls(model)
        model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        model_qat = torch.quantization.prepare_qat(model)
        model_q8 = torch.quantization.convert(model_qat.eval())
        return model_q8


# Base unary operations
INPUT_AND_MODELS = [
    (torch.rand(13, 10), UnaryPrimitive(op))
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
        # }
        partial(torch.pow, exponent=2.0),
        partial(torch.pow, exponent=-2.0),
        # partial(torch.transpose, dim0=1, dim1=0), # tract does not find same results ??
        # partial(torch.permute, dims=[1, 0]), # tract does not find same results ??
        partial(torch.unsqueeze, dim=1),
        # need torch_graph to handle ops with multi outputs {
        #    partial(torch.split, split_size_or_sections=5, dim=1),
        #    partial(torch.unbind, dim=1)
        # }
    ]
]
INPUT_AND_MODELS += [
    (torch.rand(13, 10, 1), UnaryPrimitive(op))
    for op in [
        partial(torch.squeeze, dim=2),
        TensorFnPrimitive("mean", {"dim": 1}),
        TensorFnPrimitive("mean", {"dim": 1, "keepdim": True}),
        TensorFnPrimitive("sum", {"dim": 1}),
        # TensorFnPrimitive("max", {"dim": 1}),  # 2x outputs
        # TensorFnPrimitive("min", {"dim": 1}), # 2x outputs
        TensorFnPrimitive("argmax", {"dim": 1}),
        TensorFnPrimitive("argmin", {"dim": 1}),
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
        # TensorFnPrimitive("any", {"dim": 1}), # not implemented in tract
        # TensorFnPrimitive("all", {"dim": 1}), # not implemented in tract
    ]
]

# partial(torch.cat, axis=1),  # ?

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
        # unsupported in tract ? {
        # torch.less,
        # torch.eq,
        # torch.ne,
        # torch.greater,
        # torch.less_equal,
        # torch.greater_equal,
        # }
    ]
]

INPUT_AND_MODELS += [
    ((torch.rand(13, 10), torch.rand(13, 10).T), BinaryPrimitive(op))
    for op in [
        # torch.matmul, # tract not same results ??
    ]
]

# INPUT_AND_MODELS = [
# (torch.tensor([True, False, True]), UnaryPrimitive(torch.bitwise_not))
# ]
# INPUT_AND_MODELS += [
# (
# (torch.tensor([True, False, True]), torch.tensor([True, False, False])),
# BinaryPrimitive(op),
# )
# for op in [
# # tract does not handle io being bool
# (lambda x, y: x & y),  # and
# (lambda x, y: x | y),  # or
# ]
# ]

# tract do not handle cast op
# INPUT_AND_MODELS += [
# (torch.tensor([True, False, True]), TensorFnPrimitive("to", {"dtype":torch.bool}))
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
        # nn.BatchNorm1d(10, eps=0, momentum=0.1),
        # nn.MaxPool1d(10, stride=3, padding=2, dilation=1),
        # nn.AvgPool1d(10),
        # nn.ConvTranspose1d(10, 20, 3),
        # Should we handle LSTM and GRU ???
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
        # We could add this with appropriate fragments
        # nn.GELU(),  # No definition for operator `gelu' in tract
        # nn.SELU(), # No definition for operator `selu' in tract
        # nn.SiLU(),  # No definition for operator `silu' in tract
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

# Test classical vision models
INPUT_AND_MODELS += [
    (
        torch.rand(1, 3, 224, 224),
        vision_mdl.alexnet(pretrained=True),
    ),
]
INPUT_AND_MODELS += [
    (
        torch.rand(1, 3, 256, 256),
        model,
    )
    for model in [
        vision_mdl.resnet50(pretrained=True),
        # vision_mdl.regnet_y_8gf(pretrained=True), # works - similar to resnet
        # vision_mdl.efficientnet_b0(pretrained=True),  # missing silu
    ]
]


# Test with quantization
"""
INPUT_AND_MODELS = [
    (torch.rand(1, 10, 100), WithQuantDeQuant.quantize_model_and_stub(mod))
    for mod in [
        nn.Sequential(
            nn.intrinsic.ConvBnReLU1d(
                nn.Conv1d(10, 20, 3), nn.BatchNorm1d(20), nn.ReLU()
            ),
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
"""


def tract_convert_onnx_to_nnef(onnx_path, io_npz_path, nnef_path):
    subprocess.check_call(
        f'tract {onnx_path} --input-bundle {io_npz_path}  dump --nnef {nnef_path}',
        shell=True,
    )


def tract_assert_io(nnef_path: Path, io_npz_path: Path):
    cmd = f"tract {nnef_path} --input-bundle {io_npz_path} -O run --assert-output-bundle {io_npz_path}"
    try:
        subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


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
            base_path=export_path,
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


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_model_export(test_input, model):
    """Test simple models"""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        model = model.eval()

        tup_inputs = (
            test_input if isinstance(test_input, tuple) else (test_input,)
        )
        input_names = [f"input_{idx}" for idx, _ in enumerate(tup_inputs)]
        output_names = ["output"]
        test_output = model(*tup_inputs)
        export_model_to_nnef(
            model=model,
            args=test_input,
            base_path=export_path,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
        )

        kwargs = {
            f"input_{idx}": input_arg.detach().numpy()
            for idx, input_arg in enumerate(tup_inputs)
        }
        kwargs["output"] = test_output.detach().numpy()

        np.savez(io_npz_path, **kwargs)
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
            tract_exp_path = exp_path / "tract"
            tract_exp_path.mkdir()
            onnx_path = tract_exp_path / "model.onnx"
            torch.onnx.export(
                model,
                test_input,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
            )
            nnef_path = tract_exp_path / "tract_onnx_converted_model.nnef"
            tract_convert_onnx_to_nnef(
                onnx_path,
                io_npz_path,
                nnef_path=nnef_path,
            )
            subprocess.check_output(
                f"cd {tract_exp_path} && tar -xvf {nnef_path}", shell=True
            )
            raise exp
