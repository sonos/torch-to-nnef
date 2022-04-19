"""Tests simple primitives."""

import os
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.log import log
from torch_to_nnef.tract import tract_assert_io

from .utils import _test_check_model_io, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))


class UnaryPrimitive(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def extra_repr(self):
        return f"op={self.op}"

    def forward(self, x):
        return self.op(x)


class BinaryPrimitive(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def extra_repr(self):
        return f"op={self.op}"

    def forward(self, x1, x2):
        return self.op(x1, x2)


class TensorFnPrimitive(nn.Module):
    def __init__(self, op, kwargs=None, args=None):
        super().__init__()
        self.op = op
        self.args = args or tuple()
        self.kwargs = kwargs or {}

    def extra_repr(self):
        return f"op={self.op}"

    def forward(self, x):
        return getattr(x, self.op)(*self.args, **self.kwargs)


class ListInputPrim(nn.Module):
    def __init__(self, op, dims):
        super().__init__()
        self.y = torch.rand(*dims)
        self.op = op

    def extra_repr(self):
        return f"op={self.op}"

    def forward(self, x):
        return self.op([x, self.y], dim=1)


# Base unary operations
_condition_1 = torch.eye(5, 4).to(torch.bool)
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
        torch.tan,
        torch.asin,
        torch.acos,
        torch.atan,
        torch.sinh,
        torch.cosh,
        torch.tanh,
        torch.asinh,
        torch.acosh,
        torch.atanh,
        torch.zeros_like,
        # unimplemented tract {
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
        partial(torch.clamp, min=5, max=20.0),
        partial(torch.clamp, min=10),
        partial(torch.clamp, max=11),
        # lambda x: torch.where(
        # _condition_1,
        # input=_input0,
        # other=x,
        # ),
    ]
]
INPUT_AND_MODELS += [
    # N x L x  H
    ((torch.rand(10) - 0.5) * 3, layer)
    for layer in [
        # test slice
        TensorFnPrimitive("trunc"),
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
        TensorFnPrimitive("view", args=(13, 5, 2)),
        TensorFnPrimitive("repeat", kwargs={}, args=([1, 2, 1],)),
        TensorFnPrimitive("expand", args=(2, 13, 10, 1)),
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
    (
        (
            torch.arange(10).reshape(5, 2).float(),
            torch.arange(10).reshape(5, 2).T.float(),
        ),
        BinaryPrimitive(op),
    )
    for op in [
        torch.matmul,
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
        nn.Hardtanh(-1, 10),
        nn.LogSoftmax(1),
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
        partial(torch.split, split_size_or_sections=[3, 3, 4], dim=1),
        lambda x: torch.max(x, dim=1, keepdim=True)[0],
        lambda x: torch.min(x, dim=1, keepdim=False)[0],
    ]
]

INPUT_AND_MODELS += [
    # L x N x H
    (torch.rand(33, 1, 20), layer)
    for layer in [
        nn.LSTM(20, 5),
        nn.GRU(20, 5),
        nn.RNN(20, 5),
        nn.GRU(20, 9, num_layers=3),
        nn.LSTM(20, 5, num_layers=2),
        nn.RNN(20, 5, num_layers=3),
        nn.GRU(20, 5, bidirectional=True, num_layers=1),
        nn.LSTM(20, 5, bidirectional=True, num_layers=2),
        nn.RNN(20, 5, bidirectional=True, num_layers=3),
        nn.LSTM(20, 5, proj_size=3, num_layers=2),
    ]
]
INPUT_AND_MODELS += [
    # N x L x  H
    (torch.rand(1, 3, 10), layer)
    for layer in [
        nn.GRU(10, 5, batch_first=True, num_layers=1),
        nn.GRU(10, 5, batch_first=True, bidirectional=True, num_layers=1),
        nn.LSTM(10, 5, batch_first=True, bidirectional=True, num_layers=2),
        nn.RNN(10, 5, batch_first=True, bidirectional=True, num_layers=1),
    ]
]


INPUT_AND_MODELS += [
    # N x L x  H
    (torch.rand(1, 3, 10), layer)
    for layer in [
        # test slice
        UnaryPrimitive(lambda x: x[:, 2:, :]),
        torch.nn.LayerNorm(10),
        torch.nn.GLU(),
    ]
]


def _test_ids(test_fixtures):
    test_names = []
    for data, module in test_fixtures:
        data_fmt = ""
        if isinstance(data, torch.Tensor):
            data_fmt = f"{data.dtype}{list(data.shape)}"
        else:
            for d in data:
                data_fmt += f"{d.dtype}{list(d.shape)}, "
        if len(str(module)) > 100:
            module = str(module.__class__.__name__) + "__" + str(module)[:100]
        test_name = f"{module}({data_fmt})"
        test_names.append(test_name)
    return test_names


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
            log_level=log.WARNING,
        )

        np.savez(
            io_npz_path,
            input=test_input.detach().numpy(),
            output=test_output.detach().numpy()
            + 1,  # <-- here we artificially add 1 to make it FAIL
        )
        assert not tract_assert_io(
            export_path.with_suffix(".nnef.tgz"),
            io_npz_path,
            raise_exception=False,
        ), f"SHOULD fail tract io check with {model}"


@pytest.mark.parametrize(
    "test_input,model", INPUT_AND_MODELS, ids=_test_ids(INPUT_AND_MODELS)
)
def test_primitive_export(test_input, model):
    """Test simple models"""
    _test_check_model_io(model=model, test_input=test_input)
