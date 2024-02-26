"""Tests simple primitives."""

import os
import tempfile
import typing as T
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import TractError
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.log import log
from torch_to_nnef.tract import tract_assert_io, tract_version_lower_than

from .utils import check_model_io_test, id_tests, set_seed  # noqa: E402

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


class TernaryPrimitive(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def extra_repr(self):
        return f"op={self.op}"

    def forward(self, x1, x2, x3):
        return self.op(x1, x2, x3)


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


class TorchFnPrimitive(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def extra_repr(self):
        return f"torch.op={self.op}"

    def forward(self, *args, **kwargs):
        return getattr(torch, self.op)(*args, **kwargs)


class ListInputPrim(nn.Module):
    def __init__(self, op, y):
        super().__init__()
        self.y = y
        self.op = op

    def extra_repr(self):
        return f"op={self.op}"

    def forward(self, x):
        return self.op([x, self.y], dim=1)


INPUT_AND_MODELS = [
    # this would not work with nnef_spec_strict activated
    (torch.tensor(val), UnaryPrimitive(torch.round))
    for val in [0.5, 1.5, -2.4]
]


# Base unary operations
_condition_1 = torch.eye(5, 4).to(torch.bool)
_input0 = torch.zeros(5, 4)
INPUT_AND_MODELS += [
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
        torch.ones_like,
        # unimplemented tract {
        # torch.reciprocal,
        # torch.clone,
        # partial(nn.functional.pad, pad=(0, 1), mode="replicate"),
        # }
        partial(torch.pow, exponent=2.0),
        partial(torch.pow, exponent=-2.0),
        partial(torch.pow, exponent=-0.5),
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
        TensorFnPrimitive("norm", kwargs=dict(p=2, dim=1, keepdim=True)),
        TensorFnPrimitive("norm", kwargs=dict(p=1, dim=1, keepdim=True)),
        TensorFnPrimitive("clamp_min", args=(0.5,)),
        TensorFnPrimitive("clamp_max", args=(0.5,)),
        TensorFnPrimitive("new_zeros", kwargs=dict(size=(3, 2))),
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
    (torch.rand(3, 1, 1, 10), UnaryPrimitive(op))
    for op in [
        TensorFnPrimitive("expand", args=(3, 2, 1, 10)),
        TensorFnPrimitive("expand", args=(-1, 2, -1, -1)),
        TensorFnPrimitive("expand", args=(2, 3, 2, 1, 10)),
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

# tract do handle cast op with specific ops
INPUT_AND_MODELS += [
    (
        torch.tensor([1, 0, 1]),
        TensorFnPrimitive("to", {"dtype": torch.bool}),
    )
]


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
        nn.Conv2d(3, 64, (3, 7), padding="same"),
        nn.Flatten(start_dim=1, end_dim=2),
        nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False
        ),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=4, padding=0, dilation=2),
        nn.AvgPool2d(kernel_size=3, stride=4, padding=0),
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
        nn.Conv1d(10, 20, 3, padding="valid"),
        nn.Conv1d(10, 20, 3, padding="same"),
        nn.BatchNorm1d(10, eps=0, momentum=0.1),
        nn.MaxPool1d(10, stride=3, padding=2, dilation=1),
        nn.AvgPool1d(10),
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
        UnaryPrimitive(torch.erf),
        nn.GELU(),
        nn.SELU(),
        nn.SiLU(),
        nn.Hardtanh(-1, 10),
        nn.LogSoftmax(1),
        nn.LogSoftmax(dim=-1),
        nn.ReLU6(),
        nn.Hardswish(),
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
        ListInputPrim(torch.cat, torch.rand(13, 10, 1)),
    ]
]


INPUT_AND_MODELS += [
    (torch.rand(13, 10, 1), UnaryPrimitive(op))
    for op in [
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
        # UnaryPrimitive(lambda x: x[..., 1::2]),
        # UnaryPrimitive(lambda x: x[..., :2, 1::2]),
        torch.nn.LayerNorm(10),
        torch.nn.LayerNorm((3, 10), eps=1e-5, elementwise_affine=True),
        torch.nn.GLU(),
    ]
]

INPUT_AND_MODELS += [
    (torch.arange(6).reshape(1, 2, 3).float(), op)
    for op in [
        UnaryPrimitive(partial(torch.unbind, axis=1)),
        ListInputPrim(torch.stack, y=torch.arange(6).reshape(1, 2, 3).float()),
    ]
]


#
mdl = nn.GroupNorm(num_groups=3, num_channels=6, eps=0.0)
mdl.requires_grad_ = False
mdl.eval()
INPUT_AND_MODELS += [
    (torch.arange(12).reshape(1, 6, 2).float(), mdl),
    (torch.arange(12).reshape(1, 6, 2, 1).float(), mdl),
]

# torch.select
INPUT_AND_MODELS += [
    (torch.arange(6).reshape(1, 2, 3).float(), op)
    for op in [
        UnaryPrimitive(partial(torch.select, dim=1, index=0)),
        UnaryPrimitive(partial(torch.select, dim=1, index=1)),
        UnaryPrimitive(partial(torch.select, dim=2, index=2)),
        TensorFnPrimitive("select", {"dim": 1, "index": 0}),
    ]
]

# torch.erf
INPUT_AND_MODELS += [
    (torch.arange(6).reshape(1, 2, 3).float(), UnaryPrimitive(torch.erf))
]


hidden_dim = 4
n_heads = 2
keys = torch.randint(2, (1, 2, hidden_dim)).float()
values = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float()
queries = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float() * 2
INPUT_AND_MODELS += [
    ((keys, values, queries), op)
    for op in [
        torch.nn.MultiheadAttention(
            hidden_dim, num_heads=n_heads, dropout=0.0, batch_first=True
        )
    ]
]

INPUT_AND_MODELS += [
    (
        (
            torch.arange(15).reshape(1, 5, 3).float(),  # input=(b×n×p)
            torch.arange(10).reshape(1, 5, 2).float(),  # batch1=(b×n×m)
            torch.arange(6).reshape(1, 2, 3).float(),  # batch2=(b×m×p)
        ),
        TernaryPrimitive(torch.baddbmm),
    )
]


INPUT_AND_MODELS += [
    (
        (
            torch.arange(6).reshape(1, 2, 3).float(),
            torch.arange(6).reshape(1, 2, 3).float() + 1.0,
        ),
        BinaryPrimitive(torch.remainder),
    )
]

INPUT_AND_MODELS += [
    (
        torch.arange(6).reshape(1, 2, 3).float(),
        UnaryPrimitive(partial(torch.roll, shifts=(-1, -2), dims=(1, 2))),
    )
]

INPUT_AND_MODELS += [
    (
        torch.arange(100).float() - 50,
        UnaryPrimitive(torch.log10),
    )
]


if not tract_version_lower_than("0.19.0"):
    # 0.18.5 should have been introducing tract fix that allow slice stride
    # but another bug prevented it's effectiveness
    INPUT_AND_MODELS += [
        # N x L x  H
        (
            torch.arange(qte * 10).reshape(1, qte, 10),
            UnaryPrimitive(lambda x: x[..., 0::2]),
        )
        for qte in [2, 3]
    ]


INPUT_AND_MODELS += [
    (torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]), nn.Embedding(10, 3))
]


INPUT_AND_MODELS += [
    # N x L x  H
    (
        torch.arange(12).reshape(1, 3, 4),
        UnaryPrimitive(lambda x: torch.narrow(x, dim=1, start=1, length=2)),
    )
]


class _EinSTest(nn.Module):
    def __init__(self, expr: str, tensors: T.List[torch.Tensor]):
        super().__init__()
        self.expr = expr
        self.tensors = tensors

    def forward(self, a):
        return torch.einsum(self.expr, a, *self.tensors)


def _eintest_gen(expr: str, tensors):
    a = tensors[0]
    others = tensors[1:]
    return (
        a,
        _EinSTest(expr, others),
    )


if not tract_version_lower_than("0.20.0"):
    INPUT_AND_MODELS += [
        (
            torch.arange(9).reshape(3, 3),
            UnaryPrimitive(lambda arg: torch.einsum("ii->i", arg)),
        ),
        (
            torch.arange(9).reshape(3, 3),
            UnaryPrimitive(lambda arg: torch.einsum("ij", arg)),
        ),
        (
            torch.arange(9).reshape(3, 3),
            UnaryPrimitive(lambda arg: torch.einsum("ji", arg)),
        ),
        (
            torch.arange(9).reshape(3, 3),
            UnaryPrimitive(lambda arg: torch.einsum("ii", arg)),
        ),
        (
            torch.arange(9).reshape(3, 3),
            UnaryPrimitive(lambda arg: torch.einsum("ii->", arg)),
        ),
        (
            torch.arange(9).reshape(3, 3),
            UnaryPrimitive(lambda arg: torch.einsum("ij->i", arg)),
        ),
        # NOTE: disable next test as it hangs tract
        # _eintest_gen(
        #     "i,ij->i",
        #     [
        #         torch.arange(3).float(),
        #         torch.arange(12).reshape(3, 4).float(),
        #     ],
        # ),
        _eintest_gen(
            "ij,ij->ij",
            [
                torch.arange(12).reshape(3, 4).float(),
                torch.arange(12).reshape(3, 4).float(),
            ],
        ),
        _eintest_gen(
            "ij,jk->ijk",
            [
                torch.arange(6).reshape(2, 3).float(),
                torch.arange(12).reshape(3, 4).float(),
            ],
        ),
        _eintest_gen(
            "ij,kl->ijkl",
            [
                torch.arange(2).reshape(1, 2).float(),
                torch.arange(12).reshape(3, 4).float(),
            ],
        ),
    ]


INPUT_AND_MODELS += [
    (
        (
            torch.arange(6).reshape(2, 3).float(),
            torch.arange(12).reshape(4, 3).float(),
        ),
        BinaryPrimitive(lambda x, y: torch.vstack([x, y])),
    ),
    (
        (
            torch.arange(24).reshape(2, 3, 4).float(),
            torch.arange(8).reshape(2, 1, 4).float(),
        ),
        BinaryPrimitive(lambda x, y: torch.hstack([x, y])),
    ),
]

INPUT_AND_MODELS += [
    (torch.arange(24).reshape(2, 3, 4), UnaryPrimitive(op))
    for op in [
        TensorFnPrimitive("flatten"),
    ]
]

INPUT_AND_MODELS += [
    (torch.arange(4).reshape(1, 1, 4), UnaryPrimitive(op))
    for op in [
        TensorFnPrimitive("unflatten", args=(-1, (2, 2))),
    ]
]

INPUT_AND_MODELS += [
    (
        (torch.tensor(1), torch.tensor(6), torch.tensor(3)),
        TorchFnPrimitive("arange"),
    ),
    (  # inverse
        (torch.tensor(10), torch.tensor(-1), torch.tensor(-1)),
        TorchFnPrimitive("arange"),
    ),
]


def test_should_fail_since_no_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        test_input = torch.rand(1, 10, 100)
        model = nn.Dropout()
        with pytest.raises(TractError):
            export_model_to_nnef(
                model=model,
                args=test_input,
                file_path_export=export_path,
                input_names=["input"],
                output_names=["output"],
                log_level=log.WARNING,
                check_same_io_as_tract=True,
            )


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
    "test_input,model", INPUT_AND_MODELS, ids=id_tests(INPUT_AND_MODELS)
)
def test_primitive_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
