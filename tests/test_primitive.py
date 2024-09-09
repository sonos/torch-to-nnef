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

from torch_to_nnef.exceptions import TorchToNNEFError
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import KhronosNNEF, TractNNEF
from torch_to_nnef.log import log

from .utils import (  # noqa: E402
    INFERENCE_TARGETS_TO_TESTS,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    set_seed,
)

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
    def __init__(self, op, opt_kwargs=None):
        super().__init__()
        self.op = op
        self.opt_kwargs = opt_kwargs

    def extra_repr(self):
        return f"torch.op={self.op}"

    def forward(self, *args, **kwargs):
        if self.opt_kwargs is not None:
            kwargs.update(self.opt_kwargs)
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


test_suite = TestSuiteInferenceExactnessBuilder(INFERENCE_TARGETS_TO_TESTS)


for val in [0.5, 1.5, -2.4]:
    test_suite.add(torch.tensor(val), UnaryPrimitive(torch.round))


# Base unary operations
_condition_1 = torch.eye(5, 4).to(torch.bool)
_input0 = torch.zeros(5, 4)
inp = torch.arange(20).reshape(5, 4).float()
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
    partial(torch.clamp, min=5.0, max=20.0),
    partial(torch.clamp, min=10.0),
    partial(torch.clamp, max=11.0),
    # lambda x: torch.where(
    # _condition_1,
    # input=_input0,
    # other=x,
    # ),
]:
    test_suite.add(inp, UnaryPrimitive(op))


def skip_khronos_interpreter(i):
    return not isinstance(i, KhronosNNEF)


test_suite.add(
    inp,
    UnaryPrimitive(partial(nn.functional.pad, pad=(1, 0), mode="reflect")),
    inference_conditions=skip_khronos_interpreter,  # unssuported
)

test_suite.add(
    (torch.rand(10) - 0.5) * 3,
    TensorFnPrimitive("trunc"),
    inference_conditions=skip_khronos_interpreter,  # unssuported
)


inp = torch.rand(13, 10, 1)
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
    TensorFnPrimitive("clamp_min", args=(0.5,)),
    TensorFnPrimitive("clamp_max", args=(0.5,)),
    TensorFnPrimitive("new_zeros", kwargs=dict(size=(3, 2))),
    TensorFnPrimitive("expand", args=(2, 13, 10, 1)),
]:
    test_suite.add(inp, UnaryPrimitive(op))

for op in [
    TensorFnPrimitive("norm", kwargs=dict(p=2, dim=1, keepdim=True)),
    TensorFnPrimitive("norm", kwargs=dict(p=1, dim=1, keepdim=True)),
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
]:
    test_suite.add(
        inp, UnaryPrimitive(op), inference_conditions=skip_khronos_interpreter
    )

for op in [
    TensorFnPrimitive("expand", args=(3, 2, 1, 10)),
    TensorFnPrimitive("expand", args=(-1, 2, -1, -1)),
    TensorFnPrimitive("expand", args=(2, 3, 2, 1, 10)),
]:
    test_suite.add(
        torch.rand(3, 1, 1, 10),
        UnaryPrimitive(op),
        inference_conditions=skip_khronos_interpreter,
    )

for op in [
    # TensorFnPrimitive("any", {"dim": 1}),
    # TensorFnPrimitive("all", {"dim": 1}),
]:
    test_suite.add(
        torch.tensor(
            [[True, False, True], [True, True, True], [False, False, False]]
        ),
        UnaryPrimitive(op),
    )

# ______________________________________________________________________________________
# _binary
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
]:
    test_suite.add(
        (torch.rand(13, 10), torch.rand(13, 10)), BinaryPrimitive(op)
    )


for op in [
    torch.matmul,
]:
    test_suite.add(
        (
            torch.arange(10).reshape(5, 2).float(),
            torch.arange(10).reshape(5, 2).T.float(),
        ),
        BinaryPrimitive(op),
    )

test_suite.add(
    torch.tensor([True, False, True]), UnaryPrimitive(torch.bitwise_not)
)

for op in [
    (lambda x, y: x & y),  # and
    (lambda x, y: x | y),  # or
]:
    test_suite.add(
        (torch.tensor([True, False, True]), torch.tensor([True, False, False])),
        BinaryPrimitive(op),
    )

# tract do handle cast op with specific ops
test_suite.add(
    torch.tensor([1, 0, 1]),
    TensorFnPrimitive("to", {"dtype": torch.bool}),
    inference_conditions=skip_khronos_interpreter,
)


# Base Layers
for layer in [
    nn.Linear(10, 20, bias=False),
    nn.Linear(10, 32),
]:
    test_suite.add(torch.rand(13, 10), layer)

# f16
for layer in [
    nn.Linear(10, 20, bias=False, dtype=torch.float16),
    nn.Linear(10, 32, dtype=torch.float16),
]:
    test_suite.add(
        torch.rand(13, 10, dtype=torch.float16),
        layer,
        inference_conditions=skip_khronos_interpreter,
    )

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
]:
    test_suite.add(torch.rand(1, 3, 256, 256), layer)

for layer in [
    nn.Conv1d(10, 20, 3),
    nn.Conv1d(10, 20, 3, stride=2),
    nn.Conv1d(10, 20, 3, groups=10),
    nn.Conv1d(10, 20, 3, bias=False),
    nn.Conv1d(10, 20, 3, padding=3),
    nn.Conv1d(10, 20, 3, padding="valid"),
    nn.Conv1d(10, 20, 3, padding="same"),
    nn.MaxPool1d(10, stride=3, padding=2, dilation=1),
    nn.AvgPool1d(10),
]:
    test_suite.add(torch.rand(1, 10, 100), layer)

for layer in [
    nn.BatchNorm1d(10, eps=0, momentum=0.1),
    nn.ConvTranspose1d(10, 20, 3),
    nn.ConvTranspose1d(10, 20, 3, padding=2, dilation=4),
]:
    test_suite.add(
        torch.rand(1, 10, 100),
        layer,
        inference_conditions=skip_khronos_interpreter,
    )  # issue with KhronosNNEF on our side (dim expansion)

# Activations
for activation in [
    nn.Sigmoid(),
    nn.ReLU(),
    nn.Tanh(),
]:
    test_suite.add(torch.rand(1, 10, 100), activation)

for activation in [
    nn.LeakyReLU(),
    nn.ELU(),
    nn.PReLU(),
    nn.Softmax(1),
    nn.Softplus(),
    UnaryPrimitive(torch.erf),
    nn.GELU(),
    nn.GELU(approximate="tanh"),
    nn.SELU(),
    nn.SiLU(),
    nn.Hardtanh(-1.0, 10.0),
    nn.LogSoftmax(1),
    nn.LogSoftmax(dim=-1),
    nn.ReLU6(),
    nn.Hardswish(),
]:
    test_suite.add(
        torch.rand(1, 10, 100),
        activation,
        inference_conditions=skip_khronos_interpreter,
    )

# Test composition is expanded correctly
test_suite.add(
    torch.rand(1, 10, 100),
    nn.Sequential(
        nn.Sequential(nn.Conv1d(10, 20, 3)),
        nn.Conv1d(20, 30, 5),
        nn.Conv1d(30, 50, 1),
    ),
)

test_suite.add(
    torch.rand(13, 10, 1),
    ListInputPrim(torch.cat, torch.rand(13, 10, 1)),
)


for op in [
    partial(torch.split, split_size_or_sections=[3, 3, 4], dim=1),
    lambda x: torch.max(x, dim=1, keepdim=True)[0],
    lambda x: torch.min(x, dim=1, keepdim=False)[0],
]:
    test_suite.add(torch.rand(13, 10, 1), UnaryPrimitive(op))

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
]:
    # L x N x H
    test_suite.add(
        torch.rand(33, 1, 20),
        layer,
        inference_conditions=skip_khronos_interpreter,
    )

for layer in [
    nn.GRU(10, 5, batch_first=True, num_layers=1),
    nn.GRU(10, 5, batch_first=True, bidirectional=True, num_layers=1),
    nn.LSTM(10, 5, batch_first=True, bidirectional=True, num_layers=2),
    nn.RNN(10, 5, batch_first=True, bidirectional=True, num_layers=1),
]:
    # N x L x  H
    test_suite.add(
        torch.rand(1, 3, 10),
        layer,
        inference_conditions=skip_khronos_interpreter,
    )


for layer in [
    # test slice
    UnaryPrimitive(lambda x: x[:, 2:, :]),
    UnaryPrimitive(lambda x: x[..., 1::2]),
    UnaryPrimitive(lambda x: x[..., :2, 1::2]),
    torch.nn.LayerNorm(10),
    torch.nn.LayerNorm((3, 10), eps=1e-5, elementwise_affine=True),
    torch.nn.GLU(),
]:
    test_suite.add(
        torch.rand(1, 3, 10),
        layer,
        inference_conditions=skip_khronos_interpreter,
    )

test_suite.add(
    torch.arange(6).reshape(1, 2, 3).float(),
    ListInputPrim(torch.stack, y=torch.arange(6).reshape(1, 2, 3).float()),
)
test_suite.add(
    torch.arange(6).reshape(1, 2, 3).float(),
    UnaryPrimitive(partial(torch.unbind, axis=1)),
    inference_conditions=skip_khronos_interpreter,
)


#
mdl = nn.GroupNorm(num_groups=3, num_channels=6, eps=0.0)
mdl.requires_grad_ = False
mdl.eval()
test_suite.add(
    torch.arange(12).reshape(1, 6, 2).float(),
    mdl,
    inference_conditions=skip_khronos_interpreter,
)
test_suite.add(
    torch.arange(12).reshape(1, 6, 2, 1).float(),
    mdl,
    inference_conditions=skip_khronos_interpreter,
)

# torch.select
for op in [
    UnaryPrimitive(partial(torch.select, dim=1, index=0)),
    UnaryPrimitive(partial(torch.select, dim=1, index=1)),
    UnaryPrimitive(partial(torch.select, dim=2, index=2)),
    TensorFnPrimitive("select", {"dim": 1, "index": 0}),
]:
    test_suite.add(torch.arange(6).reshape(1, 2, 3).float(), op)

# torch.erf
test_suite.add(
    torch.arange(6).reshape(1, 2, 3).float(),
    UnaryPrimitive(torch.erf),
    inference_conditions=skip_khronos_interpreter,  # unssuported by interpreter
)


# mha basic
hidden_dim = 4
n_heads = 2
keys = torch.randint(2, (1, 2, hidden_dim)).float()
values = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float()
queries = torch.arange(2 * hidden_dim).reshape(1, 2, hidden_dim).float() * 2
test_suite.add(
    (keys, values, queries),
    torch.nn.MultiheadAttention(
        hidden_dim, num_heads=n_heads, dropout=0.0, batch_first=True
    ),
    inference_conditions=skip_khronos_interpreter,
)

test_suite.add(
    (
        torch.arange(15).reshape(1, 5, 3).float(),  # input=(b×n×p)
        torch.arange(10).reshape(1, 5, 2).float(),  # batch1=(b×n×m)
        torch.arange(6).reshape(1, 2, 3).float(),  # batch2=(b×m×p)
    ),
    TernaryPrimitive(torch.baddbmm),
    inference_conditions=skip_khronos_interpreter,
)


test_suite.add(
    (
        torch.arange(6).reshape(1, 2, 3).float(),
        torch.arange(6).reshape(1, 2, 3).float() + 1.0,
    ),
    BinaryPrimitive(torch.remainder),
    inference_conditions=skip_khronos_interpreter,
)

test_suite.add(
    torch.arange(6).reshape(1, 2, 3).float(),
    UnaryPrimitive(partial(torch.roll, shifts=(-1, -2), dims=(1, 2))),
)

test_suite.add(
    torch.arange(100).float() - 50,
    UnaryPrimitive(torch.log10),
)


for qte in [2, 3]:
    # 0.18.5 should have been introducing tract fix that allow slice stride
    # but another bug prevented it's effectiveness
    test_suite.add(
        # N x L x  H
        torch.arange(qte * 10).reshape(1, qte, 10),
        UnaryPrimitive(lambda x: x[..., 0::2]),
        inference_conditions=lambda i: isinstance(i, TractNNEF)
        and "0.19.0" <= i.version,
    )


test_suite.add(
    torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]), nn.Embedding(10, 3)
)


test_suite.add(
    torch.arange(12).reshape(1, 3, 4),
    UnaryPrimitive(lambda x: torch.narrow(x, dim=1, start=1, length=2)),
)


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


def cond_tract_ge_0_20_0(i):
    return isinstance(i, TractNNEF) and "0.20.0" <= i.version


inp = torch.arange(9).reshape(3, 3)
for op in [
    lambda arg: torch.einsum("ii->i", arg),
    lambda arg: torch.einsum("ij", arg),
    lambda arg: torch.einsum("ji", arg),
    lambda arg: torch.einsum("ii", arg),
    lambda arg: torch.einsum("ii->", arg),
    lambda arg: torch.einsum("ij->i", arg),
]:
    test_suite.add(
        torch.arange(9).reshape(3, 3),
        UnaryPrimitive(op),
        inference_conditions=cond_tract_ge_0_20_0,
    )
# NOTE: disable next test as it hangs tract
# _eintest_gen(
#     "i,ij->i",
#     [
#         torch.arange(3).float(),
#         torch.arange(12).reshape(3, 4).float(),
#     ],
# ),
test_suite.add(
    *_eintest_gen(
        "ij,ij->ij",
        [
            torch.arange(12).reshape(3, 4).float(),
            torch.arange(12).reshape(3, 4).float(),
        ],
    ),
    inference_conditions=cond_tract_ge_0_20_0,
)
test_suite.add(
    *_eintest_gen(
        "ij,jk->ijk",
        [
            torch.arange(6).reshape(2, 3).float(),
            torch.arange(12).reshape(3, 4).float(),
        ],
    ),
    inference_conditions=cond_tract_ge_0_20_0,
)
test_suite.add(
    *_eintest_gen(
        "ij,kl->ijkl",
        [
            torch.arange(2).reshape(1, 2).float(),
            torch.arange(12).reshape(3, 4).float(),
        ],
    ),
    inference_conditions=cond_tract_ge_0_20_0,
)


test_suite.add(
    (
        torch.arange(6).reshape(2, 3).float(),
        torch.arange(12).reshape(4, 3).float(),
    ),
    BinaryPrimitive(lambda x, y: torch.vstack([x, y])),
)
test_suite.add(
    (
        torch.arange(24).reshape(2, 3, 4).float(),
        torch.arange(8).reshape(2, 1, 4).float(),
    ),
    BinaryPrimitive(lambda x, y: torch.hstack([x, y])),
)

test_suite.add(
    torch.arange(24).reshape(2, 3, 4),
    UnaryPrimitive(TensorFnPrimitive("flatten")),
)

test_suite.add(
    torch.arange(4).reshape(1, 1, 4),
    UnaryPrimitive(TensorFnPrimitive("unflatten", args=(-1, (2, 2)))),
)

test_suite.add(
    (torch.tensor(1), torch.tensor(6), torch.tensor(3)),
    TorchFnPrimitive("arange"),
    inference_conditions=cond_tract_ge_0_20_0,
)

test_suite.add(  # inverse
    (torch.tensor(10), torch.tensor(-1), torch.tensor(-1)),
    TorchFnPrimitive("arange"),
    inference_conditions=cond_tract_ge_0_20_0,
)


# test groups in conv
for layer in [
    nn.ConvTranspose2d(128, 64, kernel_size=(4, 1), groups=32),
    nn.ConvTranspose2d(128, 64, kernel_size=(4, 1), groups=64),
    nn.Conv2d(128, 64, kernel_size=(4, 1), groups=32),
]:
    test_suite.add(
        torch.rand(1, 128, 8, 3),
        layer,
        inference_conditions=skip_khronos_interpreter,
    )

try:
    from torch.nn.utils import weight_norm as wn

    test_suite.add(
        torch.rand(1, 1, 5, 5),
        wn(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            dim=2,
        ),
        inference_conditions=skip_khronos_interpreter,
    )
except ImportError as exp:
    print("not yet weight_norm import:", exp)


def cond_tract_ge_0_21_4(i):
    return isinstance(i, TractNNEF) and "0.21.4" <= i.version


test_suite.add(
    torch.ones(5, 5),
    TorchFnPrimitive("triu"),
    inference_conditions=cond_tract_ge_0_21_4,
)
test_suite.add(
    torch.ones(5, 5),
    TorchFnPrimitive("tril"),
    inference_conditions=cond_tract_ge_0_21_4,
)

test_suite.add(
    torch.tensor([[1, 2], [3, 4]]),
    TorchFnPrimitive("repeat_interleave", opt_kwargs={"repeats": 4, "dim": 1}),
)
test_suite.add(
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]),
    TorchFnPrimitive("repeat_interleave", opt_kwargs={"repeats": 4, "dim": 0}),
)
test_suite.add(
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]),
    TorchFnPrimitive("repeat_interleave", opt_kwargs={"repeats": 3, "dim": 2}),
)


# More advanced slicing
test_suite.add(
    (torch.tensor([[1, 2], [3, 4], [5, 6]]).float(), torch.tensor([0, 2])),
    BinaryPrimitive(lambda x, y: x[y]),
)
test_suite.add(
    (torch.tensor([[1, 2], [3, 4], [5, 6]]).float(), torch.tensor([1])),
    BinaryPrimitive(lambda x, y: x[:, y]),
)
test_suite.add(
    (
        torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]).float(),
        torch.tensor([1]),
    ),
    BinaryPrimitive(lambda x, y: x[:, :, y]),
)

# issues with export:
# INPUT_AND_MODELS = [
#     (
#         (
#             torch.arange(20).reshape(5, 4).float(),
#             torch.randint(0, 1, (5, 4)).bool(),
#             torch.tensor(1.2)
#         ),
#         TernaryPrimitive(torch.masked_fill)
#     )
# ]


def test_should_fail_since_no_input():
    inference_target = INFERENCE_TARGETS_TO_TESTS[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        test_input = torch.rand(1, 10, 100)
        model = nn.Dropout()
        with pytest.raises(TorchToNNEFError):
            export_model_to_nnef(
                model=model,
                args=test_input,
                file_path_export=export_path,
                input_names=["input"],
                output_names=["output"],
                log_level=log.WARNING,
                inference_target=inference_target,
            )


def test_should_fail_since_false_output():
    inference_target = INFERENCE_TARGETS_TO_TESTS[0]
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
            inference_target=inference_target,
        )

        np.savez(
            io_npz_path,
            input=test_input.detach().numpy(),
            output=test_output.detach().numpy()
            + 1,  # <-- here we artificially add 1 to make it FAIL
        )
        assert not inference_target.tract_cli.assert_io(
            export_path.with_suffix(".nnef.tgz"),
            io_npz_path,
            raise_exception=False,
        ), f"SHOULD fail tract io check with {model}"


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_primitive_export(id, test_input, model, inference_target):
    """Test simple aten PyTorch core"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
