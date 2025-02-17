import typing as T

import torch
from torch import nn


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


class Einsum(nn.Module):
    def __init__(self, expr: str, tensors: T.List[torch.Tensor]):
        super().__init__()
        self.expr = expr
        self.tensors = tensors

    def forward(self, a):
        return torch.einsum(self.expr, a, *self.tensors)
