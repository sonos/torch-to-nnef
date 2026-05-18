"""Mixed-dtype binary ops: PyTorch promotes, t2n must keep round-trip parity.

PyTorch is silent about dtype promotion:

    int_tensor + float_tensor    -> float_tensor   (operands cast to float)
    int_tensor * float_scalar    -> float_tensor
    int_tensor < float_tensor    -> bool_tensor    (operands cast to float)
    torch.maximum(int, float)    -> float_tensor

NNEF / tract are strict and require both operands to share dtype. t2n
covers the common arithmetic / comparison / logical ops through
three registered lists in `torch_to_nnef.op.helper`:

- `OPS_IMPLICIT_CAST_BY_OUTPUT_DTYPE`: mul, div, add, sub, rsub, pow
  (casts every input up to the trace's output dtype).
- `OPS_IMPLICIT_CAST_CONSISTENT_INPS`: ne, ge, le, gt, eq, lt, max,
  min (casts every input up to the lowest-priority dtype among them).
- `OPS_IMPLICIT_CAST_BINARY`: and, or, xor (casts both to bool).

Float-result unary ops (sqrt, log, exp, trig, ...) are bridged in
`torch_to_nnef/op/aten/unary.py:generic_unary` via the
float-promotion fix landed earlier in this PR.

These tests pin each family across the patterns most likely to occur
in real models (integer buffer + float input, integer constant +
float tensor) so a future refactor that drops any of the lists or
the unary bridge is caught immediately.

A wider proptest spec refinement that draws mixed-dtype operands as
part of every binary op's strategy is the right follow-up (see
`feedback_prefer_proptest_refinement` in the project memory). The
standalone cases below are the regression floor.
"""

from __future__ import annotations

import pytest
import torch

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class _MixedAddSub(torch.nn.Module):
    """`int_buf + float_x` and `float_x - int_buf`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offsets", torch.tensor([1, 2, 3, 4]))  # int64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.offsets) - self.offsets


class _MixedMul(torch.nn.Module):
    """`float_x * int_buf`: the common scale-by-int-shape pattern."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([2, 4, 8, 16]))  # int64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _MixedPow(torch.nn.Module):
    """`float_x ** int_tensor`."""

    def forward(self, x: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
        return x.pow(exps)


class _MixedCompare(torch.nn.Module):
    """`int_buf < float_x` casts both to float, then compares."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("thresh", torch.tensor([0, 1, 2, 3]))  # int64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multiple comparison ops in one graph to exercise the codepath
        # at each shape: returns a float mask combining lt / gt / maximum.
        lt = (self.thresh < x).to(torch.float32)
        gt = (self.thresh > x).to(torch.float32)
        return torch.maximum(lt, gt)


class _MixedRemainder(torch.nn.Module):
    """`float_x % int_buf`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("period", torch.tensor([2, 3, 4, 5]))  # int64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x, self.period)


class _MixedAtan2(torch.nn.Module):
    """`atan2(float_y, int_x)`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("xs", torch.tensor([1, 2, 3, 4]))  # int64

    def forward(self, ys: torch.Tensor) -> torch.Tensor:
        return torch.atan2(ys, self.xs)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_add_sub_mixed_dtype(inference_target):
    """`float + int_buf` and `float - int_buf`."""
    x = torch.randn(4)
    check_model_io_test(_MixedAddSub(), x, inference_target)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_mul_mixed_dtype(inference_target):
    """`float * int_buf`."""
    x = torch.randn(4)
    check_model_io_test(_MixedMul(), x, inference_target)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_pow_mixed_dtype(inference_target):
    """`float ** int_tensor`."""
    x = torch.rand(4) + 0.1  # avoid 0**neg edge cases
    exps = torch.tensor([2, 3, 1, 2], dtype=torch.int64)
    check_model_io_test(_MixedPow(), (x, exps), inference_target)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_compare_and_maximum_mixed_dtype(inference_target):
    """`int_buf < float`, `int_buf > float`, `torch.maximum(int, float)`."""
    x = torch.randn(4)
    check_model_io_test(_MixedCompare(), x, inference_target)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_remainder_mixed_dtype(inference_target):
    """`float % int_buf`."""
    x = torch.randn(4) * 5.0
    check_model_io_test(_MixedRemainder(), x, inference_target)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_atan2_mixed_dtype(inference_target):
    """`atan2(float, int_buf)`."""
    ys = torch.randn(4)
    check_model_io_test(_MixedAtan2(), ys, inference_target)
