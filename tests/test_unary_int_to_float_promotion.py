"""Float-result unary ops over integer inputs (PyTorch promotes silently).

PyTorch: `torch.sqrt(torch.tensor([4], dtype=torch.int64))` returns a
float32 tensor (auto-promotion). NNEF / tract have no such implicit
promotion: emitting the bare op on an integer tensor either fails
type-checking with `no super type for F32 and I64` (e.g. when the float
result feeds back into an arithmetic op against a float operand) or
produces the wrong dtype. t2n bridges that by inserting a cast on
the unary op's input when the trace's recorded output dtype differs
from the input dtype.

These tests pin the promotion for `sqrt` on int64 inputs and for the
upstream-DPDFNet pattern that surfaced the gap:
`int_buffer.sqrt() + float_eps` then divided into a float tensor.

Note: a wider proptest spec refinement that exercises every
float-result unary op on every integer dtype would be a strictly
better regression net (see `feedback_prefer_proptest_refinement` in
memory) and is the right follow-up.
"""

from __future__ import annotations

import pytest
import torch

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class _SqrtOnIntBuffer(torch.nn.Module):
    """Minimal repro: `sqrt(int64_buffer) + float_eps` -> float_tensor / ...

    Matches the upstream DPDFNet `MagNorm48` shape that originally
    surfaced the gap. The int64 buffer comes from
    `torch.full(shape, int_value)` which PyTorch infers as int64.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("var0", torch.full((1, 1, 8), 40 * 40))  # int64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = self.var0.sqrt() + 1e-8
        return x / denom


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_sqrt_on_int_buffer_promotes_to_float(inference_target):
    """`int64_buf.sqrt() + float_eps` mixed with float ops, DPDFNet pattern."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8)
    check_model_io_test(_SqrtOnIntBuffer(), x, inference_target)


class _IntSqrtStandalone(torch.nn.Module):
    """Just `torch.sqrt(int_tensor) -> float_tensor`, no further arithmetic."""

    def forward(self, x: torch.Tensor, ints: torch.Tensor) -> torch.Tensor:
        return x * ints.sqrt()


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_sqrt_on_int_tensor_promotes_to_float(inference_target):
    """`sqrt(int_input)` * float_tensor -- dtype must match."""
    torch.manual_seed(0)
    x = torch.randn(4, dtype=torch.float32)
    ints = torch.tensor([1, 4, 9, 16], dtype=torch.int64)
    check_model_io_test(_IntSqrtStandalone(), (x, ints), inference_target)
