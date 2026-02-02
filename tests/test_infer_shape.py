"""Test inference shape exactness for IR tracing."""

import typing as T

import pytest
import torch

from torch_to_nnef.torch_graph.ir_graph import module_tracer_into_ir_graph
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.ir_op import TorchOp


class TestSuiteIrOpBuilder:
    """Test suite for IR op inference shape exactness by tracing."""

    def __init__(self):
        self.tests = []
        self._ids = []

    def add(self, test_id: str, ir_op: TorchOp):
        self.tests.append(ir_op)
        self._ids.append(test_id)

    @property
    def ir_ops(self):
        return self.tests

    @property
    def ids(self):
        return self._ids


test_suite = TestSuiteIrOpBuilder()


def parse_ir_and_get_first_op(
    model: torch.nn.Module, args: T.Tuple[torch.Tensor, ...], op_kind: str
) -> TorchOp:
    """Parse IR text and return list of TorchOp with names in op_names."""
    torch_ir_graph = module_tracer_into_ir_graph(
        TorchModuleTracer(
            model,
            args=args,
        ),
        is_root_module=True,
    )
    for op in torch_ir_graph.op_nodes:
        if op.kind == op_kind:
            return op
    torch_ir_graph.printall()
    raise ValueError(f"Op of kind {op_kind} not found in traced graph.")


test_suite.add(
    "conv2d",
    parse_ir_and_get_first_op(
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        (torch.randn(1, 3, 32, 32),),
        "aten::_convolution",
    ),
)

test_suite.add(
    "conv2d_stride2_padding3",
    parse_ir_and_get_first_op(
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=3),
        (torch.randn(1, 3, 32, 32),),
        "aten::_convolution",
    ),
)

test_suite.add(
    "conv2d_stride2_padding3_group2",
    parse_ir_and_get_first_op(
        torch.nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=3, groups=2),
        (torch.randn(1, 4, 32, 32),),
        "aten::_convolution",
    ),
)

test_suite.add(
    "conv1d_stride2_padding3_group2",
    parse_ir_and_get_first_op(
        torch.nn.Conv1d(4, 16, kernel_size=3, stride=2, padding=3, groups=2),
        (torch.randn(1, 4, 224),),
        "aten::_convolution",
    ),
)

test_suite.add(
    "linear",
    parse_ir_and_get_first_op(
        torch.nn.Linear(128, 64),
        (torch.randn(1, 128),),
        "aten::linear",
    ),
)

test_suite.add(
    "linear_high_dim",
    parse_ir_and_get_first_op(
        torch.nn.Linear(32, 64),
        (torch.randn(1, 16, 32),),
        "aten::linear",
    ),
)


class MatMulModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)


test_suite.add(
    "matmul_high_dim",
    parse_ir_and_get_first_op(
        MatMulModule(),
        (
            torch.randn(1, 16, 32),
            torch.randn(1, 32, 16),
        ),
        "aten::matmul",
    ),
)


@pytest.mark.parametrize(
    "ir_op",
    test_suite.ir_ops,
    ids=test_suite.ids,
)
def test_check_ir_op_infer_trace_result(ir_op: TorchOp):
    """Check Infer the output shape of an IR op by tracing it.

    Args:
        ir_op: The IR op to trace.
        inputs: The inputs to the IR op.
        inference_target: The target device for inference.

    """
    a = ir_op._infer_trace_result(approx=True)
    b = ir_op._infer_trace_result(approx=False)
    assert a.shape == b.shape, (
        f"Infered shape {a.shape} does not match exact shape {b.shape}"
    )
    assert a.dtype == b.dtype, (
        f"Infered dtype {a.dtype} does not match exact dtype {b.dtype}"
    )
