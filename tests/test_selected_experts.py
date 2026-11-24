"""Selected-expert fallback retains dynamic routing and custom activations."""

import pytest
import torch
from torch import nn

from torch_to_nnef.compress import DEFAULT_COMPRESSION
from torch_to_nnef.modules import SelectedGatedExperts
from torch_to_nnef.tensor.quant import QTensor


class SoftcappedGate(nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate.tanh()) * (2 * (up / 2).tanh())


def _experts():
    torch.manual_seed(7)
    return SelectedGatedExperts(
        nn.Parameter(torch.randn(4, 16, 8) * 0.1),
        nn.Parameter(torch.randn(4, 8, 8) * 0.1),
        SoftcappedGate(),
    ).eval()


def _reference(model, x, indices, weights):
    rows = []
    for token in range(len(x)):
        outputs = []
        for slot, expert in enumerate(indices[token]):
            gate, up = (
                model.gate_up_proj[expert].reshape(16, 8) @ x[token]
            ).chunk(2)
            outputs.append(
                (
                    model.down_proj[expert].reshape(8, 8)
                    @ model.activation(gate, up)
                ).float()
                * weights[token, slot]
            )
        rows.append(sum(outputs))
    return torch.stack(rows).to(x.dtype)


@pytest.mark.parametrize("quantize", [False, True])
def test_selected_experts_changes_routes_after_trace(quantize):
    model = _experts()
    if quantize:
        DEFAULT_COMPRESSION["min_max_q4_0_with_embeddings_and_experts"](model)
        assert isinstance(model.gate_up_proj, QTensor)
        assert isinstance(model.down_proj, QTensor)
    with torch.no_grad():
        args = (
            torch.randn(2, 8),
            torch.zeros(2, 2, dtype=torch.long),
            torch.rand(2, 2),
        )
        # QTensor is converted to an opaque reference by the export pipeline;
        # direct JIT tracing of its packed storage is not supported.
        traced = model if quantize else torch.jit.trace(model, args)
        for n in (1, 5):
            x, indices, weights = (
                torch.randn(n, 8),
                torch.randint(1, 4, (n, 2)),
                torch.rand(n, 2),
            )
            torch.testing.assert_close(
                traced(x, indices, weights),
                _reference(model, x, indices, weights),
            )


def test_selected_experts_registry_respects_skip_and_keeps_router_float():
    block = nn.Module()
    block.experts = _experts()
    block.gate = nn.Linear(32, 4)
    block.projection = nn.Linear(32, 32)
    DEFAULT_COMPRESSION["min_max_q4_0_with_embeddings_and_experts"](
        block, modules_to_skip=(block.experts,)
    )
    assert not isinstance(block.experts.gate_up_proj, QTensor)
    assert not isinstance(block.experts.down_proj, QTensor)
    assert not isinstance(block.gate.weight, QTensor)
    assert isinstance(block.projection.weight, QTensor)


@pytest.mark.parametrize("quantize", [False, True])
def test_selected_experts_dynamic_nnef(tmp_path, quantize, monkeypatch):
    from tests.test_symbolic_window_ops import _check_dynamic
    from torch_to_nnef.tensor import opaque

    model = _experts()
    if quantize:
        monkeypatch.setattr(opaque, "NEW_OPAQUE_TRACING_STRATEGY", False)
        DEFAULT_COMPRESSION["min_max_q4_0_with_embeddings_and_experts"](model)
    _check_dynamic(
        tmp_path,
        model,
        [
            (torch.randn(n, 8), torch.randint(0, 4, (n, 2)), torch.rand(n, 2))
            for n in (2, 1, 5)
        ],
        {"input_0": {0: "S"}, "input_1": {0: "S"}, "input_2": {0: "S"}},
    )
