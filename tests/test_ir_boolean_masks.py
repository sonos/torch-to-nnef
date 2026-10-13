"""Boolean masks must keep their values while tracing derived indices."""

import pytest
import torch

from torch_to_nnef.torch_graph.ir_graph import module_tracer_into_ir_graph
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.torch_const import ATEN_CUMSUM


class _Splice(torch.nn.Module):
    def forward(self, mask, features):
        slots = mask.long().cumsum(0)
        slots = torch.where(mask, slots, torch.zeros_like(slots))
        bank = torch.cat([torch.zeros_like(features[:1]), features], dim=0)
        return bank.index_select(0, slots)


class _Model(torch.nn.Module):
    def __init__(self, bool_input):
        super().__init__()
        self.bool_input = bool_input
        self.splice = _Splice()

    def forward(self, tokens, features):
        mask = tokens if self.bool_input else tokens == 7
        return self.splice(mask, features)


@pytest.mark.parametrize("bool_input", [False, True])
def test_mask_indices_retain_sample_values(monkeypatch, bool_input):
    # Make the allocator-dependent NuExtract failure deterministic: an empty
    # float placeholder becomes an all-True mask instead of the supplied mask.
    original_empty = torch.empty

    def filled_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        return tensor.fill_(1)

    monkeypatch.setattr(torch, "empty", filled_empty)
    tokens = torch.tensor([0, 7, 0, 7, 0])
    if bool_input:
        tokens = tokens == 7
    model = _Model(bool_input).eval()
    graph = module_tracer_into_ir_graph(
        TorchModuleTracer(model, args=(tokens, torch.ones(2, 4)))
    )
    cumulative = next(op for op in graph.op_nodes if op.kind == ATEN_CUMSUM)
    torch.testing.assert_close(
        cumulative.outputs[0].tracing_data, torch.tensor([0, 1, 1, 2, 2])
    )
    assert graph.outputs[0].shape == [5, 4]


class _Membership(torch.nn.Module):
    def __init__(self, assume_unique, invert):
        super().__init__()
        self.assume_unique = assume_unique
        self.invert = invert

    def forward(self, elements, candidates):
        return torch.isin(
            elements,
            candidates,
            assume_unique=self.assume_unique,
            invert=self.invert,
        )


@pytest.mark.parametrize("assume_unique", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_isin_traces_keyword_only_flags(assume_unique, invert):
    model = _Membership(assume_unique, invert)
    inputs = (torch.tensor([1, 4, 7]), torch.tensor([1, 3, 7]))
    graph = module_tracer_into_ir_graph(TorchModuleTracer(model, args=inputs))
    torch.testing.assert_close(graph.outputs[0].tracing_data, model(*inputs))
