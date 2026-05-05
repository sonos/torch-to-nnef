"""Tests for the LSTMCellExtractor and the JIT-derived input handling.

These cover:
- Standalone `nn.LSTMCell` exported via t2n + tract `check_io`.
- The Silero-VAD-style case where input_size == hidden_size, which is the
  trickier shape-disambiguation path in the extractor.
- The new `strip_assertion_ifs` JIT pass and the constant/getattr extensions
  (`List[int]` constant + non-Tensor `prim::GetAttr`) by going through an
  inlined LSTMCell graph end-to-end via `_jit_pass_inline`.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from torch_to_nnef.torch_graph import strip_assertion_ifs

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class LSTMCellWrapper(nn.Module):
    def __init__(self, in_size: int, hidden: int):
        super().__init__()
        self.cell = nn.LSTMCell(in_size, hidden)

    def forward(self, x, h, c):
        h_new, c_new = self.cell(x, (h, c))
        return h_new, c_new


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_lstm_cell_distinct_sizes(inference_target):
    """input_size != hidden_size: shape disambiguates input vs (h, c)."""
    in_size, hidden = 8, 4
    batch = 2
    module = LSTMCellWrapper(in_size, hidden)
    inference_target = deepcopy(inference_target)
    check_model_io_test(
        model=module,
        test_input=(
            torch.randn(batch, in_size),
            torch.randn(batch, hidden),
            torch.randn(batch, hidden),
        ),
        input_names=["x", "h", "c"],
        output_names=["h_new", "c_new"],
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_lstm_cell_equal_sizes(inference_target):
    """Cover input_size == hidden_size (Silero-VAD's 128/128 case).

    Shape alone cannot disambiguate the cell-call inputs; the extractor
    must rely on positional ordering of the trace.
    """
    in_size = hidden = 6
    batch = 3
    module = LSTMCellWrapper(in_size, hidden)
    inference_target = deepcopy(inference_target)
    check_model_io_test(
        model=module,
        test_input=(
            torch.randn(batch, in_size),
            torch.randn(batch, hidden),
            torch.randn(batch, hidden),
        ),
        input_names=["x", "h", "c"],
        output_names=["h_new", "c_new"],
        inference_target=inference_target,
    )


def test_strip_assertion_ifs_drops_dim_check_branches():
    """Strip nn.LSTMCell's compiled-in input-dim assertion branch.

    nn.LSTMCell carries `if input.dim() not in (1, 2): raise`, which becomes
    a `prim::If` whose true branch only raises. After inline + DCE +
    strip_assertion_ifs the count should drop while the model output stays
    bitwise-identical.
    """
    cell = nn.LSTMCell(8, 4).eval()
    scripted = torch.jit.script(cell)
    torch._C._jit_pass_inline(scripted.graph)
    torch._C._jit_pass_dce(scripted.graph)
    n_if_before = sum(
        1 for n in scripted.graph.nodes() if n.kind() == "prim::If"
    )
    stripped = strip_assertion_ifs(scripted.graph)
    torch._C._jit_pass_dce(scripted.graph)

    # We expect at least the input-dim check to have been stripped.
    assert stripped >= 1, (
        f"strip_assertion_ifs should fold at least the input.dim() check "
        f"(stripped={stripped}, n_if_before={n_if_before})"
    )

    # Output parity vs the un-stripped cell.
    cell_ref = nn.LSTMCell(8, 4).eval()
    cell_ref.load_state_dict(cell.state_dict())
    x = torch.randn(2, 8)
    h = torch.zeros(2, 4)
    c = torch.zeros(2, 4)
    h_ref, c_ref = cell_ref(x, (h, c))
    h_new, c_new = scripted(x, (h, c))
    assert torch.allclose(h_new, h_ref, atol=1e-7)
    assert torch.allclose(c_new, c_ref, atol=1e-7)
