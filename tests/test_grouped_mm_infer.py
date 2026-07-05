import torch

from torch_to_nnef.torch_graph.ir_op import INFER_RULES
from torch_to_nnef.torch_graph.torch_const import ATEN_GATHER, ATEN_GROUPED_MM


def test_grouped_mm_infer_rule_uses_shape_not_kernel_execution():
    rule = INFER_RULES[ATEN_GROUPED_MM]

    shape = rule.fn(torch.empty(5, 16), torch.empty(3, 16, 32))

    assert shape == torch.Size([5, 32])


def test_gather_infer_rule_uses_index_shape_not_kernel_execution():
    rule = INFER_RULES[ATEN_GATHER]

    shape = rule.fn(
        torch.empty(2, 4),
        1,
        torch.empty(2, 3, dtype=torch.int64),
    )

    assert shape == torch.Size([2, 3])
