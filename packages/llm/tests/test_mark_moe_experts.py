import pytest
from torch import nn

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.custom_extractors.moe import (
    MoEFFN,
    mark_moe_experts_for_q40,
)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(8, 8)
        self.moe = MoEFFN(num_experts=4, d_model=32, d_hidden=32)


def test_autodetect_marks_registered_moe_classes():
    model = Wrapper()
    marked = mark_moe_experts_for_q40(model, layout="linear")
    assert marked == 1
    assert model.moe._t2n_quantize_moe_experts_q40 is True
    assert model.moe._t2n_moe_expert_layout == "linear"
    assert not hasattr(model.dense, "_t2n_quantize_moe_experts_q40")


def test_explicit_class_names_restrict_matching():
    model = Wrapper()
    marked = mark_moe_experts_for_q40(
        model, module_class_names={"MoEFFN"}, layout="canonical"
    )
    assert marked == 1
    assert model.moe._t2n_moe_expert_layout == "canonical"


def test_layout_only_marking_does_not_enable_q40():
    model = Wrapper()
    marked = mark_moe_experts_for_q40(
        model, quantize_q40=False, layout="linear"
    )
    assert marked == 1
    assert not getattr(model.moe, "_t2n_quantize_moe_experts_q40", False)
    assert model.moe._t2n_moe_expert_layout == "linear"


def test_nothing_matched_raises():
    with pytest.raises(T2NErrorNotImplemented, match="no MoE module found"):
        mark_moe_experts_for_q40(nn.Linear(8, 8))


def test_bad_layout_raises():
    with pytest.raises(T2NErrorNotImplemented, match="expert layout"):
        mark_moe_experts_for_q40(Wrapper(), layout="bogus")
