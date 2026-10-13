"""Mistral4 MoE and text-only export of the tiny Mistral Small 4 checkpoint."""

import tarfile
from types import SimpleNamespace

import pytest
import torch
from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX
from torch import nn

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.custom_extractors.moe import (
    MOE_FFN_MIN_TRACT_VERSION,
    _Mistral4Adapter,
)
from torch_to_nnef_llm.loader import load_model
from torch_to_nnef_llm.models.handlers.default import DefaultArchitectureHandler


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, "softmax_topk"),
        ({"norm_topk_prob": False}, "softmax_all"),
        ({"n_group": 2}, None),
        ({"topk_group": 2}, None),
        ({"routed_scaling_factor": 2.0}, None),
    ],
)
def test_mistral4_routing_constraints(overrides, expected):
    config = dict(
        n_group=1, topk_group=1, routed_scaling_factor=1.0, norm_topk_prob=True
    )
    config.update(overrides)
    module = SimpleNamespace(config=SimpleNamespace(**config))
    if expected is None:
        with pytest.raises(T2NErrorNotImplemented):
            _Mistral4Adapter().gate(module)
    else:
        assert _Mistral4Adapter().gate(module) == expected


@pytest.fixture(scope="module")
def tiny_mistral4():
    pytest.importorskip("transformers.models.mistral4.modeling_mistral4")
    model = load_model(
        "tiny-random/mistral-small-4",
        force_module_dtype="f32",
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    assert all(
        p.dtype == torch.float32
        for p in model.parameters()
        if p.is_floating_point()
    )
    return model


class TextLogits(nn.Module):
    """Exercise only the language model and output head, with no image input."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        mask, positions, _ = (
            DefaultArchitectureHandler.build_causal_mask_with_past(
                seq_length=input_ids.shape[1],
                past_length=0,
                device=input_ids.device,
            )
        )
        return self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
        ).logits


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_tiny_mistral4_text_export(tmp_path, tiny_mistral4, inference_target):
    if inference_target.version < MOE_FFN_MIN_TRACT_VERSION:
        pytest.skip(
            f"tract_moe_ffn requires tract >= {MOE_FFN_MIN_TRACT_VERSION}"
        )
    model = TextLogits(tiny_mistral4).eval()
    artifact = export_model_to_nnef(
        model,
        (torch.tensor([[1, 42, 101, 7]], dtype=torch.long),),
        tmp_path / "mistral4-text.nnef.tgz",
        inference_target,
        input_names=["input_ids"],
        output_names=["logits"],
    )
    with tarfile.open(artifact) as archive:
        graph = archive.extractfile("graph.nnef").read().decode()
    assert (
        graph.count("= tract_moe_ffn(")
        == tiny_mistral4.config.text_config.num_hidden_layers
    )
    assert "_se_out" in graph
    assert "se_router" not in graph
    assert "vision_tower" not in graph
