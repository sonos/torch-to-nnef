from types import SimpleNamespace

import torch
from torch import nn

from torch_to_nnef_llm.models.handlers import GptOssArchitectureHandler
from torch_to_nnef_llm.models.handlers.registry import get_handler


class FakeGptOssExpert(nn.Module):
    def __init__(self, experts_implementation):
        super().__init__()
        self.config = SimpleNamespace(
            _experts_implementation=experts_implementation
        )


class FakeGptOssModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_experts_implementation="grouped_mm")
        self.grouped = FakeGptOssExpert("grouped_mm")
        self.eager = FakeGptOssExpert("eager")


def test_gpt_oss_handler_registered():
    assert get_handler("gpt_oss") is GptOssArchitectureHandler


def test_gpt_oss_handler_uses_traceable_experts_implementation():
    model = FakeGptOssModel()

    GptOssArchitectureHandler().prepare_model_for_export(model)

    assert model.config._experts_implementation == "batched_mm"
    assert model.grouped.config._experts_implementation == "batched_mm"
    assert model.eager.config._experts_implementation == "eager"


NEG = torch.finfo(torch.float32).min


def _wrapper(*, sliding_window=4, layer_types=None, force_causal_mask=True):
    if layer_types is None:
        layer_types = ["sliding_attention", "full_attention"]
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                sliding_window=sliding_window,
                layer_types=layer_types,
            )
        ),
        force_causal_mask=force_causal_mask,
        with_dyn_cache=False,
    )


def _inputs(*, seq_length, past_length, n_layers=2):
    input_ids = torch.zeros(1, seq_length, dtype=torch.long)
    kv = [torch.zeros(1, 2, past_length, 8) for _ in range(2 * n_layers)]
    return (input_ids, *kv)


def _visible(mask):
    """Boolean [S, K] view of an additive mask: True where attention is kept."""
    return mask[0, 0] == 0.0


def test_full_and_sliding_masks_are_distinct():
    handler = GptOssArchitectureHandler()
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=8, past_length=0),
        wrapper=_wrapper(sliding_window=4),
    )
    mapping = ctx.model_inputs["attention_mask"]

    assert isinstance(mapping, dict)
    assert set(mapping) == {"full_attention", "sliding_attention"}
    # The whole point of the handler: these must not be the same tensor.
    assert not torch.equal(
        mapping["full_attention"], mapping["sliding_attention"]
    )


def test_full_mask_is_plain_causal():
    handler = GptOssArchitectureHandler()
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=4, past_length=0),
        wrapper=_wrapper(sliding_window=2),
    )
    visible = _visible(ctx.model_inputs["attention_mask"]["full_attention"])
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(visible, expected)


def test_sliding_mask_keeps_exactly_window_keys():
    handler = GptOssArchitectureHandler()
    window = 3
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=6, past_length=0),
        wrapper=_wrapper(sliding_window=window),
    )
    visible = _visible(ctx.model_inputs["attention_mask"]["sliding_attention"])

    for q in range(6):
        kept = visible[q].nonzero().flatten().tolist()
        # Query q sees keys in (q - window, q], clipped at 0.
        assert kept == list(range(max(0, q - window + 1), q + 1))
        assert len(kept) <= window


def test_sliding_mask_accounts_for_past_length():
    handler = GptOssArchitectureHandler()
    window, past = 4, 10
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=2, past_length=past),
        wrapper=_wrapper(sliding_window=window),
    )
    visible = _visible(ctx.model_inputs["attention_mask"]["sliding_attention"])

    assert visible.shape == (2, past + 2)
    # First new token sits at absolute position 10 and sees keys 7..10.
    assert visible[0].nonzero().flatten().tolist() == [7, 8, 9, 10]
    assert visible[1].nonzero().flatten().tolist() == [8, 9, 10, 11]


def test_masks_are_additive_with_neg_inf_where_hidden():
    handler = GptOssArchitectureHandler()
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=3, past_length=0),
        wrapper=_wrapper(sliding_window=2),
    )
    mask = ctx.model_inputs["attention_mask"]["sliding_attention"]
    assert mask.shape == (1, 1, 3, 3)
    assert mask.dtype == torch.float32
    # Row 2 with window 2 hides key 0 only.
    assert mask[0, 0, 2, 0] == NEG
    assert mask[0, 0, 2, 1] == 0.0
    assert mask[0, 0, 2, 2] == 0.0


def test_full_attention_only_model_keeps_base_single_mask():
    """No sliding layers means the base handler's single mask is correct."""
    handler = GptOssArchitectureHandler()
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=4, past_length=0),
        wrapper=_wrapper(sliding_window=0, layer_types=["full_attention"] * 2),
    )
    assert isinstance(ctx.model_inputs["attention_mask"], torch.Tensor)


def test_no_mask_mapping_when_causal_mask_not_forced():
    handler = GptOssArchitectureHandler()
    ctx = handler.build_forward_inputs(
        inputs=_inputs(seq_length=4, past_length=0),
        wrapper=_wrapper(force_causal_mask=False),
    )
    assert ctx.model_inputs["attention_mask"] is None
