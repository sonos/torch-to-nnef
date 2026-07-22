"""Qwen3-VL joint-export tests (DeepStack).

Default: systematic export on a shrunk dummy config with tract ``check_io``
(:mod:`tests.multimodal_dummy`). The real-checkpoint parity tests -- which
validate the DeepStack re-injection end to end -- are gated behind
``--run-experimental`` (they download Qwen3-VL-2B-Instruct).
"""

import pytest
import torch
from multimodal_dummy import assert_dummy_multimodal_export
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "Qwen/Qwen3-VL-2B-Instruct"


def _dummy_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        image_token_id=1,
        vision_start_token_id=2,
        vision_config=dict(
            depth=4,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[1, 2],
        ),
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=100,
        ),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            "f32",
            marks=pytest.mark.xfail(
                reason="qwen3-vl DeepStack dummy: the decoder's per-layer "
                "re-injection slot count and the shrunk-config feature count "
                "disagree; the real-checkpoint experimental test covers "
                "DeepStack end to end",
                strict=True,
            ),
        ),
    ],
)
def test_dummy_multimodal_export(dtype, tmp_path):
    assert_dummy_multimodal_export(
        _dummy_config(),
        Qwen3VLForConditionalGeneration,
        dtype,
        tmp_path / "export",
    )


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


def _encoder_inputs(model, handler):
    vc = model.config.vision_config
    t, h, w = handler.SAMPLE_GRID_THW
    grid = torch.tensor([handler.SAMPLE_GRID_THW], dtype=torch.long)
    patch_dim = (
        vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    )
    return grid, torch.randn(t * h * w, patch_dim)


@pytest.mark.experimental
def test_encoder_matches_reference_main_and_deepstack(exporter):
    model = exporter.hf_model_causal.eval()
    handler = exporter.encoder_handlers[0]
    n_deep = len(model.config.vision_config.deepstack_visual_indexes)
    grid, pixel_values = _encoder_inputs(model, handler)

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        ours = encoder(pixel_values)
        ref = model.model.get_image_features(pixel_values, image_grid_thw=grid)
        ref_main = torch.cat(ref.pooler_output, dim=0)

    assert len(ours) == 1 + n_deep
    assert (ours[0] - ref_main).abs().max().item() < 1e-3
    for i in range(n_deep):
        assert (
            ours[1 + i] - ref.deepstack_features[i]
        ).abs().max().item() < 1e-3


@pytest.mark.experimental
def test_deepstack_chain_matches_reference(exporter):
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    handler = exporter.encoder_handlers[0]
    grid, pixel_values = _encoder_inputs(model, handler)
    num_tokens = (grid.prod().item()) // (vc.spatial_merge_size**2)
    image_token_id = conf.image_token_id

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        enc_out = encoder(pixel_values)
    img_emb, deep = enc_out[0], list(enc_out[1:])

    seq = 2 + num_tokens + 1
    input_ids = torch.full((1, seq), 100, dtype=torch.long)
    input_ids[:, 0] = getattr(conf, "vision_start_token_id", image_token_id - 1)
    input_ids[:, 1 : 1 + num_tokens] = image_token_id
    _, _, past_kv, _ = ch.build_kv_cache_infos(0)
    past_kv = [t_.to(torch.float32) for t_ in past_kv]
    state = [
        img_emb,
        torch.zeros((0, img_emb.shape[-1])),
        grid,
        torch.zeros((0, 3), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long),
    ]

    wrapped = exporter.decoder_exporter.wrapped_model.eval()
    with torch.no_grad():
        chain_logits = wrapped(input_ids, *past_kv, *state, *deep)[0]
        ref_logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            mm_token_type_ids=(input_ids == image_token_id).int(),
        ).logits

    kept = chain_logits.shape[1]
    ref_tail = ref_logits[:, -kept:, :]
    assert chain_logits.shape == ref_tail.shape
    assert torch.equal(chain_logits.argmax(-1), ref_tail.argmax(-1))
