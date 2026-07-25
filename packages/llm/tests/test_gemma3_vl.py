"""Gemma 3 joint-export tests.

Default: systematic export on a shrunk dummy config with tract ``check_io``
(:mod:`tests.multimodal_dummy`). The real-checkpoint parity tests are gated
behind ``--run-experimental`` (they download gemma-3-4b-it).
"""

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import Gemma3Config, Gemma3ForConditionalGeneration

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "google/gemma-3-4b-it"


def _dummy_config() -> Gemma3Config:
    return Gemma3Config(
        image_token_index=1,
        mm_tokens_per_image=4,
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_channels=3,
            image_size=64,
            patch_size=16,
        ),
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=100,
            head_dim=32,
        ),
    )


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_dummy_multimodal_export(dtype, tmp_path):
    assert_dummy_multimodal_export(
        _dummy_config(),
        Gemma3ForConditionalGeneration,
        dtype,
        tmp_path / "export",
    )


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


@pytest.mark.experimental
def test_encoder_matches_reference_image_features(exporter):
    model = exporter.hf_model_causal.eval()
    vc = exporter.config_helper.conf.vision_config
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()

    pv = torch.randn(1, vc.num_channels, vc.image_size, vc.image_size)
    with torch.no_grad():
        ours = encoder(pv)
        ref = model.model.get_image_features(
            pixel_values=pv
        ).pooler_output.reshape(-1, ours.shape[-1])

    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 1e-4


@pytest.mark.experimental
def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def _assert_chain_matches_reference(exporter):
    """Encoder embeddings fed to the decoder wrapper reproduce HF logits.

    The no-download end-to-end guard (also run on the shrunk dummy config):
    exercises Gemma 3's bidirectional image-span mask on top of the splice.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    mm_tokens = int(conf.mm_tokens_per_image)
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    wrapped = exporter.decoder_exporter.wrapped_model.eval()

    pv = torch.randn(1, vc.num_channels, vc.image_size, vc.image_size)
    seq = 1 + mm_tokens + 1
    input_ids = torch.full((1, seq), 10, dtype=torch.long)
    input_ids[:, 1 : 1 + mm_tokens] = conf.image_token_id
    token_type_ids = (input_ids == conf.image_token_id).long()
    _, _, past_kv, _ = ch.build_kv_cache_infos(0)
    past_kv = [t.to(torch.float32) for t in past_kv]

    with torch.no_grad():
        feats = encoder(pv)
        chain_logits = wrapped(input_ids, *past_kv, feats)[0]
        ref_logits = model(
            input_ids=input_ids, pixel_values=pv, token_type_ids=token_type_ids
        ).logits

    kept = chain_logits.shape[1]
    ref_tail = ref_logits[:, -kept:, :]
    assert chain_logits.shape == ref_tail.shape
    assert torch.equal(chain_logits.argmax(-1), ref_tail.argmax(-1))


def test_dummy_chain_parity():
    exporter = build_dummy_exporter(
        _dummy_config(), Gemma3ForConditionalGeneration, "f32"
    )
    _assert_chain_matches_reference(exporter)


@pytest.mark.experimental
def test_encoder_decoder_chain_matches_reference(exporter):
    _assert_chain_matches_reference(exporter)
