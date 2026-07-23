"""Idefics3 / SmolVLM joint-export tests.

By default, a tiny random-weight model is built from a shrunk config and run
through the real exporter (encoder + decoder + manifest) with tract
``check_io`` -- see :mod:`tests.multimodal_dummy`. The real-checkpoint parity
tests are gated behind ``--run-experimental`` (they download SmolVLM).
"""

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import Idefics3Config, Idefics3ForConditionalGeneration

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _dummy_config() -> Idefics3Config:
    return Idefics3Config(
        image_token_id=1,
        scale_factor=2,
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
            max_position_embeddings=128,
        ),
    )


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_dummy_multimodal_export(dtype, tmp_path):
    assert_dummy_multimodal_export(
        _dummy_config(),
        Idefics3ForConditionalGeneration,
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
    handler = exporter.encoder_handlers[0]
    encoder = handler.get_encoder_module(model).eval()

    pv = torch.randn(1, vc.num_channels, vc.image_size, vc.image_size)
    with torch.no_grad():
        ours = encoder(pv)
        ref = model.model.get_image_features(
            pixel_values=pv.unsqueeze(0)
        ).pooler_output.reshape(-1, ours.shape[-1])

    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 1e-4


@pytest.mark.experimental
def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def _assert_chain_matches_reference(exporter):
    """Encoder embeddings fed to the decoder wrapper reproduce HF logits.

    The no-download end-to-end guard (also run on the shrunk dummy config):
    catches image-splice / mask / position bugs that ``check_io`` cannot see.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    image_seq_len = int(
        (vc.image_size // vc.patch_size) ** 2 / (conf.scale_factor**2)
    )
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    wrapped = exporter.decoder_exporter.wrapped_model.eval()

    pv = torch.randn(1, vc.num_channels, vc.image_size, vc.image_size)
    seq = 1 + image_seq_len + 1
    input_ids = torch.full((1, seq), 10, dtype=torch.long)
    input_ids[:, 1 : 1 + image_seq_len] = conf.image_token_id
    _, _, past_kv, _ = ch.build_kv_cache_infos(0)
    past_kv = [t.to(torch.float32) for t in past_kv]

    with torch.no_grad():
        feats = encoder(pv)
        chain_logits = wrapped(input_ids, *past_kv, feats)[0]
        ref_logits = model(input_ids=input_ids, pixel_values=pv.unsqueeze(0))[
            "logits"
        ]

    kept = chain_logits.shape[1]
    ref_tail = ref_logits[:, -kept:, :]
    assert chain_logits.shape == ref_tail.shape
    assert (chain_logits.float() - ref_tail.float()).abs().max().item() < 1e-3
    assert torch.equal(chain_logits.argmax(-1), ref_tail.argmax(-1))


def test_dummy_chain_parity():
    exporter = build_dummy_exporter(
        _dummy_config(), Idefics3ForConditionalGeneration, "f32"
    )
    _assert_chain_matches_reference(exporter)


@pytest.mark.experimental
def test_encoder_decoder_chain_matches_reference(exporter):
    _assert_chain_matches_reference(exporter)
