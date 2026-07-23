"""Voxtral audio joint-export tests.

Default: systematic export on a shrunk dummy config with tract ``check_io``
(:mod:`tests.multimodal_dummy`) -- proving the encoder abstraction generalizes
to audio (modality="audio"). The real-checkpoint parity tests are gated behind
``--run-experimental`` (they download Voxtral-Mini-3B-2507).
"""

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import VoxtralConfig, VoxtralForConditionalGeneration

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "mistralai/Voxtral-Mini-3B-2507"


def _dummy_config() -> VoxtralConfig:
    return VoxtralConfig(
        audio_token_id=1,
        audio_config=dict(
            num_mel_bins=8,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            max_source_positions=16,
            intermediate_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
        ),
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=100,
        ),
    )


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_dummy_multimodal_export(dtype, tmp_path):
    assert_dummy_multimodal_export(
        _dummy_config(),
        VoxtralForConditionalGeneration,
        dtype,
        tmp_path / "export",
    )


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


def _input_features(model):
    ac = model.config.audio_config
    return torch.randn(1, ac.num_mel_bins, ac.max_source_positions * 2)


@pytest.mark.experimental
def test_encoder_matches_reference_audio_features(exporter):
    model = exporter.hf_model_causal.eval()
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    input_features = _input_features(model)
    with torch.no_grad():
        ours = encoder(input_features)
        ref = model.get_audio_features(input_features).pooler_output
    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 1e-3


@pytest.mark.experimental
def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def _assert_audio_chain_matches_reference(exporter):
    """Audio embeddings fed to the decoder wrapper reproduce HF logits.

    The no-download end-to-end guard (also run on the shrunk dummy config):
    proves the audio splice is modality-neutral, matching HF's native forward.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    input_features = _input_features(model)

    with torch.no_grad():
        audio_emb = encoder(input_features)
    n_audio = audio_emb.shape[0]

    seq = 1 + n_audio + 1
    input_ids = torch.full((1, seq), 10, dtype=torch.long)
    input_ids[:, 1 : 1 + n_audio] = conf.audio_token_id
    _, _, past_kv, _ = ch.build_kv_cache_infos(0)
    past_kv = [t_.to(torch.float32) for t_ in past_kv]

    wrapped = exporter.decoder_exporter.wrapped_model.eval()
    with torch.no_grad():
        chain_logits = wrapped(input_ids, *past_kv, audio_emb)[0]
        ref_logits = model(
            input_ids=input_ids, input_features=input_features
        ).logits

    kept = chain_logits.shape[1]
    ref_tail = ref_logits[:, -kept:, :]
    assert chain_logits.shape == ref_tail.shape
    assert torch.equal(chain_logits.argmax(-1), ref_tail.argmax(-1))


def test_dummy_audio_chain_parity():
    exporter = build_dummy_exporter(
        _dummy_config(), VoxtralForConditionalGeneration, "f32"
    )
    _assert_audio_chain_matches_reference(exporter)


@pytest.mark.experimental
def test_audio_chain_matches_reference(exporter):
    _assert_audio_chain_matches_reference(exporter)
