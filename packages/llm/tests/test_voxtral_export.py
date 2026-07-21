"""Integration tests for Voxtral audio joint export.

Marked ``experimental`` (needs ``--run-experimental``): downloads
Voxtral-Mini-3B-2507. PyTorch-parity only (no tract), f32 for CPU numerics.
Proves the encoder abstraction generalizes to audio: same EncoderHandler /
EmbeddingContract with modality="audio".
"""

import pytest
import torch

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "mistralai/Voxtral-Mini-3B-2507"

pytestmark = pytest.mark.experimental


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


def _input_features(model):
    ac = model.config.audio_config
    return torch.randn(1, ac.num_mel_bins, ac.max_source_positions * 2)


def test_encoder_matches_reference_audio_features(exporter):
    model = exporter.hf_model_causal.eval()
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    input_features = _input_features(model)
    with torch.no_grad():
        ours = encoder(input_features)
        ref = model.get_audio_features(input_features).pooler_output
    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 1e-3


def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def test_audio_chain_matches_reference(exporter):
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    input_features = _input_features(model)

    with torch.no_grad():
        audio_emb = encoder(input_features)
    n_audio = audio_emb.shape[0]

    seq = 1 + n_audio + 1
    input_ids = torch.full((1, seq), 100, dtype=torch.long)
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
