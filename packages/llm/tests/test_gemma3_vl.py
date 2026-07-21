"""Integration tests for Gemma 3 joint export.

Marked ``experimental`` (needs ``--run-experimental``): downloads gemma-3-4b-it.
PyTorch-parity only (no tract), loaded in f32 for CPU-friendly numerics; this
does not run the tract check_io that must stay memory-bounded on laptops.
"""

import pytest
import torch

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "google/gemma-3-4b-it"

pytestmark = pytest.mark.experimental


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


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


def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def test_encoder_decoder_chain_matches_reference(exporter):
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
