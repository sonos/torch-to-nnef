"""Integration tests for Idefics3 / SmolVLM joint export.

Marked ``experimental`` (needs ``--run-experimental``) because they download a
real checkpoint. They validate that the encoder + decoder handlers reproduce the
reference multimodal forward in PyTorch.
"""

import pytest
import torch

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "HuggingFaceTB/SmolVLM-256M-Instruct"

pytestmark = pytest.mark.experimental


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


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


def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def test_encoder_decoder_chain_matches_reference(exporter):
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
