"""Gemma 4 vision joint-export tests (milestone 1: vision branch).

Default: systematic export on a shrunk dummy config with tract ``check_io``
plus a no-download HF chain-parity check (:mod:`tests.multimodal_dummy`). The
real-checkpoint parity tests are gated behind ``--run-experimental``.

Gemma 4 exercises two decoder mechanics the other handlers do not: per-layer
input embeddings, and a direct ``language_model`` call (the top-level model
cannot be fed pre-spliced embeddings). Audio/video/MoE branches are separate.
"""

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import Gemma4Config, Gemma4ForConditionalGeneration

from torch_to_nnef_llm.models.handlers.gemma4_vl import (
    _grid_position_ids,
    _num_soft_tokens,
    _sample_grid_side,
)
from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "google/gemma-4-e2b-it"
VOCAB = 300


def _dummy_config() -> Gemma4Config:
    return Gemma4Config(
        image_token_id=4,
        video_token_id=5,
        audio_token_id=6,
        boi_token_id=7,
        eoi_token_id=8,
        boa_token_id=9,
        eoa_token_index=10,
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            sliding_window=64,
            vocab_size_per_layer_input=VOCAB,
            hidden_size_per_layer_input=32,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            patch_size=16,
            pooling_kernel_size=2,
            position_embedding_size=64,
        ),
        audio_config=dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            output_proj_dims=64,
            subsampling_conv_channels=(16, 8),
        ),
    )


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_dummy_multimodal_export(dtype, tmp_path):
    assert_dummy_multimodal_export(
        _dummy_config(),
        Gemma4ForConditionalGeneration,
        dtype,
        tmp_path / "export",
    )


def _assert_chain_matches_reference(exporter):
    """Encoder embeddings fed to the decoder wrapper reproduce HF logits.

    The no-download end-to-end guard (also run on the shrunk dummy config):
    exercises per-layer input embeddings + the image splice against HF's native
    multimodal forward.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    side = _sample_grid_side(vc)
    num_tokens = _num_soft_tokens(vc)
    pos_ids = _grid_position_ids(side)
    patch_dim = 3 * vc.patch_size * vc.patch_size
    pixel_values = torch.rand(1, side * side, patch_dim)
    image_token_id = conf.image_token_id
    vocab_size = ch.decoder_conf.vocab_size
    filler = next(i for i in range(vocab_size) if i not in {image_token_id})

    encoder = exporter.encoder_handlers[0].get_encoder_module(model).eval()
    wrapped = exporter.decoder_exporter.wrapped_model.eval()

    seq = 1 + num_tokens + 2
    input_ids = torch.full((1, seq), filler, dtype=torch.long)
    input_ids[:, 1 : 1 + num_tokens] = image_token_id
    _, _, past_kv, _ = ch.build_kv_cache_infos(0)
    past_kv = [t_.to(torch.float32) for t_ in past_kv]

    with torch.no_grad():
        img_emb = encoder(pixel_values)
        chain_logits = wrapped(input_ids, *past_kv, img_emb)[0]
        ref_logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=pos_ids,
        ).logits

    kept = chain_logits.shape[1]
    ref_tail = ref_logits[:, -kept:, :]
    assert chain_logits.shape == ref_tail.shape
    assert (chain_logits.float() - ref_tail.float()).abs().max().item() < 1e-3
    assert torch.equal(chain_logits.argmax(-1), ref_tail.argmax(-1))


def test_dummy_chain_parity():
    exporter = build_dummy_exporter(
        _dummy_config(), Gemma4ForConditionalGeneration, "f32"
    )
    _assert_chain_matches_reference(exporter)


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


@pytest.mark.experimental
def test_chain_matches_reference(exporter):
    _assert_chain_matches_reference(exporter)
