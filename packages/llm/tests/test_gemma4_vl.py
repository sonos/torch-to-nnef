"""Gemma 4 vision joint-export tests (milestone 1: vision branch).

Default: systematic export on a shrunk dummy config with tract ``check_io``
plus a no-download HF chain-parity check (:mod:`tests.multimodal_dummy`). The
real-checkpoint parity tests are gated behind ``--run-experimental``.

Gemma 4 exercises two decoder mechanics the other handlers do not: per-layer
input embeddings, and a direct ``language_model`` call (the top-level model
cannot be fed pre-spliced embeddings). Audio/video/MoE branches are separate.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import Gemma4Config, Gemma4ForConditionalGeneration

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCheckTolerance, build_io
from torch_to_nnef_llm.exporter import LM_VAR_SCHEME
from torch_to_nnef_llm.models.base import (
    BaseEncoder,
    update_forward_signature,
)
from torch_to_nnef_llm.models.handlers.base import scatter_features_by_mask
from torch_to_nnef_llm.models.handlers.gemma4_vl import (
    Gemma4AudioEncoderHandler,
    Gemma4VideoEncoderHandler,
    Gemma4VisionEncoderHandler,
    _grid_position_ids,
    _num_soft_tokens,
    _sample_audio_frames,
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
    exercises per-layer input embeddings + all three modality splices (image
    and video from the vision tower, audio from the conformer tower) against
    HF's native multimodal forward, in the decoder ``MODALITIES`` order.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    ac = conf.audio_config
    side = _sample_grid_side(vc)
    num_tokens = _num_soft_tokens(vc)
    pos_ids = _grid_position_ids(side)
    patch_dim = 3 * vc.patch_size * vc.patch_size
    # encoder takes the 2D grid; the HF reference takes the flattened patches.
    image_grid = torch.rand(1, side, side, patch_dim)
    video_grid = torch.rand(1, side, side, patch_dim)
    pixel_values = image_grid.reshape(1, side * side, patch_dim)
    pixel_values_videos = video_grid.reshape(1, side * side, patch_dim)
    frames = _sample_audio_frames(ac)
    input_features = torch.randn(1, frames, ac.subsampling_conv_channels[0])
    feature_mask = torch.ones(1, frames, dtype=torch.bool)
    image_token_id = conf.image_token_id
    video_token_id = conf.video_token_id
    audio_token_id = conf.audio_token_id
    vocab_size = ch.decoder_conf.vocab_size
    specials = {image_token_id, video_token_id, audio_token_id}
    filler = next(i for i in range(vocab_size) if i not in specials)

    vision_enc = Gemma4VisionEncoderHandler().get_encoder_module(model).eval()
    video_enc = Gemma4VideoEncoderHandler().get_encoder_module(model).eval()
    audio_enc = Gemma4AudioEncoderHandler().get_encoder_module(model).eval()
    wrapped = exporter.decoder_exporter.wrapped_model.eval()

    with torch.no_grad():
        img_emb = vision_enc(image_grid)
        vid_emb = video_enc(video_grid)
        aud_emb = audio_enc(input_features)
    n_audio = aud_emb.shape[0]

    seq = 1 + 2 * num_tokens + n_audio + 2
    input_ids = torch.full((1, seq), filler, dtype=torch.long)
    input_ids[:, 1 : 1 + num_tokens] = image_token_id
    input_ids[:, 1 + num_tokens : 1 + 2 * num_tokens] = video_token_id
    input_ids[:, 1 + 2 * num_tokens : 1 + 2 * num_tokens + n_audio] = (
        audio_token_id
    )
    _, _, past_kv, _ = ch.build_kv_cache_infos(0)
    past_kv = [t_.to(torch.float32) for t_ in past_kv]

    with torch.no_grad():
        chain_logits = wrapped(input_ids, *past_kv, img_emb, vid_emb, aud_emb)[
            0
        ]
        ref_logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=pos_ids,
            pixel_values_videos=pixel_values_videos.unsqueeze(0),
            video_position_ids=pos_ids.unsqueeze(0),
            input_features=input_features,
            input_features_mask=feature_mask,
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


def test_dynamic_resolution_vision_multi_size():
    """Export ONCE, run tract at several square resolutions.

    Proves the grid axis is genuinely dynamic (not baked) end to end.
    """
    exporter = build_dummy_exporter(
        _dummy_config(), Gemma4ForConditionalGeneration, "f32"
    )
    model = exporter.hf_model_causal.eval()
    handler = exporter.encoder_handlers[0]
    vc = model.config.vision_config
    pool = vc.pooling_kernel_size
    patch_dim = 3 * vc.patch_size * vc.patch_size
    encoder = handler.get_encoder_module(model).eval()
    wrapper = BaseEncoder(encoder, handler).eval()
    io_spec = handler.build_input_spec(
        config_helper=exporter.config_helper, inputs_dtype=torch.float32
    )
    update_forward_signature(wrapper, io_spec)
    target = TractNNEF(version=TractNNEF.latest_version(), check_io=False)

    def make(side):
        return (torch.rand(1, side, side, patch_dim),)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        nnef = tmpd / "vision.nnef.tgz"
        first = make(2 * pool)
        in_names, out_names = build_io(
            wrapper, first, tmpd / "i0.npz", tmpd / "o0.npz"
        )
        target.dynamic_axes = {in_names[0]: {1: "G", 2: "G"}}
        export_model_to_nnef(
            model=wrapper,
            args=first,
            file_path_export=nnef,
            inference_target=target,
            input_names=in_names,
            output_names=out_names,
            nnef_variable_naming_scheme=LM_VAR_SCHEME,
        )
        for k, side in enumerate((2 * pool, 3 * pool)):
            pv = make(side)
            build_io(
                wrapper,
                pv,
                tmpd / f"i{k}.npz",
                tmpd / f"o{k}.npz",
                in_names,
                out_names,
            )
            target.tract_cli.assert_io(
                nnef,
                tmpd / f"i{k}.npz",
                tmpd / f"o{k}.npz",
                check_tolerance=TractCheckTolerance.APPROXIMATE,
            )


def test_dummy_audio_chain_parity():
    """Audio encoder + merge match HF in PyTorch (tract export pending).

    The audio conformer tower is not yet registered for joint export (needs
    tract ops), so this validates the recipe directly: run the audio encoder
    module, splice its soft tokens the way the decoder would, and compare
    logits to HF's native audio forward.
    """
    exporter = build_dummy_exporter(
        _dummy_config(), Gemma4ForConditionalGeneration, "f32"
    )
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    ac = conf.audio_config
    frames = _sample_audio_frames(ac)
    mel = ac.subsampling_conv_channels[0]
    input_features = torch.randn(1, frames, mel)
    feature_mask = torch.ones(1, frames, dtype=torch.bool)

    encoder = Gemma4AudioEncoderHandler().get_encoder_module(model).eval()
    with torch.no_grad():
        audio_emb = encoder(input_features)
    n_audio = audio_emb.shape[0]

    audio_token_id = conf.audio_token_id
    pad_id = conf.text_config.pad_token_id
    vocab = ch.decoder_conf.vocab_size
    filler = next(i for i in range(vocab) if i not in {audio_token_id})
    seq = 1 + n_audio + 2
    input_ids = torch.full((1, seq), filler, dtype=torch.long)
    input_ids[:, 1 : 1 + n_audio] = audio_token_id

    lm = model.model.language_model
    audio_mask = input_ids == audio_token_id
    llm_ids = torch.where(
        audio_mask, torch.full_like(input_ids, pad_id), input_ids
    )
    embeds = model.model.get_input_embeddings()(llm_ids)
    per_layer = lm.get_per_layer_inputs(llm_ids, None)
    embeds = scatter_features_by_mask(
        inputs_embeds=embeds, token_mask=audio_mask, features=audio_emb
    )
    with torch.no_grad():
        out = lm(
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
        )
        my_logits = model.lm_head(out.last_hidden_state)
        ref_logits = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=feature_mask,
        ).logits

    assert (my_logits - ref_logits).abs().max().item() < 1e-3
    assert torch.equal(my_logits.argmax(-1), ref_logits.argmax(-1))


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


@pytest.mark.experimental
def test_chain_matches_reference(exporter):
    _assert_chain_matches_reference(exporter)
