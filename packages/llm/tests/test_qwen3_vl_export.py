"""Qwen3-VL joint-export tests (DeepStack).

Default: systematic export on a shrunk dummy config with tract ``check_io``
(:mod:`tests.multimodal_dummy`). The real-checkpoint parity tests -- which
validate the DeepStack re-injection end to end -- are gated behind
``--run-experimental`` (they download Qwen3-VL-2B-Instruct).
"""

import tempfile
from pathlib import Path

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractNNEF,
    build_io,
)
from torch_to_nnef_llm.exporter import LM_VAR_SCHEME
from torch_to_nnef_llm.models.base import BaseEncoder, update_forward_signature
from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "Qwen/Qwen3-VL-2B-Instruct"

# Like qwen2.5-vl, the f16 vision encoder is bit-exact with tract's optimizer
# disabled but hits the tract `-O` einsum(acc=f32)+cast mis-compile. Fixed in
# tract main; gate so it auto-passes once t2n uses tract >= 0.23.5.
_TRACT_HAS_FP16_OPT_FIX = TractNNEF.latest_version() >= "0.23.5"


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
        "f32",
        pytest.param(
            "f16",
            marks=pytest.mark.xfail(
                condition=not _TRACT_HAS_FP16_OPT_FIX,
                reason="qwen3-vl vision f16: same tract `-O` optimizer bug as "
                "qwen2.5-vl (bit-exact with -O disabled). Fixed in tract main; "
                "passes once tract >= 0.23.5 is used.",
                # non-strict: if tract backports the fix to a version we gate as
                # broken it should just pass quietly, not error on xpass.
                strict=False,
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
    """Return (grid_thw, flat pixel_values, shaped grid).

    The dynamic encoder consumes the merge-block grid
    ``[MH, MW, merge, merge, patch_dim]`` (a pure reshape of the processor's
    flat, merge-block-major patch tensor); the HF reference consumes the flat
    tensor + grid_thw.
    """
    vc = model.config.vision_config
    merge = vc.spatial_merge_size
    t, h, w = handler.SAMPLE_GRID_THW
    grid = torch.tensor([handler.SAMPLE_GRID_THW], dtype=torch.long)
    patch_dim = (
        vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    )
    pixel_values = torch.randn(t * h * w, patch_dim)
    shaped = pixel_values.reshape(
        h // merge, w // merge, merge, merge, patch_dim
    )
    return grid, pixel_values, shaped


@pytest.mark.experimental
def test_encoder_matches_reference_main_and_deepstack(exporter):
    model = exporter.hf_model_causal.eval()
    handler = exporter.encoder_handlers[0]
    n_deep = len(model.config.vision_config.deepstack_visual_indexes)
    grid, pixel_values, shaped = _encoder_inputs(model, handler)

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        ours = encoder(shaped)
        ref = model.model.get_image_features(pixel_values, image_grid_thw=grid)
        ref_main = torch.cat(ref.pooler_output, dim=0)

    assert len(ours) == 1 + n_deep
    assert (ours[0] - ref_main).abs().max().item() < 1e-3
    for i in range(n_deep):
        assert (
            ours[1 + i] - ref.deepstack_features[i]
        ).abs().max().item() < 1e-3


def _assert_deepstack_chain_matches_reference(exporter):
    """Encoder embeddings fed to the decoder wrapper reproduce HF logits.

    The end-to-end guard the shrunk dummy config exercises without download:
    the exported chain (encoder -> spliced embeddings + DeepStack streams ->
    decoder wrapper) must match the native HF multimodal forward. Catches
    injection / mRoPE / DeepStack wiring bugs that ``check_io`` (tract vs the
    same wrapper) cannot.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    handler = exporter.encoder_handlers[0]
    grid, pixel_values, shaped = _encoder_inputs(model, handler)
    num_tokens = (grid.prod().item()) // (vc.spatial_merge_size**2)
    image_token_id = conf.image_token_id
    vision_start = getattr(conf, "vision_start_token_id", image_token_id - 1)
    vocab_size = ch.decoder_conf.vocab_size
    filler = next(
        t for t in range(vocab_size) if t not in {image_token_id, vision_start}
    )

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        enc_out = encoder(shaped)
    img_emb, deep = enc_out[0], list(enc_out[1:])

    seq = 2 + num_tokens + 1
    input_ids = torch.full((1, seq), filler, dtype=torch.long)
    input_ids[:, 0] = vision_start
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


def test_dummy_deepstack_chain_parity():
    exporter = build_dummy_exporter(
        _dummy_config(), Qwen3VLForConditionalGeneration, "f32"
    )
    _assert_deepstack_chain_matches_reference(exporter)


def test_dummy_dynamic_resolution_multi_size():
    """Export the vision tower ONCE, run tract at several grid resolutions.

    Proves the two grid axes are truly dynamic (not baked) end to end,
    including the DeepStack streams.
    """
    exporter = build_dummy_exporter(
        _dummy_config(), Qwen3VLForConditionalGeneration, "f32"
    )
    model = exporter.hf_model_causal.eval()
    handler = exporter.encoder_handlers[0]
    vc = model.config.vision_config
    merge = vc.spatial_merge_size
    patch_dim = (
        vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    )
    wrapper = BaseEncoder(handler.get_encoder_module(model), handler).eval()
    io_spec = handler.build_input_spec(
        config_helper=exporter.config_helper, inputs_dtype=torch.float32
    )
    update_forward_signature(wrapper, io_spec)
    target = TractNNEF(version=TractNNEF.latest_version(), check_io=False)

    def make(mh, mw):
        return (torch.rand(mh, mw, merge, merge, patch_dim),)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        nnef = tmpd / "vision.nnef.tgz"
        in_names, out_names = build_io(
            wrapper, make(4, 4), tmpd / "i0.npz", tmpd / "o0.npz"
        )
        target.dynamic_axes = {in_names[0]: {0: "IMG_H", 1: "IMG_W"}}
        export_model_to_nnef(
            model=wrapper,
            args=make(4, 4),
            file_path_export=nnef,
            inference_target=target,
            input_names=in_names,
            output_names=out_names,
            nnef_variable_naming_scheme=LM_VAR_SCHEME,
        )
        for k, (mh, mw) in enumerate(((4, 4), (3, 5), (6, 2))):
            build_io(
                wrapper,
                make(mh, mw),
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


@pytest.mark.experimental
def test_deepstack_chain_matches_reference(exporter):
    _assert_deepstack_chain_matches_reference(exporter)
