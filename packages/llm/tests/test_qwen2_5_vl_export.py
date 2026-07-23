"""Qwen2.5-VL joint-export tests.

Default: systematic export on a shrunk dummy config with tract ``check_io``
(:mod:`tests.multimodal_dummy`). The real-checkpoint parity tests are gated
behind ``--run-experimental`` (they download Qwen2.5-VL-3B-Instruct).
"""

import tempfile
from pathlib import Path

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractNNEF,
    build_io,
)
from torch_to_nnef_llm.exporter import LM_VAR_SCHEME
from torch_to_nnef_llm.models.base import BaseEncoder, update_forward_signature
from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter


def _encoder_inputs(model, handler, windows=None):
    """Return (grid_thw, flat pixel_values, shaped window grid).

    The dynamic encoder consumes the window-structured grid
    ``[NH, W, NW, W, merge, merge, patch_dim]`` (a pure reshape of the
    processor's flat, merge-block-major patches, padded to whole windows); the
    HF reference consumes the flat tensor + a window-aligned grid_thw.
    """
    vc = model.config.vision_config
    merge = vc.spatial_merge_size
    win = handler.merger_window(vc)
    nh, nw = windows or handler.SAMPLE_WINDOWS
    h, w = nh * win * merge, nw * win * merge
    grid = torch.tensor([[1, h, w]], dtype=torch.long)
    patch_dim = (
        vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    )
    pixel_values = torch.randn(nh * win * nw * win * merge * merge, patch_dim)
    shaped = pixel_values.reshape(nh, win, nw, win, merge, merge, patch_dim)
    return grid, pixel_values, shaped


SLUG = "Qwen/Qwen2.5-VL-3B-Instruct"

# The qwen2.5-vl f16 gap is a tract `-O` optimizer bug (the emitted graph is
# correct with -O disabled), fixed in tract main. Gate the xfail on the tract
# version so it becomes an expected pass automatically once t2n's supported
# tract bumps past the fix (assumed to land in the next patch, >= 0.23.5).
_TRACT_HAS_FP16_OPT_FIX = TractNNEF.latest_version() >= "0.23.5"


def _dummy_config() -> Qwen2_5_VLConfig:
    return Qwen2_5_VLConfig(
        image_token_id=1,
        vision_start_token_id=2,
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            window_size=112,
            fullatt_block_indexes=[1],
        ),
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=100,
            rope_scaling=dict(mrope_section=[8, 4, 4], rope_type="default"),
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
                reason="qwen2.5-vl vision f16: tract `-O` optimizer bug -- "
                "the emitted graph is correct (bit-exact with -O disabled); "
                "tract's -O mis-compiles the einsum(acc=f32)+cast pattern. "
                "Fixed in tract main; passes once tract >= 0.23.5 is used.",
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
        Qwen2_5_VLForConditionalGeneration,
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
    handler = exporter.encoder_handlers[0]
    grid, pixel_values, shaped = _encoder_inputs(model, handler)

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        ours = encoder(shaped)
        ref = torch.cat(
            model.model.get_image_features(
                pixel_values, image_grid_thw=grid
            ).pooler_output,
            dim=0,
        )

    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 1e-3


@pytest.mark.experimental
def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()


def _assert_chain_matches_reference(exporter):
    """Encoder embeddings fed to the decoder wrapper reproduce HF logits.

    The no-download end-to-end guard (also run on the shrunk dummy config):
    exercises the shared Qwen decoder handler (mRoPE + splice), no DeepStack.
    """
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    handler = exporter.encoder_handlers[0]
    grid, pixel_values, shaped = _encoder_inputs(model, handler)
    num_tokens = (grid.prod().item()) // (vc.spatial_merge_size**2)
    image_token_id = conf.image_token_id
    vision_start = conf.vision_start_token_id
    vocab_size = ch.decoder_conf.vocab_size
    filler = next(
        i for i in range(vocab_size) if i not in {image_token_id, vision_start}
    )

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        img_emb = encoder(shaped)

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
        chain_logits = wrapped(input_ids, *past_kv, *state)[0]
        ref_logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
        ).logits

    kept = chain_logits.shape[1]
    ref_tail = ref_logits[:, -kept:, :]
    assert chain_logits.shape == ref_tail.shape
    assert torch.equal(chain_logits.argmax(-1), ref_tail.argmax(-1))


def test_dummy_chain_parity():
    exporter = build_dummy_exporter(
        _dummy_config(), Qwen2_5_VLForConditionalGeneration, "f32"
    )
    _assert_chain_matches_reference(exporter)


def test_dummy_dynamic_resolution_multi_size():
    """Export the vision tower ONCE, run tract at several window counts.

    Proves the window-count axes are genuinely dynamic (not baked) end to end,
    exercising both windowed and full-attention blocks.
    """
    exporter = build_dummy_exporter(
        _dummy_config(), Qwen2_5_VLForConditionalGeneration, "f32"
    )
    model = exporter.hf_model_causal.eval()
    handler = exporter.encoder_handlers[0]
    vc = model.config.vision_config
    merge = vc.spatial_merge_size
    win = handler.merger_window(vc)
    patch_dim = (
        vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    )
    wrapper = BaseEncoder(handler.get_encoder_module(model), handler).eval()
    io_spec = handler.build_input_spec(
        config_helper=exporter.config_helper, inputs_dtype=torch.float32
    )
    update_forward_signature(wrapper, io_spec)
    target = TractNNEF(version=TractNNEF.latest_version(), check_io=False)

    def make(nh, nw):
        return (torch.rand(nh, win, nw, win, merge, merge, patch_dim),)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        nnef = tmpd / "vision.nnef.tgz"
        in_names, out_names = build_io(
            wrapper, make(2, 2), tmpd / "i0.npz", tmpd / "o0.npz"
        )
        target.dynamic_axes = {in_names[0]: {0: "WIN_H", 2: "WIN_W"}}
        export_model_to_nnef(
            model=wrapper,
            args=make(2, 2),
            file_path_export=nnef,
            inference_target=target,
            input_names=in_names,
            output_names=out_names,
            nnef_variable_naming_scheme=LM_VAR_SCHEME,
        )
        for k, (nh, nw) in enumerate(((1, 1), (2, 1), (2, 3))):
            build_io(
                wrapper,
                make(nh, nw),
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
def test_chain_matches_reference(exporter):
    _assert_chain_matches_reference(exporter)
