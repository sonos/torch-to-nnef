"""Qwen3.5 dense VLM (``qwen3_5``, e.g. Hcompany/Holo-3.1) joint-export tests.

Default: systematic export on a shrunk dummy config with tract ``check_io``
(:mod:`tests.multimodal_dummy`) plus a no-download chain-parity check of the
whole encoder -> hybrid decoder pipeline against HF's native multimodal forward.
The decoder is the interesting part: a gated-delta-net (GDN) linear-attention
layer for every ``linear_attention`` entry in ``config.layer_types`` (streaming
conv state + matrix recurrent state, emitted via the ``gated_delta_scan`` op)
and a standard attention layer (KV cache) for every ``full_attention`` entry.
"""

import pytest
import torch
from multimodal_dummy import (
    assert_dummy_multimodal_export,
    build_dummy_exporter,
)
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractNNEF,
    build_io,
)
from torch_to_nnef_llm.models.base import BaseEncoder, update_forward_signature
from torch_to_nnef_llm.models.handlers.qwen3_5_vl import (
    Qwen35ArchitectureHandler,
)

# fp16 vision tower: same tract `-O` einsum(acc=f32)+cast mis-compile the other
# Qwen VL towers gate; bit-exact with the optimizer disabled. Fixed in tract
# main, so auto-passes once t2n uses tract >= 0.23.5.
_TRACT_HAS_FP16_OPT_FIX = TractNNEF.latest_version() >= "0.23.5"


def _dummy_config() -> Qwen3_5Config:
    """Shrunk config with the 3 GDN : 1 full-attention layer pattern.

    Tiny GDN dims keep the recurrent state small.
    """
    return Qwen3_5Config(
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            head_dim=16,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        ),
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
                reason="qwen3.5 vision f16: same tract `-O` optimizer bug as "
                "the other Qwen VL towers (bit-exact with -O disabled). Fixed "
                "in tract main; passes once tract >= 0.23.5.",
                strict=False,
            ),
        ),
    ],
)
def test_dummy_multimodal_export(dtype, tmp_path):
    assert_dummy_multimodal_export(
        _dummy_config(),
        Qwen3_5ForConditionalGeneration,
        dtype,
        tmp_path / "export",
    )


def _encoder_inputs(model, handler):
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


def test_dummy_chain_parity():
    """Encoder embeddings fed to the hybrid decoder reproduce HF logits.

    The end-to-end guard without any download: the exported chain (vision
    encoder -> spliced image embeddings -> hybrid GDN/attention decoder) must
    match the native HF multimodal forward. Catches injection / mRoPE / GDN /
    KV wiring bugs that ``check_io`` (tract vs the same wrapper) cannot.
    """
    exporter = build_dummy_exporter(
        _dummy_config(), Qwen3_5ForConditionalGeneration, "f32"
    )
    model = exporter.hf_model_causal.eval()
    ch = exporter.config_helper
    conf = ch.conf
    vc = conf.vision_config
    handler = exporter.encoder_handlers[0]
    grid, pixel_values, shaped = _encoder_inputs(model, handler)
    num_tokens = grid.prod().item() // (vc.spatial_merge_size**2)
    image_token_id = conf.image_token_id
    vision_start = getattr(conf, "vision_start_token_id", image_token_id - 1)
    vocab_size = ch.decoder_conf.vocab_size
    filler = next(
        t for t in range(vocab_size) if t not in {image_token_id, vision_start}
    )

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        img_emb = encoder(shaped)

    seq = 2 + num_tokens + 1
    input_ids = torch.full((1, seq), filler, dtype=torch.long)
    input_ids[:, 0] = vision_start
    input_ids[:, 1 : 1 + num_tokens] = image_token_id

    # fresh (prefill) hybrid state: zero conv/recurrent states, empty KV.
    dec_handler = Qwen35ArchitectureHandler()
    state_inputs, _ = dec_handler._build_state_inputs(
        ch.decoder_conf, 0, torch.float32
    )
    passthrough = [
        img_emb,
        grid,
        torch.zeros((1, 1), dtype=torch.long),
    ]

    wrapped = exporter.decoder_exporter.wrapped_model.eval()
    with torch.no_grad():
        chain_logits = wrapped(input_ids, *state_inputs, *passthrough)[0]
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
    assert (chain_logits - ref_tail).abs().max().item() < 1e-3


def test_dummy_dynamic_resolution_multi_size():
    """Export the vision tower ONCE, run tract at several grid resolutions.

    Proves the two grid axes stay dynamic (not baked) end to end.
    """
    exporter = build_dummy_exporter(
        _dummy_config(), Qwen3_5ForConditionalGeneration, "f32"
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
    target.dynamic_axes = io_spec.dynamic_axes

    def make(mh, mw):
        return (torch.randn(mh, mw, merge, merge, patch_dim),)

    import tempfile
    from pathlib import Path

    from torch_to_nnef import export_model_to_nnef

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nnef = tmp / "enc.nnef.tgz"
        first = make(4, 4)
        inn, outn = build_io(
            wrapper,
            first,
            tmp / "i.npz",
            tmp / "o.npz",
            io_spec.input_names,
            io_spec.output_names,
        )
        export_model_to_nnef(
            model=wrapper,
            args=first,
            file_path_export=nnef,
            inference_target=target,
            input_names=io_spec.input_names,
            output_names=io_spec.output_names,
        )
        for j, (mh, mw) in enumerate([(4, 6), (8, 4)]):
            a = make(mh, mw)
            build_io(
                wrapper,
                a,
                tmp / f"i{j}.npz",
                tmp / f"o{j}.npz",
                io_spec.input_names,
                io_spec.output_names,
            )
            target.tract_cli.assert_io(
                nnef,
                tmp / f"i{j}.npz",
                tmp / f"o{j}.npz",
                check_tolerance=TractCheckTolerance.VERY,
            )
