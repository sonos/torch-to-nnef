"""Tiny, download-free Gemma4 MoE proofs, including every modality tower."""

import copy
import json
import os
import subprocess

import numpy as np
import pytest
import torch
from multimodal_dummy import DummyTokenizer, _graph_nnef
from test_gemma4_vl import _dummy_config
from transformers import Gemma4ForConditionalGeneration

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef_llm.exporter import LM_VAR_SCHEME, LLMExporter
from torch_to_nnef_llm.models.handlers.gemma4_vl import (
    Gemma4MoEFFN,
    _grid_position_ids,
    _sample_audio_frames,
    _sample_grid_side,
)
from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter


def _moe_config(audio=False):
    config = _dummy_config()
    text = config.text_config
    text.enable_moe_block = True
    text.num_experts = 4
    text.top_k_experts = 2
    text.moe_intermediate_size = 32
    text.hidden_size_per_layer_input = 32 if audio else 0
    text.use_bidirectional_attention = "vision"
    text.layer_types = ["sliding_attention", "full_attention"]
    text.global_head_dim = 64
    text.num_key_value_heads = 2
    text.num_global_key_value_heads = 1
    text.attention_k_eq_v = True
    text.final_logit_softcapping = 3.0
    if not audio:
        config.audio_config = None
    return config


def _model(config):
    torch.manual_seed(42)
    model = Gemma4ForConditionalGeneration(config).eval()
    model.set_attn_implementation("eager")
    # Default all-one scales would hide missing folds. Exercise distinct,
    # nonzero channel and expert scales, including a zero norm coefficient.
    with torch.no_grad():
        for layer in model.model.language_model.layers:
            layer.router.scale.uniform_(0.5, 1.5)
            layer.router.per_expert_scale.copy_(
                torch.tensor([0.3, 0.8, 1.4, 2.0])
            )
            layer.pre_feedforward_layernorm_2.weight.uniform_(0.5, 1.5)
            layer.pre_feedforward_layernorm_2.weight[0] = 0
    return model


def _exporter(model):
    return MultiModalExporter(
        LLMExporter(
            hf_model_causal=model,
            tokenizer=DummyTokenizer(),
            force_module_dtype="f32",
            num_logits_to_keep=0,
        )
    )


def _target(exporter):
    target = exporter.decoder_exporter.build_inference_target(
        tract_specific_path=os.environ.get("T2N_TEST_TRACT_PATH"),
        tract_specific_version=os.environ.get("T2N_TEST_TRACT_VERSION"),
    )
    if target.version < "0.23.9" and not (
        os.environ.get("T2N_TEST_TRACT_PATH")
        or os.environ.get("T2N_TEST_TRACT_VERSION")
    ):
        pytest.skip("tract_moe_ffn needs tract 0.23.9 or an explicit binary")
    return target


@pytest.mark.parametrize("audio", [False, True])
def test_gemma4_moe_multimodal_export(audio, tmp_path):
    """Real joint exporter: tract checks decoder, vision, video and audio."""
    model = _model(_moe_config(audio))
    reference = copy.deepcopy(model)
    exporter = _exporter(model)
    raw, inputs, expected, next_expected = _multimodal_sample(
        exporter, reference, larger=True
    )
    assert (
        sum(p.numel() for p in exporter.hf_model_causal.parameters())
        < 1_000_000
    )
    target = _target(exporter)
    exporter.export(tmp_path, target, LM_VAR_SCHEME)
    graph = _graph_nnef(tmp_path / "decoder/model.nnef.tgz")
    assert graph.count("= tract_moe_ffn(") == 2
    assert graph.count("activation = 'gelu'") == 2
    manifest = json.loads((tmp_path / "multimodal.json").read_text())
    assert {entry["modality"] for entry in manifest["encoders"]} == (
        {"image", "video", "audio"} if audio else {"image", "video"}
    )
    for modality in (
        ("vision", "video", "audio") if audio else ("vision", "video")
    ):
        assert (tmp_path / modality / "model.nnef.tgz").exists()
    # Run the actual exported chain at a different image resolution/sequence
    # length than tracing. Decoder inputs come from tract's encoders.
    for handler, contract in zip(
        exporter.encoder_handlers, exporter.contracts, strict=True
    ):
        encoded = _run_tract(
            target,
            tmp_path,
            handler.MODALITY,
            {handler.MODEL_INPUT_NAME: raw[handler.MODALITY][0].numpy()},
        )
        inputs[contract.input_name] = encoded[contract.output_name]
    outputs = _run_tract(target, tmp_path, "decoder", inputs)
    np.testing.assert_allclose(
        outputs["outputs"], expected, atol=3e-4, rtol=3e-3
    )
    # Continue generation with tract's own heterogeneous KV caches. Modality
    # embeddings remain state inputs, but the new token is text-only.
    inputs["input_ids"] = np.array([[42]], dtype=np.int64)
    for name in inputs:
        if name.startswith("in_cache_"):
            inputs[name] = outputs[name.replace("in_", "out_", 1)]
    decoded = _run_tract(target, tmp_path, "decoder", inputs)
    np.testing.assert_allclose(
        decoded["outputs"], next_expected, atol=3e-4, rtol=3e-3
    )


def _run_tract(target, directory, name, inputs):
    in_path = directory / f"{name}-inputs.npz"
    out_path = directory / f"{name}-outputs.npz"
    np.savez(in_path, **inputs)
    subprocess.run(
        target.tract_cli.run_save_outputs_cmd_str(
            directory / name / "model.nnef.tgz", in_path, out_path
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(out_path) as bundle:
        return dict(bundle)


@pytest.mark.parametrize("audio", [False, True])
def test_gemma4_moe_native_multimodal_parity(audio):
    """Compare all logits to the untouched HF model, not just our wrappers."""
    model = _model(_moe_config(audio))
    reference = copy.deepcopy(model)
    exporter = _exporter(model)
    _multimodal_sample(exporter, reference)


def _multimodal_sample(exporter, reference, larger=False):
    model = exporter.hf_model_causal
    config = model.config
    vc = config.vision_config
    side = _sample_grid_side(vc)
    if larger:
        side += vc.pooling_kernel_size
    positions = _grid_position_ids(side)
    patch_dim = 3 * vc.patch_size**2
    image = torch.rand(1, side, side, patch_dim)
    video = torch.rand_like(image)
    raw = {"vision": (image,), "video": (video,)}
    kwargs = dict(
        pixel_values=image.reshape(1, side * side, patch_dim),
        image_position_ids=positions,
        pixel_values_videos=video.reshape(1, 1, side * side, patch_dim),
        video_position_ids=positions.unsqueeze(0),
    )
    if config.audio_config is not None:
        ac = config.audio_config
        features = torch.randn(
            1, _sample_audio_frames(ac), ac.subsampling_conv_channels[0]
        )
        raw["audio"] = (features,)
        kwargs.update(
            input_features=features,
            input_features_mask=torch.ones(
                features.shape[:2], dtype=torch.bool
            ),
        )
    with torch.no_grad():
        embeddings = [
            handler.get_encoder_module(model).eval()(*raw[handler.MODALITY])
            for handler in exporter.encoder_handlers
        ]
        # Text separates the vision spans to exercise the group boundaries.
        ids = [1]
        types = [0]
        for embedding, (_, attr), token_type in zip(
            embeddings,
            exporter.config_helper.handler.MODALITIES,
            (1, 2, 3),
            strict=False,
        ):
            ids.extend([getattr(config, attr)] * len(embedding) + [2])
            types.extend([token_type] * len(embedding) + [0])
        input_ids = torch.tensor([ids])
        cache_names, _, cache, _ = exporter.config_helper.build_kv_cache_infos(
            0
        )
        actual = exporter.decoder_exporter.wrapped_model(
            input_ids, *cache, *embeddings
        )[0]
        expected = reference(
            input_ids=input_ids,
            mm_token_type_ids=torch.tensor([types]),
            use_cache=False,
            **kwargs,
        ).logits
        next_expected = reference(
            input_ids=torch.cat([input_ids, torch.tensor([[42]])], dim=1),
            mm_token_type_ids=torch.tensor([types + [0]]),
            use_cache=False,
            **kwargs,
        ).logits[:, -1:]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    inputs = {"input_ids": input_ids.numpy()}
    inputs.update(
        {
            name: value.numpy()
            for name, value in zip(cache_names, cache, strict=True)
        }
    )
    inputs.update(
        {
            c.input_name: value.numpy()
            for c, value in zip(exporter.contracts, embeddings, strict=True)
        }
    )
    return raw, inputs, expected.numpy(), next_expected.numpy()


def test_gemma4_moe_rejects_other_activation(tmp_path):
    model = _model(_moe_config())
    model.model.language_model.layers[0].experts.act_fn = torch.nn.ReLU()
    exporter = _exporter(model)
    block = model.model.language_model.layers[0].mlp
    assert isinstance(block, Gemma4MoEFFN)
    with pytest.raises(T2NErrorNotImplemented, match="tanh-approximate GELU"):
        export_model_to_nnef(
            block,
            (torch.randn(1, 5, 64),),
            tmp_path / "bad.nnef.tgz",
            _target(exporter),
        )
