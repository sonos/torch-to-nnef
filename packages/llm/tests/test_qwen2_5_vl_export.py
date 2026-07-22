"""Integration tests for Qwen2.5-VL joint export.

Marked ``experimental`` (needs ``--run-experimental``): downloads
Qwen2.5-VL-3B-Instruct. PyTorch-parity only (no tract), f32 for CPU numerics.
"""

import pytest
import torch

from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

SLUG = "Qwen/Qwen2.5-VL-3B-Instruct"

pytestmark = pytest.mark.experimental


@pytest.fixture(scope="module")
def exporter():
    return MultiModalExporter.load(
        SLUG, force_module_dtype="f32", force_inputs_dtype="f32"
    )


def test_encoder_matches_reference_image_features(exporter):
    model = exporter.hf_model_causal.eval()
    handler = exporter.encoder_handlers[0]
    vc = model.config.vision_config
    t, h, w = handler.SAMPLE_GRID_THW
    grid = torch.tensor([handler.SAMPLE_GRID_THW], dtype=torch.long)
    patch_dim = (
        vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    )
    pixel_values = torch.randn(t * h * w, patch_dim)

    encoder = handler.get_encoder_module(model).eval()
    with torch.no_grad():
        ours = encoder(pixel_values)
        ref = torch.cat(
            model.model.get_image_features(
                pixel_values, image_grid_thw=grid
            ).pooler_output,
            dim=0,
        )

    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 1e-3


def test_decoder_wrapper_self_consistent(exporter):
    exporter.decoder_exporter.check_wrapper_io()
