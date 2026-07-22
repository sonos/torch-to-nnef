"""Skeleton tests for the multimodal joint-export abstraction.

Covers the modality-neutral encoder abstraction with fakes only (no real
checkpoint, no tract): contract wiring, the encoder registry, the BaseEncoder
pipeline, and the pure manifest builder.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorConsistency
from torch_to_nnef_llm.models.base import BaseEncoder
from torch_to_nnef_llm.models.handlers import (
    DefaultArchitectureHandler,
    EmbeddingContract,
    EncoderHandler,
    IOSpec,
    MultiModalArchitectureHandler,
    StateContext,
    get_encoder_handlers,
    is_multimodal,
    register_encoder_handler,
)
from torch_to_nnef_llm.multimodal_exporter import (
    EncoderArtifact,
    build_manifest,
)

HIDDEN = 8
FAKE_ARCH = "fake_mm_test"


class FakeEncoderModule(nn.Module):
    """Maps a [n, ...] input to n embedding rows of width HIDDEN."""

    def forward(self, pixel_values):
        n = pixel_values.shape[0]
        return torch.arange(n * HIDDEN, dtype=torch.float32).view(n, HIDDEN)


@register_encoder_handler
class FakeVisionEncoderHandler(EncoderHandler):
    MODALITY = "vision"
    ARCH_NAMES = (FAKE_ARCH,)

    def get_encoder_module(self, hf_model):
        return hf_model.vision_tower

    def build_input_spec(self, *, config_helper, inputs_dtype):
        pixel_values = torch.zeros((4, 3, HIDDEN), dtype=inputs_dtype)
        return IOSpec(
            inputs=(pixel_values,),
            input_names=["pixel_values"],
            output_names=["out_image_embeddings"],
            dynamic_axes={"pixel_values": {0: "IMG"}},
        )

    def build_forward_inputs(self, *, inputs, wrapper):
        return StateContext(model_inputs={"pixel_values": inputs[0]}, state={})

    def build_forward_outputs(self, *, model_outputs, state_context):
        return [model_outputs]

    def contracts(self, config_helper):
        return [
            EmbeddingContract(
                modality="image",
                hidden_size=HIDDEN,
                placeholder_token_id_attr="image_token_id",
                dynamic_axis="IMG",
            )
        ]


@register_encoder_handler
class FakeAudioEncoderHandler(EncoderHandler):
    MODALITY = "audio"
    ARCH_NAMES = (FAKE_ARCH,)

    def get_encoder_module(self, hf_model):
        return hf_model.audio_tower

    def build_input_spec(self, *, config_helper, inputs_dtype):
        features = torch.zeros((6, HIDDEN), dtype=inputs_dtype)
        return IOSpec(
            inputs=(features,),
            input_names=["input_features"],
            output_names=["out_audio_embeddings"],
            dynamic_axes={"input_features": {0: "AUD"}},
        )

    def build_forward_inputs(self, *, inputs, wrapper):
        return StateContext(model_inputs={"pixel_values": inputs[0]}, state={})

    def build_forward_outputs(self, *, model_outputs, state_context):
        return [model_outputs]

    def contracts(self, config_helper):
        return [
            EmbeddingContract(
                modality="audio",
                hidden_size=HIDDEN,
                placeholder_token_id_attr="audio_token_id",
                dynamic_axis="AUD",
            )
        ]


def test_embedding_contract_derived_names():
    contract = EmbeddingContract(
        modality="image",
        hidden_size=HIDDEN,
        placeholder_token_id_attr="image_token_id",
        dynamic_axis="IMG",
    )
    assert contract.input_name == "in_image_embeddings"
    assert contract.output_name == "out_image_embeddings"
    assert contract.injection_layers == ()


def test_encoder_registry_resolution():
    handlers = get_encoder_handlers(FAKE_ARCH)
    assert FakeVisionEncoderHandler in handlers
    assert FakeAudioEncoderHandler in handlers
    assert is_multimodal(FAKE_ARCH)
    assert not is_multimodal("some_text_only_arch")
    assert get_encoder_handlers("some_text_only_arch") == []


def test_encoder_registry_rejects_duplicate_modality():
    with pytest.raises(T2NErrorConsistency):

        @register_encoder_handler
        class DupVisionHandler(EncoderHandler):
            MODALITY = "vision"
            ARCH_NAMES = (FAKE_ARCH,)

            def get_encoder_module(self, hf_model):
                return hf_model

            def build_input_spec(self, *, config_helper, inputs_dtype):
                return IOSpec((), [], [], {})

            def build_forward_inputs(self, *, inputs, wrapper):
                return StateContext({}, {})

            def build_forward_outputs(self, *, model_outputs, state_context):
                return []

            def contracts(self, config_helper):
                return []


def test_base_encoder_runs_handler_pipeline():
    wrapper = BaseEncoder(FakeEncoderModule(), FakeVisionEncoderHandler())
    outputs = wrapper(torch.zeros(4, 3, HIDDEN))
    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (4, HIDDEN)


def test_multimodal_handler_aggregates_contracts():
    handler = MultiModalArchitectureHandler(
        decoder_handler=DefaultArchitectureHandler(),
        encoder_handlers=[
            FakeVisionEncoderHandler(),
            FakeAudioEncoderHandler(),
        ],
    )
    modalities = {c.modality for c in handler.contracts(None)}
    assert modalities == {"image", "audio"}


def test_build_manifest_wires_encoder_output_to_decoder_input():
    config = SimpleNamespace(
        model_type=FAKE_ARCH, image_token_id=42, video_token_id=43
    )
    contract = EmbeddingContract(
        modality="image",
        hidden_size=HIDDEN,
        placeholder_token_id_attr="image_token_id",
        dynamic_axis="IMG",
        injection_layers=(8, 16, 24),
    )
    artifact = EncoderArtifact(
        label="vision",
        rel_path="vision/model.nnef.tgz",
        contracts=[contract],
    )
    manifest = build_manifest(
        config=config,
        decoder_rel_path="decoder/model.nnef.tgz",
        encoders=[artifact],
        inputs_dtype_str="f16",
    )

    assert manifest["decoder"]["path"] == "decoder/model.nnef.tgz"
    entry = manifest["encoders"][0]
    assert entry["modality"] == "image"
    assert entry["path"] == "vision/model.nnef.tgz"
    assert entry["placeholder_token_id"] == 42
    output = entry["outputs"][0]
    assert output["name"] == "out_image_embeddings"
    assert output["feeds"] == "in_image_embeddings"
    assert output["shape"] == ["IMG", HIDDEN]
    assert output["dtype"] == "f16"
    assert manifest["injection_layers"] == {"image": [8, 16, 24]}


def test_build_manifest_omits_injection_layers_when_absent():
    config = SimpleNamespace(model_type=FAKE_ARCH, image_token_id=42)
    contract = EmbeddingContract(
        modality="image",
        hidden_size=HIDDEN,
        placeholder_token_id_attr="image_token_id",
        dynamic_axis="IMG",
    )
    artifact = EncoderArtifact(
        label="vision", rel_path="vision/model.nnef.tgz", contracts=[contract]
    )
    manifest = build_manifest(
        config=config,
        decoder_rel_path="decoder/model.nnef.tgz",
        encoders=[artifact],
        inputs_dtype_str="f32",
    )
    assert "injection_layers" not in manifest
