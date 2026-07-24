"""Skeleton tests for the multimodal joint-export abstraction.

Covers the modality-neutral encoder abstraction with fakes only (no real
checkpoint, no tract): contract wiring, the encoder registry, the BaseEncoder
pipeline, and the pure manifest builder.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorConsistency, T2NErrorMisuse
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef_llm.exporter import _reject_multimodal_in_llm_dump
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
from torch_to_nnef_llm.models.handlers.base import (
    reset_special_ids_to_filler,
    resolve_submodule,
    scatter_features_by_mask,
)
from torch_to_nnef_llm.multimodal_exporter import (
    EncoderArtifact,
    _loosest_tolerance,
    build_manifest,
)

HIDDEN = 8
FAKE_ARCH = "fake_mm_test"


@pytest.fixture(scope="module", autouse=True)
def _cleanup_fake_encoder_registry():
    """Drop this module's fake encoder handlers from the global registry.

    The fakes register at import time (module-level decorators) into the
    process-wide ``_ENCODER_REGISTRY``; without teardown they leak into every
    later test in the session.
    """
    yield
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef_llm.models.handlers.registry import _ENCODER_REGISTRY

    _ENCODER_REGISTRY.pop(FAKE_ARCH, None)


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
        injection_layers=(0, 1, 2),
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
    assert manifest["injection_layers"] == {"image": [0, 1, 2]}
    # DeepStack per-layer streams: stream i feeds in_image_deepstack_i,
    # injected at injection_layers[i].
    deepstack = entry["deepstack"]
    assert [d["layer"] for d in deepstack] == [0, 1, 2]
    assert deepstack[0]["name"] == "out_image_deepstack_0"
    assert deepstack[0]["feeds"] == "in_image_deepstack_0"
    assert deepstack[0]["shape"] == ["IMG", HIDDEN]  # falls back to dyn axis
    assert deepstack[2]["name"] == "out_image_deepstack_2"
    assert deepstack[2]["dtype"] == "f16"


def test_scatter_features_by_mask_replace_and_additive():
    inputs_embeds = torch.zeros((1, 4, 3))
    token_mask = torch.tensor([[False, True, False, True]])
    features = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    replaced = scatter_features_by_mask(
        inputs_embeds=inputs_embeds, token_mask=token_mask, features=features
    )
    assert torch.equal(replaced[0, 1], features[0])
    assert torch.equal(replaced[0, 3], features[1])
    assert torch.equal(replaced[0, 0], torch.zeros(3))  # unmasked untouched

    base = torch.ones((1, 4, 3))
    added = scatter_features_by_mask(
        inputs_embeds=base,
        token_mask=token_mask,
        features=features,
        additive=True,
    )
    assert torch.equal(added[0, 1], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.equal(added[0, 0], torch.ones(3))  # unmasked unchanged


def test_scatter_features_by_mask_rejects_count_mismatch():
    with pytest.raises(ValueError):
        scatter_features_by_mask(
            inputs_embeds=torch.zeros((1, 4, 3)),
            token_mask=torch.tensor([[False, True, False, True]]),
            features=torch.ones((3, 3)),  # 3 features for 2 slots
        )


def test_scatter_features_by_mask_empty_is_noop():
    embeds = torch.arange(12.0).view(1, 4, 3)
    out = scatter_features_by_mask(
        inputs_embeds=embeds,
        token_mask=torch.zeros((1, 4), dtype=torch.bool),
        features=torch.zeros((0, 3)),
    )
    assert torch.equal(out, embeds)


def test_reset_special_ids_to_filler():
    ids = torch.tensor([[0, 1, 2, 3, 1]])
    reset_special_ids_to_filler(ids, {1, 2}, vocab_size=4)
    assert 1 not in ids.tolist()[0]
    assert 2 not in ids.tolist()[0]

    # no room for a non-special filler -> left untouched (no crash)
    ids2 = torch.tensor([[0, 1]])
    reset_special_ids_to_filler(ids2, {0, 1}, vocab_size=2)
    assert ids2.tolist() == [[0, 1]]


def test_resolve_submodule():
    tree = SimpleNamespace(a=SimpleNamespace(b=123))
    assert resolve_submodule(tree, "a.b") == 123
    with pytest.raises(T2NErrorConsistency):
        resolve_submodule(tree, "a.missing")


def test_loosest_tolerance_picks_looser():
    assert (
        _loosest_tolerance(
            TractCheckTolerance.APPROXIMATE, TractCheckTolerance.ULTRA
        )
        == TractCheckTolerance.ULTRA
    )
    # never tightens below the caller's looser choice
    assert (
        _loosest_tolerance(TractCheckTolerance.ULTRA, TractCheckTolerance.VERY)
        == TractCheckTolerance.ULTRA
    )


def test_cleanup_runs_when_forward_raises():
    class _RaisingHandler(FakeVisionEncoderHandler):
        def call_encoder(self, *, model, state_context, wrapper):
            raise RuntimeError("boom")

        def cleanup(self, *, state_context, wrapper):
            wrapper.cleaned = True

    wrapper = BaseEncoder(FakeEncoderModule(), _RaisingHandler())
    wrapper.cleaned = False
    with pytest.raises(RuntimeError):
        wrapper(torch.zeros(4, 3, HIDDEN))
    assert wrapper.cleaned, "cleanup() must run even when the forward raises"


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
    assert "deepstack" not in manifest["encoders"][0]


def test_dump_llm_refuses_multimodal_checkpoint():
    """`dump_llm` refuses a multimodal model_type instead of under-exporting.

    The decoder-only path would drop the modality tower(s) + manifest; the
    guard raises and points at the dedicated multimodal command.
    """
    with pytest.raises(T2NErrorMisuse) as exc:
        _reject_multimodal_in_llm_dump(FAKE_ARCH)
    msg = str(exc.value)
    assert "t2n_export_multimodal_to_tract" in msg
    # it names the modalities the decoder-only path would drop
    assert "audio" in msg and "vision" in msg
    # a text-only arch passes through untouched (no raise)
    _reject_multimodal_in_llm_dump("some_text_only_arch")
