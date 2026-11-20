from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX
from torch import nn

from torch_to_nnef.op.custom_extractors.mamba2 import MAMBA2_MIN_TRACT_VERSION
from torch_to_nnef.op.custom_extractors.moe import (
    MOE_FFN_NEMOTRON_H_MIN_TRACT_VERSION,
)
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.opaque import (
    OpaqueTensorRef,
    set_opaque_tensor_in_params_as_ref,
)
from torch_to_nnef_llm.config import HFConfigHelper
from torch_to_nnef_llm.models.base import BaseCausal
from torch_to_nnef_llm.models.handlers import (
    NemotronHOmniArchitectureHandler,
    NemotronHOmniAudioEncoderHandler,
    NemotronHOmniVisionEncoderHandler,
    get_handler,
)
from torch_to_nnef_llm.models.handlers.nemotron_h_omni import (
    NemotronHOmniAudioEncoder,
    NemotronHOmniVisionEncoder,
)

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "NemotronH_Omni_Reasoning_V3"):
    pytest.skip(
        "transformers too old for nemotron_h_omni", allow_module_level=True
    )

IMAGE_TOKEN, AUDIO_TOKEN = 10, 11


def _config():
    text = dict(
        vocab_size=128,
        hidden_size=32,
        layers_block_type=[
            "linear_attention",
            "moe",
            "full_attention",
            "linear_attention",
        ],
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=40,
        n_routed_experts=8,
        num_experts_per_tok=3,
        routed_scaling_factor=2.5,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    vision = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=16,
        max_img_size=128,
        # as C-RADIOv4-H: 10 summary tokens, not a whole number of merge cells
        num_cls_tokens=3,
        num_registers=7,
        summary_idxs=[0, 1, 2],
        video_temporal_patch_size=2,
    )
    audio = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        conv_kernel_size=9,
        subsampling_conv_channels=16,
        num_mel_bins=16,
        projection_hidden_size=48,
        projection_bias=False,
    )
    return transformers.NemotronH_Omni_Reasoning_V3_Config(
        text_config=text,
        vision_config=vision,
        audio_config=audio,
        vision_hidden_size=32,
        projector_hidden_size=48,
        image_token_id=IMAGE_TOKEN,
        audio_token_id=AUDIO_TOKEN,
    )


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = transformers.NemotronH_Omni_Reasoning_V3(_config()).float().eval()
    nemotron_h = pytest.importorskip(
        "transformers.models.nemotron_h.modeling_nemotron_h"
    )
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nemotron_h.NemotronHTopkRouter):
                module.e_score_correction_bias.normal_(std=0.3)
            if isinstance(module, nemotron_h.NemotronHMamba2Mixer):
                module.time_step_limit = (0.0, float("inf"))
    return model


def test_nemotron_h_omni_handlers_are_registered():
    assert get_handler("nemotron_h_omni") is NemotronHOmniArchitectureHandler


@pytest.mark.parametrize("grid", [(2, 3), (5, 2)])
def test_vision_encoder_matches_native(tiny_model, grid):
    """Merge-cell ordered patches give the native features for any grid."""
    mh, mw = grid
    patches = torch.randn(mh, mw, 2, 2, 3 * 16 * 16)
    row_major = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3 * 16 * 16)
    grid_hw = torch.tensor([[2 * mh, 2 * mw]])
    with torch.no_grad():
        expected = tiny_model.get_image_features(row_major, grid_hw)
        actual = NemotronHOmniVisionEncoder(tiny_model)(patches)
    torch.testing.assert_close(actual, expected.pooler_output[0])


def test_vision_weights_reload_after_decoder_export(tiny_model, tmp_path):
    """The decoder export leaves offloaded weights wrapped as references."""
    vision = tiny_model.vision_model
    expected = {n: p.detach().clone() for n, p in vision.named_parameters()}
    for module in vision.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            offloaded = OffloadedTensor.from_original_tensor(
                param.detach(), name, offload_dir=tmp_path
            )
            setattr(module, name, nn.Parameter(offloaded, requires_grad=False))
    set_opaque_tensor_in_params_as_ref(vision)
    NemotronHOmniVisionEncoderHandler().prepare_model_for_export(tiny_model)
    for name, param in vision.named_parameters():
        assert not isinstance(param, OpaqueTensorRef), name
        torch.testing.assert_close(param.data, expected[name])


@pytest.mark.parametrize("frames", [41, 40, 57, 2])
def test_audio_encoder_matches_native(tiny_model, frames):
    """Same embeddings as the native masked call on the processor's output.

    The processor's mask covers all frames of a clip but the last one, which
    it zeroes; 41 and 57 frames leave a multiple of the subsampling factor
    valid, where cropping to the mask would drop an embedding.
    """
    features = torch.randn(1, frames, 16)
    features[:, -1] = 0
    mask = torch.ones(1, frames, dtype=torch.long)
    mask[:, -1] = 0
    with torch.no_grad():
        expected = tiny_model.get_audio_features(features, mask).pooler_output
        NemotronHOmniAudioEncoderHandler().prepare_model_for_export(tiny_model)
        actual = NemotronHOmniAudioEncoder(tiny_model)(features)
    assert actual.shape[0] == (frames + 7) // 8
    torch.testing.assert_close(actual, expected)


def _decoder_spec(model):
    return NemotronHOmniArchitectureHandler().build_input_spec(
        tokenizer=lambda *a, **kw: SimpleNamespace(
            input_ids=torch.arange(20, 52).reshape(1, -1)
        ),
        config_helper=HFConfigHelper(model.config),
        inputs_dtype=torch.float32,
        sample_text="",
        n_input_tokens=6,
        n_past_input_tokens=0,
    )


def test_decoder_matches_native(tiny_model):
    """Spliced embeddings and chunked state match the native LM in one go."""
    handler = NemotronHOmniArchitectureHandler()
    handler.prepare_model_for_export(tiny_model)
    wrapper = BaseCausal(tiny_model, handler=handler, num_logits_to_keep=0)
    spec = _decoder_spec(tiny_model)
    ids, states = spec.inputs[0], spec.inputs[1:-2]
    image, audio = spec.inputs[-2:]
    assert ids[0, :3].tolist() == [IMAGE_TOKEN, IMAGE_TOKEN, AUDIO_TOKEN]
    with torch.no_grad():
        first = wrapper(ids[:, :4], *states, image, audio)
        second = wrapper(ids[:, 4:], *first[1:])
        lm = tiny_model.language_model
        embeds = lm.get_input_embeddings()(ids)
        embeds[0, :2] = image
        embeds[0, 2] = audio[0]
        expected = lm(inputs_embeds=embeds, use_cache=False).logits
    torch.testing.assert_close(first[0], expected[:, :4], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(second[0], expected[:, 4:], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_nemotron_h_omni_multimodal_export(
    tiny_model, tmp_path, inference_target
):
    required = max(
        MAMBA2_MIN_TRACT_VERSION, MOE_FFN_NEMOTRON_H_MIN_TRACT_VERSION
    )
    if inference_target.version < required:
        pytest.skip(f"NemotronH Omni requires tract >= {required}")
    cli = inference_target.tract_cli
    from torch_to_nnef_llm.exporter import LM_VAR_SCHEME, LLMExporter
    from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

    # native audio embeddings at lengths other than the traced one, taken
    # before the export, which rewrites the model in place
    audio_refs = []
    for frames in (57, 40, 2):
        features = torch.randn(1, frames, 16)
        features[:, -1] = 0
        mask = torch.ones(1, frames, dtype=torch.long)
        mask[:, -1] = 0
        with torch.no_grad():
            native = tiny_model.get_audio_features(features, mask)
        audio_refs.append((features, native.pooler_output))

    exporter = MultiModalExporter(
        LLMExporter(
            hf_model_causal=tiny_model,
            tokenizer=lambda *a, **kw: SimpleNamespace(
                input_ids=torch.arange(20, 84).reshape(1, -1)
            ),
            force_module_dtype="f32",
        )
    )
    assert [type(h) for h in exporter.encoder_handlers] == [
        NemotronHOmniVisionEncoderHandler,
        NemotronHOmniAudioEncoderHandler,
    ]
    target = exporter.decoder_exporter.build_inference_target(
        tract_specific_path=cli.tract_path
    )
    exporter.export(tmp_path / "omni", target, LM_VAR_SCHEME)
    assert (tmp_path / "omni" / "multimodal.json").exists()
    for features, expected in audio_refs:
        in_path, out_path = tmp_path / "in.npz", tmp_path / "out.npz"
        np.savez(in_path, input_features=features.numpy())
        np.savez(out_path, out_audio_embeddings=expected.numpy())
        cli.assert_io(
            tmp_path / "omni" / "audio" / "model.nnef.tgz",
            in_path,
            out_path,
            check_tolerance=target.check_io_tolerance,
        )
