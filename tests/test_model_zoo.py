"""Tests canonical models."""

import os

import pytest
import torch
import torchaudio
from torchaudio import models as audio_mdl
from torchvision import models as vision_mdl
from transformers import AlbertModel, AlbertTokenizer

from tests.shifted_window_attention_patch import (
    ExportableShiftedWindowAttention,
    ExportableSwinTransformerBlock,
)
from torch_to_nnef.inference_target import TractNNEF

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    remove_weight_norm,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))

test_suite = TestSuiteInferenceExactnessBuilder(TRACT_INFERENCES_TO_TESTS)

test_suite.add(
    torch.rand(1, 3, 224, 224),
    vision_mdl.alexnet(pretrained=True, progress=False),
    test_name="alexnet",
)

test_suite.add(
    torch.rand(1, 3, 256, 256),
    vision_mdl.resnet50(pretrained=True, progress=False),
    test_name="resnet50",
)

test_suite.add(
    torch.rand(1, 3, 256, 256),
    vision_mdl.mnasnet1_0(pretrained=True, progress=False),
    test_name="mnasnet1_0",
)
test_suite.add(
    torch.rand(1, 3, 256, 256),
    vision_mdl.efficientnet_b0(pretrained=True, progress=False),
    test_name="efficientnet_b0",
)

test_suite.add(
    torch.rand(1, 1, 100, 64),
    audio_mdl.DeepSpeech(64, n_hidden=256),
    test_name="deepspeech",
)

if hasattr(audio_mdl, "Conformer"):

    class ConformerWrapper(torch.nn.Module):
        """Avoid returning length that is not edited
        torch_to_nnef forbid to return same tensor as inputed
        by the model as this means this output is not needed
        and may introduce silent variable name alterations.
        """

        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, x, length):
            out, _ = self.model(x, length)
            return out

    # and "0.21.2" <= tract_version()
    test_suite.add(
        (torch.rand(1, 100, 64), torch.tensor([100])),
        ConformerWrapper(
            audio_mdl.Conformer(
                64,
                num_heads=2,
                num_layers=2,
                ffn_dim=128,
                depthwise_conv_kernel_size=31,
            )
        ),
        test_name="conformer",
        inference_conditions=lambda i: isinstance(i, TractNNEF)
        and i.version >= "0.21.2",
    )

if hasattr(audio_mdl, "ConvTasNet"):
    test_suite.add(
        (torch.rand(1, 1, 1024),),
        audio_mdl.ConvTasNet(
            num_sources=2,
            # encoder/decoder parameters
            enc_kernel_size=16,
            enc_num_feats=512,
            # mask generator parameters
            msk_kernel_size=3,
            msk_num_feats=128,
            msk_num_hidden_feats=512,
            msk_num_layers=2,
            msk_num_stacks=3,
            msk_activate="sigmoid",
        ),
        test_name="convtasnet",
    )


class FilterOut(torch.nn.Module):
    def __init__(self, wav2vec2_encoder):
        super().__init__()
        self.wav2vec2_encoder = wav2vec2_encoder.cpu()

    def forward(
        self,
        features,
    ):
        return self.wav2vec2_encoder.transformer(features, attention_mask=None)


wav2vec2_model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
remove_weight_norm(wav2vec2_model)
wav2vec2_model.eval()


test_suite.add(
    torch.rand(1, 2, 768),
    FilterOut(wav2vec2_model.encoder),
    test_name="wav2vec2_encoder",
)

# export pretrained work but multi_head might give different values
test_suite.add(
    torch.rand(1, 3, 224, 224),
    vision_mdl.vit_b_16(pretrained=False),
    test_name="vit_b_16",
)


# swin_transformer {
# need slice with stride
if hasattr(vision_mdl, "swin_transformer"):
    vision_mdl.swin_transformer.ShiftedWindowAttention = (
        ExportableShiftedWindowAttention
    )
    vision_mdl.swin_transformer.SwinTransformerBlock = (
        ExportableSwinTransformerBlock
    )
    data = torch.rand(1, 3, 224, 224)
    mdl = vision_mdl.swin_t()  # pretrained=False
    mdl.eval()
    mdl(data)  # precompute attn mask and few shapes
    #  and "0.19.0" < tract_version()
    test_suite.add(
        data,
        mdl,
        test_name="swin_transformer",
        inference_conditions=lambda i: isinstance(i, TractNNEF)
        and i.version > "0.19.0",
    )

# }


# albert {

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
inputs = tokenizer("Hello, I am happy", return_tensors="pt")


class ALBERTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AlbertModel.from_pretrained("albert-base-v2")

    def forward(self, *args):
        outputs = self.model(*args)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states


# }

test_suite.add(tuple(inputs.values()), ALBERTModel(), test_name="albert")


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_model_export(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
