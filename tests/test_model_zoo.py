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
from torch_to_nnef.tract import tract_version_greater_than

from .utils import (  # noqa: E402
    check_model_io_test,
    remove_weight_norm,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))

INPUT_AND_MODELS = [
    (
        torch.rand(1, 3, 224, 224),
        vision_mdl.alexnet(pretrained=True, progress=False),
    ),
]
INPUT_AND_MODELS += [
    (
        torch.rand(1, 3, 256, 256),
        model,
    )
    for model in [
        vision_mdl.resnet50(pretrained=True, progress=False),
        # vision_mdl.regnet_y_8gf(
        # pretrained=True
        # ),  # works - similar to resnet
        vision_mdl.mnasnet1_0(
            pretrained=True, progress=False
        ),  # works - nas similar to resnet
        vision_mdl.efficientnet_b0(pretrained=True, progress=False),
    ]
]

INPUT_AND_MODELS += [
    (torch.rand(1, 1, 100, 64), model)
    for model in [
        audio_mdl.DeepSpeech(64, n_hidden=256),
    ]
]

if hasattr(audio_mdl, "Conformer"):
    INPUT_AND_MODELS += [
        ((torch.rand(1, 100, 64), torch.tensor([100])), model)
        for model in [
            audio_mdl.Conformer(
                64,
                num_heads=2,
                num_layers=2,
                ffn_dim=128,
                depthwise_conv_kernel_size=31,
            )
        ]
    ]

if hasattr(audio_mdl, "ConvTasNet"):
    INPUT_AND_MODELS += [
        # input shape: batch, channel==1, frames
        (
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
        )
    ]


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

INPUT_AND_MODELS += [
    # (
    # (
    # torch.rand(1, 2048),  # batch, num_frames
    # torch.tensor([2048]),
    # ),
    # # torch_to_nnef.torch_graph.CheckError: Arity Missmatch 1 instead of 2
    # model.feature_extractor,
    # )
    (
        # torch.rand(1, 2, 512),
        torch.rand(1, 2, 768),
        FilterOut(wav2vec2_model.encoder),
    ),
]

# export pretrained work but multi_head giving slightly different values
INPUT_AND_MODELS += [
    (torch.rand(1, 3, 224, 224), vision_mdl.vit_b_16(pretrained=False)),
]


# swin_transformer {
# need slice with stride
if hasattr(vision_mdl, "swin_transformer") and tract_version_greater_than(
    "0.19.0"
):
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
    INPUT_AND_MODELS += [(data, mdl)]

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

INPUT_AND_MODELS += [(tuple(inputs.values()), ALBERTModel())]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_model_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
