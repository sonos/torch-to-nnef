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
from torch_to_nnef.tract import tract_version

from .utils import (  # noqa: E402
    check_model_io_test,
    remove_weight_norm,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))

INPUT_AND_MODELS = [
    (
        "alexnet",
        torch.rand(1, 3, 224, 224),
        vision_mdl.alexnet(pretrained=True, progress=False),
    ),
]
INPUT_AND_MODELS += [
    (
        model[0],
        torch.rand(1, 3, 256, 256),
        model[1],
    )
    for model in [
        ("resnet50", vision_mdl.resnet50(pretrained=True, progress=False)),
        # vision_mdl.regnet_y_8gf(
        # pretrained=True
        # ),  # works - similar to resnet
        (
            "mnasnet1_0",
            vision_mdl.mnasnet1_0(pretrained=True, progress=False),
        ),  # works - nas similar to resnet
        (
            "efficientnet_b0",
            vision_mdl.efficientnet_b0(pretrained=True, progress=False),
        ),
    ]
]

INPUT_AND_MODELS += [
    (
        "deepspeech",
        torch.rand(1, 1, 100, 64),
        audio_mdl.DeepSpeech(64, n_hidden=256),
    )
]

if hasattr(audio_mdl, "Conformer") and "0.21.2" <= tract_version():

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

    INPUT_AND_MODELS = [
        (
            "conformer",
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
        )
    ]

if hasattr(audio_mdl, "ConvTasNet"):
    INPUT_AND_MODELS += [
        # input shape: batch, channel==1, frames
        (
            "convtasnet",
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
        "wav2vec2_encoder",
        torch.rand(1, 2, 768),
        FilterOut(wav2vec2_model.encoder),
    ),
]

# export pretrained work but multi_head giving slightly different values
INPUT_AND_MODELS += [
    (
        "vit_b_16",
        torch.rand(1, 3, 224, 224),
        vision_mdl.vit_b_16(pretrained=False),
    ),
]


# swin_transformer {
# need slice with stride
if hasattr(vision_mdl, "swin_transformer") and "0.19.0" < tract_version():
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
    INPUT_AND_MODELS += [("swin_transformer", data, mdl)]

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

INPUT_AND_MODELS += [("albert", tuple(inputs.values()), ALBERTModel())]


def id_tests(items):
    return [i[0] for i in items]


@pytest.mark.parametrize(
    "id,test_input,model", INPUT_AND_MODELS, ids=id_tests(INPUT_AND_MODELS)
)
def test_model_export(id, test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
