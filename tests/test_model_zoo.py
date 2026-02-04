"""Tests canonical models."""

import contextlib
import os
import platform
import warnings
from copy import deepcopy

import pytest
import torch
import torchaudio
import torchvision
from torch.jit._trace import TracerWarning
from torchaudio import models as audio_mdl
from transformers import AlbertModel, AlbertTokenizer

from torch_to_nnef.inference_target.tract import TractCheckTolerance

with contextlib.suppress(ImportError):
    from tests.shifted_window_attention_patch import (
        ExportableShiftedWindowAttention,
        ExportableSwinTransformerBlock,
    )

from torch_to_nnef.inference_target import TractNNEF

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    remove_weight_norm,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))

test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


def get_vision_model(vision_model_name, pretrained=True, progress=False):
    """Handle torchvison API evolution."""
    class_name = vision_model_name.lower()
    if not hasattr(torchvision.models, class_name):
        return
    mdl_cls = getattr(torchvision.models, class_name)
    if torchvision.__version__ < "0.13.0":
        model = mdl_cls(pretrained=pretrained, progress=progress)
    else:
        weights_enum = getattr(
            torchvision.models,
            f"{vision_model_name}_Weights",
        )
        model = mdl_cls(
            weights=weights_enum.DEFAULT if pretrained else None,
            progress=progress,
        )
    return model


def add_vision_model_test_suite(input_size, name):
    inps = torch.rand(1, 3, input_size, input_size)
    mdl = get_vision_model(name)
    base_name = name.lower()
    if mdl is None:
        print(
            f"missing '{base_name}' in vision "
            f"package: {torchvision.__version__}"
        )
        return
    test_suite.add(
        (inps,),
        mdl,
        test_name=base_name,
    )


add_vision_model_test_suite(224, "AlexNet")
add_vision_model_test_suite(256, "ResNet50")
add_vision_model_test_suite(256, "MNASNet1_0")
add_vision_model_test_suite(256, "EfficientNet_B0")
add_vision_model_test_suite(224, "ViT_B_16")
# swin_transformer {
# need slice with stride
if hasattr(torchvision.models, "swin_transformer"):
    torchvision.models.swin_transformer.ShiftedWindowAttention = (
        ExportableShiftedWindowAttention
    )
    torchvision.models.swin_transformer.SwinTransformerBlock = (
        ExportableSwinTransformerBlock
    )
    data = torch.rand(1, 3, 224, 224)
    mdl = torchvision.models.swin_t()  # pretrained=False
    mdl.eval()
    mdl(data)  # precompute attn mask and few shapes
    test_suite.add(
        data,
        mdl,
        test_name="swin_transformer",
        inference_conditions=lambda i: (
            isinstance(i, TractNNEF) and i.version > "0.19.0"
        ),
    )

# }


test_suite.add(
    torch.rand(1, 1, 100, 64),
    audio_mdl.DeepSpeech(64, n_hidden=256),
    test_name="deepspeech",
)

if hasattr(audio_mdl, "Conformer"):

    class ConformerWrapper(torch.nn.Module):
        """Wrap Conformer for export.

        Avoid returning length that is not edited
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
        inference_conditions=lambda i: (
            isinstance(i, TractNNEF) and i.version >= "0.21.2"
        ),
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


wav2vec2_model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
remove_weight_norm(wav2vec2_model)
wav2vec2_model.eval()


test_suite.add(
    (torch.rand(1, 1, 512),),
    wav2vec2_model.encoder,
    test_name="wav2vec2_encoder",
)
# test_suite.add(
#     (torch.rand(1, 16000),),
#     wav2vec2_model,
#     test_name="wav2vec2",
# ) # 1: eval() called on a Dummy op. This is a bug.

# export pretrained work but multi_head might give different values


# albert {


class ALBERTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AlbertModel.from_pretrained("albert-base-v2")

    def forward(self, *args):
        outputs = self.model(*args)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states


# }


def inference_modifier_tract_tol_arm(inference_target):
    inference_target = deepcopy(inference_target)
    if (
        isinstance(inference_target, TractNNEF)
        and "arm" in platform.uname().machine.lower()
    ):
        inference_target.check_io_tolerance = TractCheckTolerance.SUPER
    return inference_target


def tract_upper_than_21_7_or_not_arm(inference_target):
    return (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= "0.21.7"
    ) or "arm" not in platform.uname().machine.lower()


try:
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    inputs = tokenizer("Hello, I am happy", return_tensors="pt")

    test_suite.add(
        tuple(inputs.values()),
        ALBERTModel(),
        test_name="albert",
        inference_conditions=tract_upper_than_21_7_or_not_arm,
        inference_modifier=inference_modifier_tract_tol_arm,
    )
except ImportError:
    print("missing deps to test on albert model")


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_model_export(id, test_input, model, inference_target):
    """Test simple models."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TracerWarning)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )
