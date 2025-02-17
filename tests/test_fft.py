import pytest
import torch
from torch import nn
from torchaudio import transforms

from tests.wrapper import UnaryPrimitive
from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from torch_to_nnef.inference_target import TractNNEF


class MyFFT(nn.Module):
    def forward(self, x):
        x = torch.fft.fft(x)
        x = torch.fft.ifft(x)
        x = torch.view_as_real(x)
        x = x[:, :, 0]
        return x


class MySTFT(nn.Module):
    def forward(self, x):
        spec_f = torch.stft(
            input=x,
            n_fft=6,
            hop_length=1,
            win_length=6,
            window=torch.tensor([0.1, 0.5, 0.5, 0.1, 0.1, 0.1]),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.view_as_real(spec_f)


test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


def cond_tract_gt_0_20_7(i) -> bool:
    return isinstance(i, TractNNEF) and i.version > "0.20.7"


def add_test(*args):
    global test_suite
    test_suite.add(*args, inference_conditions=cond_tract_gt_0_20_7)


add_test(torch.FloatTensor([[0, 1], [2, 3]]), MyFFT())
add_test(torch.arange(12).float(), MySTFT())
add_test(
    torch.arange(4.0).reshape((2, 2)),
    UnaryPrimitive(lambda x: torch.view_as_complex(x).abs()),
)
add_test(
    torch.arange(400 * 2).float() / 200,
    transforms.Spectrogram(),
)
add_test(
    torch.arange(400 * 2).float() / 200,
    transforms.MelSpectrogram(),
)
""" precision issue for now
add_test(
    torch.arange(400 * 2).float() / 400,
    transforms.MFCC(),
)
"""


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_complex_and_fft_export(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
