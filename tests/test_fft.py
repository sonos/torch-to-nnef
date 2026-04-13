from copy import deepcopy

import pytest
import torch
from torch import nn
from torchaudio import transforms

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from tests.wrapper import UnaryPrimitive
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.utils import torch_version


class MyFFT(nn.Module):
    def forward(self, x):
        x = torch.fft.fft(x)
        x = torch.fft.ifft(x)
        x = torch.view_as_real(x)
        x = x[:, :, 0]
        return x


class MySTFT(nn.Module):
    def __init__(
        self,
        window,
        n_fft=6,
        hop_length=1,
        win_length=6,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.window = window

    def forward(
        self,
        x,
    ):
        spec_f = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        return torch.view_as_real(spec_f)


test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


def cond_tract_gt_0_20_7(i) -> bool:
    return isinstance(i, TractNNEF) and i.version > "0.20.7"


def _cond_stft_supported(i) -> bool:
    """Skip tract 0.21.14/0.21.15 -- slice-fusion bug corrupts STFT."""
    return cond_tract_gt_0_20_7(i) and not (
        isinstance(i, TractNNEF) and "0.21.14" <= i.version <= "0.21.15"
    )


def add_test(*args, stft=False):
    global test_suite
    cond = _cond_stft_supported if stft else cond_tract_gt_0_20_7
    test_suite.add(*args, inference_conditions=cond)


add_test(torch.FloatTensor([[0, 1], [2, 3]]), MyFFT())
add_test(
    torch.arange(12).float(),
    MySTFT(window=torch.tensor([0.1, 0.5, 0.5, 0.1, 0.1, 0.1])),
    stft=True,
)

add_test(
    torch.arange(12).float(),
    MySTFT(
        window=torch.tensor([0.1, 0.5, 0.5, 0.1, 0.1, 0.1]), normalized=True
    ),
    stft=True,
)


add_test(
    torch.arange(4.0).reshape((2, 2)),
    UnaryPrimitive(lambda x: torch.view_as_complex(x).abs()),
)
add_test(
    torch.arange(400 * 2).float() / 200,
    transforms.Spectrogram(),
    stft=True,
)

add_test(
    torch.arange(400 * 2).float() / 200,
    transforms.MelSpectrogram(),
    stft=True,
)


def change_tol_close(it):
    it = deepcopy(it)
    it.check_io_tolerance = TractCheckTolerance.SUPER
    return it


def _cond_stft_ge_0_22(i) -> bool:
    """STFT with win_length < n_fft needs >= 0.22 (skip 0.21.14/0.21.15)."""
    return isinstance(i, TractNNEF) and i.version >= "0.22.0"


if torch_version() >= "1.11.0":
    test_suite.add(
        torch.arange(400 * 2).float() / 400,
        transforms.MFCC(),
        inference_conditions=_cond_stft_ge_0_22,
        inference_modifier=change_tol_close,
    )

test_suite.add(
    torch.arange(12).float(),
    MySTFT(
        n_fft=6,
        win_length=4,
        window=torch.tensor([0.1, 0.5, 0.5, 0.1]),
        normalized=False,
        onesided=False,
    ),
    inference_conditions=_cond_stft_ge_0_22,
)

test_suite.add(
    torch.arange(12).float(),
    MySTFT(
        n_fft=7,
        win_length=4,
        window=torch.tensor([0.1, 0.5, 0.5, 0.1]),
        normalized=False,
        onesided=False,
    ),
    inference_conditions=_cond_stft_ge_0_22,
)
test_suite.add(
    torch.arange(12).float(),
    MySTFT(
        n_fft=7,
        win_length=4,
        window=torch.tensor([0.1, 0.5, 0.5, 0.1]),
        normalized=False,
        center=True,
    ),
    inference_conditions=_cond_stft_ge_0_22,
)


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_complex_and_fft_export(id, test_input, model, inference_target):
    """Test simple models."""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
