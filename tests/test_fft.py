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


class MyRFFT(nn.Module):
    """`rfft` returns a one-sided complex spectrum (last bin = N//2+1)."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.view_as_real(torch.fft.rfft(x, dim=self.dim))


class MyFFTN(nn.Module):
    """N-dim forward FFT followed by view_as_real for IO comparability."""

    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.view_as_real(torch.fft.fftn(x, dim=self.dims))


class MyIRFFT(nn.Module):
    """Reconstruct a real signal from a one-sided complex spectrum."""

    def __init__(self, dim=-1, n=None):
        super().__init__()
        self.dim = dim
        self.n = n

    def forward(self, x):
        # Build a one-sided complex spectrum from the real input by
        # taking its rfft; then irfft to recover the original real
        # signal (within floating-point error). This pattern exercises
        # the irfft handler end-to-end.
        spec = torch.fft.rfft(x, dim=self.dim)
        return torch.fft.irfft(spec, n=self.n, dim=self.dim)


class MyIFFTN(nn.Module):
    """N-dim inverse FFT.

    Input is a real tensor cast to complex with zero imaginary part so
    the trace sees a complex-domain ifftn.
    """

    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        cmplx = torch.view_as_complex(
            torch.stack((x, torch.zeros_like(x)), dim=-1)
        )
        return torch.view_as_real(torch.fft.ifftn(cmplx, dim=self.dims))


class MyHammingWindowScaled(nn.Module):
    """Use a window inside `forward` so the trace materializes the op."""

    def __init__(self, length, win_name):
        super().__init__()
        self.length = length
        self.win_name = win_name

    def forward(self, x):
        if self.win_name == "hamming":
            w = torch.hamming_window(self.length, dtype=x.dtype)
        elif self.win_name == "blackman":
            w = torch.blackman_window(self.length, dtype=x.dtype)
        elif self.win_name == "kaiser":
            w = torch.kaiser_window(self.length, dtype=x.dtype)
        else:
            raise ValueError(self.win_name)
        return x * w


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
add_test(torch.arange(8.0).reshape(2, 4), MyRFFT())
add_test(torch.arange(12.0).reshape(3, 4), MyRFFT(dim=0))
add_test(torch.arange(8.0).reshape(2, 4), MyIRFFT())
add_test(torch.arange(12.0).reshape(3, 4), MyIRFFT(dim=0))
add_test(torch.arange(24.0).reshape(2, 3, 4), MyFFTN(dims=[1, 2]))
add_test(torch.arange(24.0).reshape(2, 3, 4), MyIFFTN(dims=[1, 2]))
add_test(torch.arange(8.0), MyHammingWindowScaled(8, "hamming"))
add_test(torch.arange(8.0), MyHammingWindowScaled(8, "blackman"))
add_test(torch.arange(8.0), MyHammingWindowScaled(8, "kaiser"))
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
