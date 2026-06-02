from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from torchaudio import transforms

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    change_dynamic_axes,
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


class MyFFTFreq(nn.Module):
    """`fft.fftfreq` / `fft.rfftfreq` inside `forward`.

    Trace materializes the op as a constant freq vector at export.
    """

    def __init__(self, n, d=1.0, kind="fft"):
        super().__init__()
        self.n = n
        self.d = d
        self.kind = kind

    def forward(self, x):
        if self.kind == "fft":
            freq = torch.fft.fftfreq(self.n, d=self.d, dtype=x.dtype)
        else:
            freq = torch.fft.rfftfreq(self.n, d=self.d, dtype=x.dtype)
        return x + freq


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
# Regression: complex IR tensors whose *logical* last axis equals 2 used
# to be mis-detected as already view-tagged, so the pre-pass + handlers
# skipped the trailing-2 promotion. The result was an IR/storage rank
# mismatch -- the second FFT in MyFFT([2,2]) was emitted on the wrong
# axis, and `fft.irfft(fft.rfft(x, dim=0), dim=0)` on a length-2 input
# tripped tract's deser bounds check. Both should now export cleanly.
add_test(torch.arange(2.0), MyRFFT(dim=0))
add_test(torch.arange(2.0), MyIRFFT(dim=0))
add_test(torch.arange(24.0).reshape(2, 3, 4), MyFFTN(dims=[1, 2]))
add_test(torch.arange(24.0).reshape(2, 3, 4), MyIFFTN(dims=[1, 2]))
add_test(torch.arange(8.0), MyHammingWindowScaled(8, "hamming"))
add_test(torch.arange(8.0), MyHammingWindowScaled(8, "blackman"))
add_test(torch.arange(8.0), MyHammingWindowScaled(8, "kaiser"))
add_test(torch.arange(8.0), MyFFTFreq(n=8, d=1.0, kind="fft"))
add_test(torch.arange(8.0), MyFFTFreq(n=8, d=0.5, kind="fft"))
# rfftfreq returns n//2 + 1 elements.
add_test(torch.arange(5.0), MyFFTFreq(n=8, d=1.0, kind="rfft"))
add_test(torch.arange(5.0), MyFFTFreq(n=8, d=0.25, kind="rfft"))


class MyComplex(nn.Module):
    """Build a complex tensor and unfold to real for comparison."""

    def forward(self, real, imag):
        return torch.view_as_real(torch.complex(real, imag))


class MyConj(nn.Module):
    """Conjugate a complex tensor (with `resolve_conj` to materialise)."""

    def forward(self, x):
        c = torch.view_as_complex(x)
        return torch.view_as_real(torch.conj(c).resolve_conj())


class MyConjPhysical(nn.Module):
    """`torch.conj_physical(complex)` -> conjugate (materialised)."""

    def forward(self, x):
        c = torch.view_as_complex(x)
        return torch.view_as_real(torch.conj_physical(c))


class MyReal(nn.Module):
    """`torch.real(complex)` -> real-part tensor."""

    def forward(self, x):
        return torch.real(torch.view_as_complex(x))


class MyImag(nn.Module):
    """`torch.imag(complex)` -> imag-part tensor."""

    def forward(self, x):
        return torch.imag(torch.view_as_complex(x))


class MyComplexThenFFT(nn.Module):
    """Chain `torch.complex` into `torch.fft.fft`.

    Locks in that `complex` (and by symmetry `polar`) leave the output's
    view-tagged dtype/shape alone so the downstream FFT handler doesn't
    re-pad the trailing complex axis. Regression for the pre-pass
    invariant.
    """

    def forward(self, real, imag):
        return torch.view_as_real(torch.fft.fft(torch.complex(real, imag)))


add_test(
    (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.5, -1.0], [-2.0, 1.5]]),
    ),
    MyComplex(),
)
add_test(
    (
        torch.tensor([[1.0, 2.0, -0.5], [3.0, 4.0, 0.25]]),
        torch.tensor([[0.5, -1.0, 0.1], [-2.0, 1.5, -0.3]]),
    ),
    MyComplexThenFFT(),
)
add_test(
    torch.tensor([[[1.0, 2.0], [3.0, -1.5]], [[-0.5, 0.7], [2.0, -2.0]]]),
    MyConj(),
)
add_test(
    torch.tensor([[[1.0, 2.0], [3.0, -1.5]], [[-0.5, 0.7], [2.0, -2.0]]]),
    MyConjPhysical(),
)
add_test(
    torch.tensor([[[1.0, 2.0], [3.0, -1.5]], [[-0.5, 0.7], [2.0, -2.0]]]),
    MyReal(),
)
add_test(
    torch.tensor([[[1.0, 2.0], [3.0, -1.5]], [[-0.5, 0.7], [2.0, -2.0]]]),
    MyImag(),
)
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


class MyISTFT(nn.Module):
    """`stft -> istft` round-trip on a real signal.

    Locks in the iSTFT handler: builds a one-sided complex spectrum via
    `torch.stft(return_complex=True)`, then reconstructs the signal via
    `torch.istft`. With a COLA-satisfying window (Hann at hop=n_fft/2)
    the round-trip is the identity modulo a small symmetric crop and
    floating-point error -- the same coverage that fed the DPDFNet
    pulse-mode pipeline regression.
    """

    def __init__(
        self,
        n_fft: int = 8,
        hop_length: int = 4,
        center: bool = True,
        window_name: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        if window_name == "hann":
            window = torch.hann_window(n_fft, dtype=torch.float32)
        elif window_name == "hamming":
            window = torch.hamming_window(n_fft, dtype=torch.float32)
        elif window_name == "sqrt_hann":
            # sqrt(Hann) satisfies w^2-COLA at hop = n_fft / 2 (plain Hann
            # does not -- it's only w-COLA there). Required for the
            # dynamic-axes istft branch.
            window = torch.hann_window(n_fft, dtype=torch.float32).sqrt()
        else:
            raise ValueError(window_name)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            return_complex=False,
        )


# Hann at hop=n_fft/2 — the canonical COLA case.
test_suite.add(
    torch.sin(torch.arange(64, dtype=torch.float32) * 0.5).unsqueeze(0),
    MyISTFT(n_fft=8, hop_length=4),
    inference_conditions=_cond_stft_ge_0_22,
)
# Hamming window — different scale but still COLA-satisfying at hop=n_fft/2.
test_suite.add(
    torch.sin(torch.arange(64, dtype=torch.float32) * 0.5).unsqueeze(0),
    MyISTFT(n_fft=8, hop_length=4, window_name="hamming"),
    inference_conditions=_cond_stft_ge_0_22,
)
# `center=False` with Hamming (non-vanishing at the boundaries; Hann
# would fail torch's "window overlap add min" check here): covers the
# no-crop finalize branch of the istft handler.
test_suite.add(
    torch.sin(torch.arange(64, dtype=torch.float32) * 0.5).unsqueeze(0),
    MyISTFT(n_fft=8, hop_length=4, center=False, window_name="hamming"),
    inference_conditions=_cond_stft_ge_0_22,
)
# Rank-3 input (no batch) — synthetic batch axis added by the handler.
test_suite.add(
    torch.sin(torch.arange(64, dtype=torch.float32) * 0.5),
    MyISTFT(n_fft=8, hop_length=4),
    inference_conditions=_cond_stft_ge_0_22,
)
# Multi-batch input (B>1) — exercises the OLA chain shape threading.
test_suite.add(
    torch.sin(torch.arange(3 * 64, dtype=torch.float32) * 0.5).reshape(3, 64),
    MyISTFT(n_fft=8, hop_length=4),
    inference_conditions=_cond_stft_ge_0_22,
)
# Dynamic-axes istft with sqrt(Hann): exercises the COLA-constant
# divisor branch (the export bakes a scalar instead of a full-length
# vector). Plain Hann at hop=n_fft/2 fails the w^2-COLA check; sqrt-Hann
# is the canonical w^2-COLA window at that hop.
test_suite.add(
    torch.sin(torch.arange(64, dtype=torch.float32) * 0.5).unsqueeze(0),
    MyISTFT(n_fft=8, hop_length=4, window_name="sqrt_hann"),
    inference_conditions=_cond_stft_ge_0_22,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes={"input_0": {1: "STREAM"}}
    ),
)


class MyComplexTranspose(nn.Module):
    """Transpose a complex tensor's logical axes.

    Exercises the view-tagged complex `transpose` handler: the IR carries
    the trailing-2 axis as storage, but PyTorch's transpose only sees
    the logical (freq, time) axes -- the handler must build a storage
    permutation that swaps those two and leaves the complex axis alone.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.view_as_complex(x)
        return torch.view_as_real(c.transpose(0, 1))


test_suite.add(
    torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2),
    MyComplexTranspose(),
    inference_conditions=cond_tract_gt_0_20_7,
)
# Same shape under dynamic axes on logical axis 0: the streaming dim
# survives the storage permutation so the transpose's downstream
# consumers still see a symbolic axis.
test_suite.add(
    torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2),
    MyComplexTranspose(),
    inference_conditions=lambda i: (
        isinstance(i, TractNNEF) and i.version >= "0.21.5"
    ),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes={"input_0": {0: "S"}}
    ),
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
