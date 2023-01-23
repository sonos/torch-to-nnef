import pytest
import torch
from torch import nn

from tests.utils import check_model_io_test, id_tests
from torch_to_nnef.tract import tract_version_lower_than

# from torchaudio import transforms


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


INPUT_AND_MODELS = []

if not tract_version_lower_than("0.19.0"):
    INPUT_AND_MODELS += [(torch.FloatTensor([[0, 1], [2, 3]]), MyFFT())]
    INPUT_AND_MODELS += [
        (
            torch.arange(12).float(),
            MySTFT(),
        )
    ]
    # INPUT_AND_MODELS += [
    # (
    # torch.arange(400 * 2).float(),
    # transforms.Spectrogram(),
    # )
    # ]


@pytest.mark.parametrize(
    "test_input,model", INPUT_AND_MODELS, ids=id_tests(INPUT_AND_MODELS)
)
def test_complex_export(test_input, model):
    """Test simple models"""
    check_model_io_test(model=model, test_input=test_input)
