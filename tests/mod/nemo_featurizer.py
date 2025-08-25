"""Featurizer borrowed from nemo package for testing purpose."""

import logging
import math
import random

import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorMissUse,
    T2NErrorNotImplemented,
)

CONSTANT = 1e-5


def splice_frames(x, frame_splicing):
    """Stacks frames together across feature dim.

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]

        # When doing stream capture to a graph, item() is not allowed
        # becuase it calls cudaStreamSynchronize(). Therefore, we are
        # sacrificing some error checking when running with cuda graphs.
        if (
            torch.cuda.is_available()
            and not torch.cuda.is_current_stream_capturing()
            and torch.any(seq_len == 1).item()
        ):
            raise T2NErrorMissUse(
                "normalize_batch with `per_feature` normalize_type received a "
                "tensor of length 1. "
                "This will result in torch.std() returning nan. "
                "Make sure your audio length has enough samples for a single "
                "feature (ex. at least `hop_length` for Mel Spectrograms)."
            )
        time_steps = (
            torch.arange(max_time, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, max_time)
        )
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(
            axis=2
        )
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

        # Subtract 1 in the denominator to correct for the bias.
        x_std = torch.sqrt(
            torch.sum(
                torch.where(
                    valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0
                )
                ** 2,
                axis=2,
            )
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2))
            / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.

    See AudioToMelSpectrogramPreprocessor for args.

    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated ; kept for config compatibility
        stft_conv=False,  # Deprecated ; kept for config compatibility
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and has been removed."
                " The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor."
                " Please set exact_pad to True as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise T2NErrorNotImplemented(
                f"{self} received exact_pad == True, but hop_size was odd."
                " If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length "
                "audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise T2NErrorMissUse(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info("PADDING: %s", pad_to)

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (
            (self.n_fft - self.hop_length) // 2 if exact_pad else None
        )
        self.exact_pad = exact_pad

        if exact_pad:
            logging.info("STFT using exact pad")
        torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "none": None,
        }
        window_fn = torch_windows.get(window)
        window_tensor = (
            window_fn(self.win_length, periodic=False) if window_fn else None
        )
        self.register_buffer("window", window_tensor)

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        import librosa

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=self.n_fft,
                n_mels=nfilt,
                fmin=lowfreq,
                fmax=highfreq,
                norm=mel_norm,
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(
            torch.tensor(max_duration * sample_rate, dtype=torch.float)
        )
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise T2NErrorMissUse(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type

    def stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=not self.exact_pad,
            window=self.window.to(dtype=torch.float, device=x.device),
            return_complex=True,
        )

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise T2NErrorMissUse(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = (
            self.stft_pad_amount * 2
            if self.stft_pad_amount is not None
            else self.n_fft // 2 * 2
        )
        seq_len = (
            torch.floor_divide(
                (seq_len + pad_amount - self.n_fft), self.hop_length
            )
            + 1
        )
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len_unfixed = self.get_seq_len(seq_len)
        # fix for seq_len = 0 for streaming; if size was 0,
        # it is always padded to 1, and normalizer fails
        seq_len = torch.where(
            seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed
        )

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1),
                (self.stft_pad_amount, self.stft_pad_amount),
                "reflect",
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                dim=1,
            )

        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]);
        # so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # disable autocast, otherwise it might be automatically casted to fp16
        # on fp16 compatible GPUs and get NaN values for input value of 65520
        with torch.amp.autocast(x.device.type, enabled=False):
            # dot with filterbank energies
            x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(
                    torch.clamp(x, min=self.log_zero_guard_value_fn(x))
                )
            else:
                raise T2NErrorMissUse("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch,
        # pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device),
            self.pad_value,
        )
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(
                x, (0, self.max_length - x.size(-1)), value=self.pad_value
            )
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(
                    x, (0, pad_to - pad_amt), value=self.pad_value
                )
        return x, seq_len
