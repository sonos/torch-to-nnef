"""Export funasr/fsmn-vad to NNEF via torch-to-nnef with tract I/O check.

Downloads the PyTorch weights + config + CMVN stats from HuggingFace, builds a
pure-PyTorch pipeline (upscale -> torchaudio kaldi fbank -> LFR -> CMVN -> FSMN
-> softmax), then runs `export_model_to_nnef` with `check_io=True` so tract is
invoked on a real audio sample and its output is compared with PyTorch.

Run:
    python export.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as tat
import yaml
from fsmn_encoder import FSMN
from huggingface_hub import hf_hub_download
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef

HF_REPO = "funasr/fsmn-vad"


def download_assets(dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("model.pt", "config.yaml", "am.mvn"):
        paths[name] = Path(
            hf_hub_download(repo_id=HF_REPO, filename=name, local_dir=str(dest))
        )
    return paths


def parse_am_mvn(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse Kaldi-style am.mvn: extract <AddShift> means and <Rescale> vars.

    FunASR applies (inputs + means) * vars (shift then rescale).
    """
    means_list: list[str] = []
    vars_list: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, raw in enumerate(lines):
        item = raw.split()
        if not item:
            continue
        if item[0] == "<AddShift>" and i + 1 < len(lines):
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                means_list = nxt[3 : len(nxt) - 1]
        elif item[0] == "<Rescale>" and i + 1 < len(lines):
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                vars_list = nxt[3 : len(nxt) - 1]
    means = torch.tensor(np.array(means_list, dtype=np.float32))
    rescales = torch.tensor(np.array(vars_list, dtype=np.float32))
    return means, rescales


def apply_lfr(
    feats: torch.Tensor, lfr_m: int = 5, lfr_n: int = 1
) -> torch.Tensor:
    """Stack `lfr_m` consecutive frames with stride `lfr_n`, no unfold.

    feats: (T, F). Returns (T_out, lfr_m * F) where T_out = T - (lfr_m - 1) +
    (lfr_m - 1) // 2.  Left-pad with (lfr_m - 1) // 2 copies of the first frame
    (matches FunASR).

    Currently only supports lfr_n == 1. Uses `end`-relative slicing so all
    pieces share the same symbolic length when exporting with dynamic axes.
    """
    assert lfr_n == 1, "only lfr_n == 1 is supported"
    left_pad_count = (lfr_m - 1) // 2
    left = feats[0:1].expand(left_pad_count, -1)
    padded = torch.cat([left, feats], dim=0)  # (T + left_pad_count, F)
    # All pieces are length padded_len - (lfr_m - 1). Using negative-end slices
    # keeps t2n happy on fixed shapes; only the preprocessor calls this, so we
    # never hit this path under dynamic_axes.
    pieces = []
    for i in range(lfr_m):
        tail = lfr_m - 1 - i
        pieces.append(padded[i:] if tail == 0 else padded[i:-tail])
    return torch.cat(pieces, dim=1)


class FsmnVadPreprocessor(nn.Module):
    """Fixed-length audio -> LFR'd CMVN'd features.

    Runs on a rolling audio buffer: input length and therefore output frame
    count are static, so we do NOT declare a dynamic axis here. The downstream
    encoder is the one that gets pulsed by tract.

    Uses torchaudio.transforms.MelSpectrogram (pre-computed hamming window
    buffer) instead of torchaudio.compliance.kaldi.fbank, because t2n does not
    trace aten::hamming_window invoked at call time. To reduce the numerical
    delta vs the Kaldi fbank the model was trained on, we apply:
      - preemphasis (y[n] = x[n] - 0.97 * x[n-1]) before STFT
      - symmetric hamming window (periodic=False, Kaldi-style)
      - f_min = 20 Hz (Kaldi default)
    Residual differences remain (no per-frame DC offset removal, log vs
    energy-floor semantics).
    """

    def __init__(
        self,
        cmvn_means: torch.Tensor,
        cmvn_rescales: torch.Tensor,
        n_mels: int = 80,
        frame_length_ms: int = 25,
        frame_shift_ms: int = 10,
        sample_rate: int = 16000,
        lfr_m: int = 5,
        lfr_n: int = 1,
        preemph_coef: float = 0.97,
    ):
        super().__init__()
        self.register_buffer("cmvn_means", cmvn_means)
        self.register_buffer("cmvn_rescales", cmvn_rescales)
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.sample_rate = sample_rate
        self.preemph_coef = preemph_coef
        win_length = int(frame_length_ms * sample_rate / 1000)
        hop_length = int(frame_shift_ms * sample_rate / 1000)
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2
        # Symmetric hamming matches Kaldi; torch default is periodic=True.
        self.mel = tat.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=lambda n: torch.hamming_window(n, periodic=False),
            center=False,
            power=2.0,
            f_min=20.0,
            mel_scale="htk",
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (1, N) float32 in [-1, 1]. Returns (1, T', lfr_m * n_mels)."""
        wav = audio * (1 << 15)
        # Preemphasis: y[n] = x[n] - coef * x[n-1]; preserve x[0] on the first
        # sample to mirror Kaldi's behavior on the window boundary.
        preemph = torch.cat(
            [wav[:, :1], wav[:, 1:] - self.preemph_coef * wav[:, :-1]], dim=-1
        )
        mel_pow = self.mel(preemph)  # (1, n_mels, T)
        feats = torch.log(mel_pow.clamp(min=1e-10)).transpose(1, 2).squeeze(0)
        feats = apply_lfr(feats, self.lfr_m, self.lfr_n)
        feats = (feats + self.cmvn_means) * self.cmvn_rescales
        return feats.unsqueeze(0)


class FsmnVadEncoder(nn.Module):
    """Features -> FSMN probs. Time axis declared dynamic for tract pulse."""

    def __init__(self, encoder: FSMN):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T', input_dim). Returns (B, T', output_dim) softmax."""
        return self.encoder(feats)


def build_encoder(cfg: dict) -> FSMN:
    enc_conf = cfg["encoder_conf"]
    return FSMN(
        input_dim=enc_conf["input_dim"],
        input_affine_dim=enc_conf["input_affine_dim"],
        fsmn_layers=enc_conf["fsmn_layers"],
        linear_dim=enc_conf["linear_dim"],
        proj_dim=enc_conf["proj_dim"],
        lorder=enc_conf["lorder"],
        rorder=enc_conf["rorder"],
        lstride=enc_conf["lstride"],
        rstride=enc_conf["rstride"],
        output_affine_dim=enc_conf["output_affine_dim"],
        output_dim=enc_conf["output_dim"],
        use_softmax=True,
    )


def load_encoder_weights(encoder: FSMN, model_pt: Path) -> None:
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    # FsmnVADStreaming stores the FSMN under `encoder.`; strip that prefix.
    stripped = {
        (k[len("encoder.") :] if k.startswith("encoder.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = encoder.load_state_dict(stripped, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys in state_dict: {unexpected}")
    if missing:
        print(f"[warn] missing keys (left at init): {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("./hf_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("./model"))
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--tract-version", type=str, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading FSMN-VAD assets from", HF_REPO)
    paths = download_assets(args.cache)
    with paths["config.yaml"].open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    means, rescales = parse_am_mvn(paths["am.mvn"])

    encoder = build_encoder(cfg)
    load_encoder_weights(encoder, paths["model.pt"])
    encoder.eval()

    preprocessor = FsmnVadPreprocessor(
        cmvn_means=means,
        cmvn_rescales=rescales,
    ).eval()

    n_samples = int(args.sample_seconds * preprocessor.sample_rate)
    audio_sample = torch.randn(1, n_samples, dtype=torch.float32) * 0.1
    with torch.no_grad():
        feats = preprocessor(audio_sample)
    print(f"Preprocessor output: {tuple(feats.shape)} (B, T', lfr_m * n_mels)")

    encoder_wrapper = FsmnVadEncoder(encoder=encoder).eval()
    with torch.no_grad():
        probs = encoder_wrapper(feats)
    print(f"Encoder output: {tuple(probs.shape)} (B, T', fsmn_output_dim)")

    tract_version = args.tract_version or TractNNEF.latest_version()
    preproc_path = args.out_dir / "preprocessor.nnef.tgz"
    enc_path = args.out_dir / "encoder.nnef.tgz"

    print(f"Exporting preprocessor (fixed shape) to {preproc_path}")
    export_model_to_nnef(
        model=preprocessor,
        args=audio_sample,
        file_path_export=preproc_path,
        inference_target=TractNNEF(version=tract_version, check_io=True),
        input_names=["audio"],
        output_names=["features"],
        debug_bundle_path=Path("./debug_preprocessor.tgz"),
    )

    print(f"Exporting encoder with dynamic time axis to {enc_path}")
    export_model_to_nnef(
        model=encoder_wrapper,
        args=feats,
        file_path_export=enc_path,
        inference_target=TractNNEF(
            version=tract_version,
            check_io=True,
            dynamic_axes={"features": {1: "ENCODER__TIME"}},
        ),
        input_names=["features"],
        output_names=["probs"],
        debug_bundle_path=Path("./debug_encoder.tgz"),
        custom_extensions=[
            # The FSMN stack has lorder=20 left context per layer; pulsed tract
            # will need at least that many frames before producing valid output.
            "tract_assert ENCODER__TIME >= 1",
        ],
    )
    print("Done.")


if __name__ == "__main__":
    main()
