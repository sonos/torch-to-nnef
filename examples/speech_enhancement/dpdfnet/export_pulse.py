"""Export a streaming-friendly DPDFNet variant for tract pulse mode.

Companion to ``export.py``. Where ``export.py`` produces a *per-frame*
NNEF with explicit state I/O, this script wraps the upstream
``model.dpdfnet.DPDFNet`` so the whole streaming pipeline (rolling
STFT, NN, iSTFT, overlap-add, GRU state) is folded into a single
streaming-axis NNEF artifact, and downstream tract handles
buffering / state in pulse mode.

Input / output:

    audio[STREAM]    float32  ->  enhanced[STREAM]   float32

where ``STREAM`` is a runtime-bound symbolic dim. The Rust wrapper
``wav-cleaner-pulse`` feeds audio chunks of a chosen pulse size
(must be a multiple of the STFT hop) and tract returns one
chunk-of-enhanced-audio per call.

Limitations:

* Drops the ``df_op`` (deep filter) head, which uses
  ``Tensor.unfold`` and traces into a per-trace-T-step stack-of-
  slices that clashes with symbolic-T streaming.
* Replaces ``ErbNorm`` / ``SpecNorm`` (per-frame EMA) with stateless
  approximations driven by the upstream init values; quality on
  short windows is close, the EMA tracking is gone. Pulse-mode
  EMAs need a tract-side state primitive that isn't wired yet.
* Tract pulse can't pulse a ``reflect`` pad, so ``center=False``
  STFT/iSTFT replacements are installed in place of the upstream
  ``center=True`` ones.
* Streams the *baseline* variant (``dprnn_num_blocks=0``). The
  DPRNN variants hit an unrelated tract pulse Scan-body warmup
  limitation: tracked in :doc:`README.md` "Known limits".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from torch_to_nnef import TractNNEF, export_model_to_nnef

HERE = Path(__file__).resolve().parent
CLONE = HERE / "_dpdfnet_clone"


def _setup_clone_path() -> None:
    """Add the bootstrapped DPDFNet clone to `sys.path`.

    The clone is third-party; we don't import it at module import time
    so that automated tooling (e.g., test discovery, linters) that
    *imports* this module without invoking `main()` doesn't fail with a
    SystemExit on a missing bootstrap.
    """
    if not CLONE.exists():
        raise SystemExit(f"missing {CLONE}; run ./bootstrap.sh first")
    sys.path.insert(0, str(CLONE))
    sys.path.insert(0, str(CLONE / "model"))


def _import_dpdfnet():
    """Lazy-import the upstream DPDFNet symbols after `_setup_clone_path`."""
    _setup_clone_path()
    # pylint: disable=import-outside-toplevel
    from model.dpdfnet import DPDFNet  # noqa: PLC0415
    from model.utils import as_complex  # noqa: PLC0415

    return DPDFNet, as_complex


COMMON_KWARGS = {
    "conv_kernel_inp": (3, 3),
    "conv_ch": 64,
    "enc_gru_dim": 256,
    "erb_dec_gru_dim": 256,
    "df_dec_gru_dim": 256,
    "enc_lin_groups": 32,
    "lin_groups": 16,
    "upsample_conv_type": "subpixel",
    "group_linear_type": "loop",
    "point_wise_type": "cnn",
    "separable_first_conv": True,
}


class ErbNormStateless(torch.nn.Module):
    """Stateless replacement for ``ErbNorm`` (drops the per-frame EMA).

    Uses the upstream module's init values for a fixed mean/std; quality
    on short windows is close to the tracked EMA but the per-utterance
    adaptation is gone.
    """

    def __init__(self, init_vals, eps: float, num_feat: int) -> None:
        super().__init__()
        step = (init_vals[1] - init_vals[0]) / (num_feat - 1)
        mu = torch.tensor(
            [init_vals[0] + i * step for i in range(num_feat)],
            dtype=torch.float32,
        )
        var = torch.full_like(mu, 40.0**2)
        self.register_buffer("mu", mu)
        self.register_buffer("inv_std", 1.0 / (var.sqrt() + eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) * self.inv_std


class SpecNormStateless(torch.nn.Module):
    """Stateless replacement for ``SpecNorm``."""

    def __init__(self, init_vals, eps: float, num_feat: int) -> None:
        super().__init__()
        step = (init_vals[1] - init_vals[0]) / (num_feat - 1)
        s = torch.tensor(
            [init_vals[0] + i * step for i in range(num_feat)],
            dtype=torch.float32,
        ).view(1, 1, num_feat, 1)
        self.register_buffer("inv_s_sqrt", 1.0 / (s + eps).sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.inv_s_sqrt


class StftCenterFalse(torch.nn.Module):
    """Drop-in for the model's ``Stft`` using ``center=False``.

    Tract pulse rejects ``reflect`` padding which ``center=True``
    requires; switching to ``center=False`` keeps the same hop / window
    semantics with no padding.
    """

    def __init__(self, src) -> None:
        super().__init__()
        self.n_fft = src.n_fft
        self.win_len = src.win_len
        self.hop = src.hop
        self.normalized = src.normalized
        self.register_buffer("w", src.w.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop,
            window=self.w,
            normalized=self.normalized,
            return_complex=True,
            center=False,
        )


class IstftCenterFalse(torch.nn.Module):
    """Drop-in for the model's ``Istft`` using ``center=False``."""

    def __init__(self, src) -> None:
        super().__init__()
        self.n_fft = src.n_fft_inv
        self.win_len = src.win_len_inv
        self.hop = src.hop_inv
        self.normalized = src.normalized
        self.register_buffer("w", src.w_inv.clone())
        # Lazy-imported here so `IstftCenterFalse` can be referenced
        # without `bootstrap.sh` having run.
        _, self._as_complex = _import_dpdfnet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_complex(x)
        return torch.istft(
            x,
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop,
            window=self.w,
            normalized=self.normalized,
            center=False,
        )


class DPDFNetMaskOnly(torch.nn.Module):
    """audio -> apply_stft -> NN mask -> mask * spec -> apply_istft.

    Drops the ``df_op`` head (see module docstring).
    """

    def __init__(self, inner: "DPDFNet") -> None:  # noqa: F821
        super().__init__()
        self.inner = inner

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spec, feat_erb, feat_spec = self.inner._feature_extraction(audio)
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        feat_erb = self.inner.pad_feat(feat_erb)
        feat_spec = self.inner.pad_feat(feat_spec)
        e0, e1, e2, e3, emb, _, _ = self.inner.enc(feat_erb, feat_spec)
        m = self.inner.erb_dec(emb, e3, e2, e1, e0)
        spec_e = self.inner.mask(spec, m)
        spec_e = torch.view_as_complex(spec_e).squeeze(1)
        return self.inner.apply_istft(spec_e)


def build(checkpoint: Path) -> torch.nn.Module:
    dpdfnet_cls, _ = _import_dpdfnet()
    model = dpdfnet_cls(dprnn_num_blocks=0, **COMMON_KWARGS).eval()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {
        k: v for k, v in state.items() if not k.endswith("num_batches_tracked")
    }
    model.load_state_dict(state, strict=False)
    model.erb_norm = ErbNormStateless(
        init_vals=model.erb_norm.init_vals,
        eps=model.erb_norm.eps,
        num_feat=model.erb_bins,
    )
    model.spec_norm = SpecNormStateless(
        init_vals=model.spec_norm.init_vals,
        eps=model.spec_norm.eps,
        num_feat=model.nb_df,
    )
    model.stft = StftCenterFalse(model.stft)
    model.istft = IstftCenterFalse(model.istft)
    return DPDFNetMaskOnly(model).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=HERE / "_checkpoints" / "baseline.pth",
        help=(
            "Baseline checkpoint (`dprnn_num_blocks=0`). Run "
            "`./bootstrap.sh baseline` to fetch it."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "baseline_pulse.nnef.tgz",
        help="Output NNEF artifact path.",
    )
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(
            f"missing {args.checkpoint}; run `./bootstrap.sh baseline`"
        )

    wrapped = build(args.checkpoint)

    torch.manual_seed(0)
    audio = torch.randn(1, 16_000, dtype=torch.float32) * 0.1
    with torch.no_grad():
        out = wrapped(audio)
    print(
        f"PyTorch sanity: in {tuple(audio.shape)}  out {tuple(out.shape)}  "
        f"mean_abs={out.abs().mean().item():.3e}"
    )

    target = TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=False,
        dynamic_axes={"audio": {1: "STREAM"}},
    )
    export_model_to_nnef(
        model=wrapped,
        args=audio,
        file_path_export=args.out,
        inference_target=target,
        input_names=["audio"],
        output_names=["enhanced"],
    )
    print(f"exported -> {args.out}")


if __name__ == "__main__":
    main()
