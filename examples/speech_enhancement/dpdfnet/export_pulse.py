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

* Drops the ``df_op`` (deep filter) head for now. ``df_op`` itself
  uses ``Tensor.unfold`` over the time axis, which is now supported
  under streaming (the ``aten::unfold`` handler emits the ``unfold``
  fragment, lowering to a Conv1d-with-identity-kernel that tract
  pulsifies natively). Wiring ``df_dec`` -> ``df_op`` back into the
  mask-only forward is left as a follow-up.
* Tract pulse can't pulse a ``reflect`` pad, so ``center=False``
  STFT/iSTFT replacements are installed in place of the upstream
  ``center=True`` ones.
* Streams the *baseline* variant (``dprnn_num_blocks=0``). The
  DPRNN variants hit an unrelated tract pulse Scan-body warmup
  limitation: tracked in :doc:`README.md` "Known limits".

``ErbNorm`` / ``SpecNorm`` keep their per-frame EMA: their forwards
are routed through ``t2n_extra::exp_{mean,unit}_norm`` (lowered to
``tract_extra_exp_*_norm``), and tract's ``OpPulsifier`` carries the
state across pulses.
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
    """Add the bootstrapped DPDFNet clone to `sys.path` (idempotent).

    The clone is third-party; we don't import it at module import time
    so that automated tooling (e.g., test discovery, linters) that
    *imports* this module without invoking `main()` doesn't fail with a
    SystemExit on a missing bootstrap. Repeated calls are no-ops -- the
    function is invoked from both `build()` and the lazy imports inside
    `IstftCenterFalse.forward`.
    """
    if not CLONE.exists():
        raise SystemExit(f"missing {CLONE}; run ./bootstrap.sh first")
    for entry in (str(CLONE), str(CLONE / "model")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


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


# `t2n_extra::exp_{unit,mean}_norm` custom ops -- declared at module
# import time so that `torch.ops.t2n_extra.exp_*_norm` is resolvable
# during tracing. The same ops are declared (independently) by
# `tests/test_t2n_extra_exp_norm.py`; the example and the test never
# share a Python process so the duplicate-registration check doesn't
# fire in practice.


@torch.library.custom_op(
    "t2n_extra::exp_unit_norm",
    mutates_args=(),
    schema=(
        "(Tensor input, Tensor state_init, int axis, float alpha, "
        "float epsilon, bool complex) -> Tensor"
    ),
)
# pylint: disable=redefined-builtin
def _exp_unit_norm(
    input: torch.Tensor,
    state_init: torch.Tensor,
    axis: int,
    alpha: float,
    epsilon: float,
    complex: bool,  # noqa: A002 -- matches tract attr name
) -> torch.Tensor:
    state = state_init.clone()
    out = input.clone()
    n = input.shape[axis]
    eps_t = torch.full_like(state, epsilon)
    for i in range(n):
        idx = [slice(None)] * input.ndim
        idx[axis] = i
        t_slice = out[tuple(idx)]
        if complex:
            mag = (t_slice * t_slice).sum(dim=-1).sqrt()
        else:
            mag = t_slice.abs()
        state = torch.maximum(mag, eps_t) * (1.0 - alpha) + state * alpha
        denom = state.sqrt()
        if complex:
            denom = denom.unsqueeze(-1)
        out[tuple(idx)] = t_slice / denom
    return out


@_exp_unit_norm.register_fake
# pylint: disable=redefined-builtin
def _exp_unit_norm_meta(input, state_init, axis, alpha, epsilon, complex):  # noqa: A002
    return input.new_empty(input.shape)


@torch.library.custom_op(
    "t2n_extra::exp_mean_norm",
    mutates_args=(),
    schema=(
        "(Tensor input, Tensor state_init, int axis, float alpha, "
        "float scaling_factor) -> Tensor"
    ),
)
# pylint: disable=redefined-builtin
def _exp_mean_norm(
    input: torch.Tensor,
    state_init: torch.Tensor,
    axis: int,
    alpha: float,
    scaling_factor: float,
) -> torch.Tensor:
    state = state_init.clone()
    out = input.clone()
    n = input.shape[axis]
    for i in range(n):
        idx = [slice(None)] * input.ndim
        idx[axis] = i
        t_slice = out[tuple(idx)]
        state = t_slice * (1.0 - alpha) + state * alpha
        out[tuple(idx)] = (t_slice - state) / scaling_factor
    return out


@_exp_mean_norm.register_fake
# pylint: disable=redefined-builtin
def _exp_mean_norm_meta(input, state_init, axis, alpha, scaling_factor):
    return input.new_empty(input.shape)


class ErbNormEMA(torch.nn.Module):
    """Streaming-friendly replacement for upstream ``ErbNorm``.

    Routes the per-frame EMA centring through ``t2n_extra::exp_mean_norm``
    so tract pulse can carry the state across pulses. Matches upstream
    semantics: ``x_norm[t] = (x[t] - mu[t]) / sqrt(var + eps)`` with
    ``var = 40^2`` (DFN3 default) and a fixed-init mu linearly
    interpolated across the feature axis from ``init_vals``.
    """

    def __init__(self, init_vals, eps: float, num_feat: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        # scaling_factor = sqrt(var + eps) ~= 40 + eps (eps << 1)
        self.scaling_factor = float(40.0 + eps)
        step = (init_vals[1] - init_vals[0]) / (num_feat - 1)
        mu_init = torch.tensor(
            [init_vals[0] + i * step for i in range(num_feat)],
            dtype=torch.float32,
        )
        self.register_buffer("mu_init", mu_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]; axis=T=1; state shape (B, F).
        state = self.mu_init.unsqueeze(0).expand(x.shape[0], -1).contiguous()
        return torch.ops.t2n_extra.exp_mean_norm(
            x, state, 1, self.alpha, self.scaling_factor
        )


class SpecNormEMA(torch.nn.Module):
    """Streaming-friendly replacement for upstream ``SpecNorm``.

    Routes the per-frame magnitude EMA through
    ``t2n_extra::exp_unit_norm`` with ``complex=True``. Matches upstream
    semantics: ``x_norm[t] = x[t] / sqrt(s[t])`` where
    ``s[t] = alpha * s[t-1] + (1 - alpha) * |x[t]|`` (with ``|x|``
    computed over the trailing-2 (re, im) axis).
    """

    def __init__(self, init_vals, eps: float, num_feat: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.eps = float(eps)
        step = (init_vals[1] - init_vals[0]) / (num_feat - 1)
        s_init = torch.tensor(
            [init_vals[0] + i * step for i in range(num_feat)],
            dtype=torch.float32,
        )
        self.register_buffer("s_init", s_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F, 2]; axis=T=1; state shape (B, F).
        state = self.s_init.unsqueeze(0).expand(x.shape[0], -1).contiguous()
        return torch.ops.t2n_extra.exp_unit_norm(
            x, state, 1, self.alpha, self.eps, True
        )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lazy import keeps the module importable without `bootstrap.sh`
        # (the clone is only required at run time). Storing the function
        # on ``self`` would force eager resolution at construction; this
        # also avoids holding a reference that an ``nn.Module`` reflects
        # under ``_modules`` / ``_buffers`` introspection.
        _, _as_complex = _import_dpdfnet()
        x = _as_complex(x)
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
    model.erb_norm = ErbNormEMA(
        init_vals=model.erb_norm.init_vals,
        eps=model.erb_norm.eps,
        num_feat=model.erb_bins,
        alpha=model.erb_norm.alpha,
    )
    model.spec_norm = SpecNormEMA(
        init_vals=model.spec_norm.init_vals,
        eps=model.spec_norm.eps,
        num_feat=model.nb_df,
        alpha=model.spec_norm.alpha,
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
