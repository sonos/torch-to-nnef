"""Constant-fold lowerings for tract, and their dynamic-axes safety.

Two ops must be folded because tract cannot represent/run them otherwise; these
exercise the fold paths the proptest harness (runtime inputs) cannot reach:

- boolean-mask advanced indexing ``x[mask]`` with a constant mask: a
  masked-select (filter), which has no NNEF op (``tract_core_gather`` would
  treat the bool mask as integer positions and give the wrong length), so a
  constant mask folds to the selected constant,
- ``argsort`` of a constant folds to a constant permutation: emitting
  ``tract_core_topk`` instead fails on the Qwen2.5-VL window_index, whose sort
  axis is a symbolic TDim even in a static export (``TDim is not a number``).

Regression for the Qwen2.5-VL vision tower window_index
(``index_padded[index_padded != -100]`` then ``argsort(window_index)``).

The folds bake trace-time values, so they must NOT fire when the value depends
on a dynamic axis (over-baking). Under ``dynamic_axes`` only genuine constants
(``.data``) fold; the dynamic-axes tests below export once with a symbolic axis
and run tract at several sizes to catch a frozen shape/index. (``reshape``/
``view`` with ``-1`` needs no fold -- tract resolves it -- so it is only checked
for dynamic correctness here.)
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCheckTolerance, build_io


def _export_and_check(module, x):
    target = TractNNEF.latest()
    target.check_io = True
    with tempfile.TemporaryDirectory() as d:
        export_model_to_nnef(
            model=module.eval(),
            args=(x,),
            inference_target=target,
            file_path_export=Path(d) / "m.nnef.tgz",
            input_names=["x"],
            output_names=["y"],
        )


def _export_dynamic(module, trace_x, dyn_axes, dst):
    target = TractNNEF.latest()
    target.check_io = False
    target.dynamic_axes = dyn_axes
    export_model_to_nnef(
        model=module.eval(),
        args=(trace_x,),
        inference_target=target,
        file_path_export=dst,
        input_names=["x"],
        output_names=["y"],
    )
    return target


def _run_dynamic_at_sizes(module, trace_x, dyn_axes, make_x, sizes):
    """Export once with a symbolic axis, then run tract at several sizes.

    Surfaces over-baking: if a fold froze a shape/index to the trace-time
    value, tract errors or mismatches at any size != the trace size.
    """
    module = module.eval()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        nnef = d / "m.nnef.tgz"
        target = _export_dynamic(module, trace_x, dyn_axes, nnef)
        cli = target.tract_cli
        for s in sizes:
            xi = make_x(s)
            inb, outb = d / f"in_{s}.npz", d / f"out_{s}.npz"
            build_io(
                module,
                (xi,),
                input_bundle_path=inb,
                output_bundle_path=outb,
                input_names=["x"],
                output_names=["y"],
            )
            cmd = cli.assert_io_cmd_str(
                nnef, inb, outb, check_tolerance=TractCheckTolerance.APPROXIMATE
            )
            proc = subprocess.run(cmd, capture_output=True, text=True)
            assert proc.returncode == 0, (
                f"tract failed/mismatched at size {s} (trace was "
                f"{tuple(trace_x.shape)}):\n{proc.stdout[-800:]}{proc.stderr[-800:]}"
            )


class _BoolMaskAndArgsort(nn.Module):
    def __init__(self):
        super().__init__()
        # constant "padded index" with sentinels, like Qwen window_index
        self.register_buffer(
            "idx", torch.tensor([3, -100, 1, -100, 2, 0], dtype=torch.long)
        )

    def forward(self, x):
        kept = self.idx[self.idx != -100]  # bool-mask select -> [3,1,2,0]
        order = torch.argsort(kept)  # argsort of a constant
        gathered = kept[order]  # reorder via folded permutation
        return x + gathered.sum().float()


class _StaticViewMinusOne(nn.Module):
    def forward(self, x):
        # x is [4, 6]; view(-1, 3) must bake the -1 to 8 on a static input
        return x.reshape(2, -1).reshape(-1, 3).sum()


def test_bool_mask_and_argsort_fold_export():
    _export_and_check(_BoolMaskAndArgsort(), torch.zeros(2, 3))


def test_static_view_minus_one_export():
    _export_and_check(_StaticViewMinusOne(), torch.arange(24.0).reshape(4, 6))


def test_non_constant_bool_mask_is_rejected():
    from torch_to_nnef.exceptions import T2NError

    class _DynMask(nn.Module):
        def forward(self, x):
            return x[x > 0]

    with pytest.raises(T2NError):
        _export_and_check(_DynMask(), torch.randn(8))


# ---------------------------------------------------------------------------
# Dynamic-axes ("multiple time dimension") coverage.
#
# The static folds above bake trace-time values. Under `dynamic_axes` those
# same folds must NOT freeze a shape/index that depends on the symbolic axis
# (over-baking): the single fixed-shape export cannot surface that, so we
# export once with a symbolic axis and drive tract at several sizes.
# ---------------------------------------------------------------------------


class _DynViewMinusOne(nn.Module):
    def forward(self, x):  # x: [1, S, 4]; -1 == S*4, depends on the dyn axis
        return x.reshape(x.shape[0], -1) * 1.0


class _DynArgsortStaticAxis(nn.Module):
    """argsort a whole-number tensor along a STATIC axis, dynamic time axis.

    ``base`` is an integer tensor (so it carries ``_traced_data``) broadcast
    over the dynamic axis. The sort is along the static last axis, so it lowers
    fine at runtime -- but the argsort fold would fold it to a constant whose
    shape freezes the dynamic axis to the trace-time length (over-baking),
    mis-sorting every other size. This is a pattern models actually use
    (ranking along a feature axis within a sequence).
    """

    def forward(self, x):  # x: [1, S, 4]
        base = torch.arange(3, -1, -1).view(1, 1, 4).expand(1, x.shape[1], 4)
        order = torch.argsort(base, dim=2)
        return torch.gather(x, 2, order) * 1.0


def test_dynamic_view_minus_one_runs_at_multiple_sizes():
    # `-1` must stay a runtime shape ref, not baked to the trace-time S*4.
    _run_dynamic_at_sizes(
        _DynViewMinusOne(),
        torch.randn(1, 3, 4),
        {"x": {1: "S"}},
        lambda s: torch.randn(1, s, 4),
        sizes=[3, 5, 7],
    )


def test_dynamic_argsort_static_axis_runs_at_multiple_sizes():
    # The argsort fold must NOT bake this whole-number sort to a constant whose
    # shape freezes the dynamic axis: it over-bakes and mis-sorts sizes other
    # than the trace (S=5, S=7 below diverge from PyTorch without the guard).
    _run_dynamic_at_sizes(
        _DynArgsortStaticAxis(),
        torch.randn(1, 3, 4),
        {"x": {1: "S"}},
        lambda s: torch.randn(1, s, 4),
        sizes=[3, 5, 7],
    )
