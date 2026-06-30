"""Regression tests for aten-handler bugs surfaced by JIT-only export.

Each test scripts a tiny module that hits a specific parser code path,
exports through the full t2n pipeline with `check_io=False` (skip tract
verification, we only need IR construction to succeed), and asserts the
export call returns without raising.

The originals were all latent bugs reachable by Silero-VAD; here we
isolate each one to a minimal repro so a regression breaks the test
without depending on `silero-vad` being installed.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF


def _export(model, args, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "model.nnef.tgz"
        export_model_to_nnef(
            model=model,
            args=args,
            file_path_export=out,
            inference_target=TractNNEF(
                version=TractNNEF.latest_version(), check_io=False
            ),
            **kwargs,
        )


class _SliceWithNoneBounds(nn.Module):
    """`t[:]` lowers to `aten::slice(t, dim, None, None, 1)`."""

    def forward(self, x):
        return x[:] + 1.0


def test_slice_with_none_bounds_exports():
    m = torch.jit.script(_SliceWithNoneBounds().eval())
    _export(m, (torch.randn(2, 3),))


class _DivOfTwoIntAttrs(nn.Module):
    """`int / int` constants surface a Python-scalar `.data` in the parser."""

    def __init__(self):
        super().__init__()
        # frozen graphs bake int attrs as `prim::Constant`; the divider
        # then folds at parser time and exercises the scalar-result path.
        self.numer = 256
        self.denom = 2

    def forward(self, x):
        ratio = float(self.numer / self.denom)
        return x * ratio


def test_div_of_two_scalars_exports():
    m = torch.jit.freeze(torch.jit.script(_DivOfTwoIntAttrs().eval()))
    _export(m, (torch.randn(2, 3),))


class _SplitWithZeroSizedSection(nn.Module):
    """`torch.split(x, [2, 0, 3])` emits a zero-sized slice."""

    def forward(self, x):
        left, empty, right = torch.split(x, [2, 0, 3], dim=-1)
        return left, empty, right


def test_split_with_zero_sized_section_exports():
    m = torch.jit.script(_SplitWithZeroSizedSection().eval())
    _export(m, (torch.randn(2, 5),))


class _Conv1dWithListPadding(nn.Module):
    """`F.conv1d(..., padding=0)` with a literal int yields list padding."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3, 3))
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, stride=1, padding=0)


def test_conv1d_with_list_padding_exports():
    """The handler is registered for aten::conv1d which has int-list padding."""
    m = torch.jit.script(_Conv1dWithListPadding().eval())
    _export(m, (torch.randn(1, 3, 16),))
