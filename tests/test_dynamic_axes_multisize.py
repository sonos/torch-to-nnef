"""Multi-size dynamic-axis soundness tests.

Each model is exported ONCE with a symbolic axis (traced at one size), then the
SAME graph is run through tract at several concrete sizes. This is what catches
a wrongly-baked dynamic axis: a mis-classified axis is correct at the traced
size but wrong at a different size, which a single-size ``check_io`` cannot
detect. These guard the per-axis dynamic-shape tracking
(``torch_to_nnef.torch_graph.dynamic_axes``).
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCheckTolerance, build_io

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX


def _assert_dynamic_correct(model, make_input, dynamic_axes, sizes):
    """Export once (symbolic) then assert tract matches torch at each size."""
    model = model.eval()
    base = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    if base.version <= "0.21.5":
        pytest.skip("dynamic slice/shape needs tract > 0.21.5")
    target = TractNNEF(
        version=base.version, check_io=False, dynamic_axes=dynamic_axes
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        first = make_input(sizes[0])
        in_names, out_names = build_io(
            model, first, tmpd / "i0.npz", tmpd / "o0.npz"
        )
        nnef = tmpd / "model.nnef.tgz"
        export_model_to_nnef(
            model=model,
            args=first,
            file_path_export=nnef,
            inference_target=target,
            input_names=in_names,
            output_names=out_names,
        )
        for k, size in enumerate(sizes):
            inp = make_input(size)
            ipath, opath = tmpd / f"i{k}.npz", tmpd / f"o{k}.npz"
            build_io(model, inp, ipath, opath, in_names, out_names)
            target.tract_cli.assert_io(
                nnef,
                ipath,
                opath,
                check_tolerance=TractCheckTolerance.APPROXIMATE,
            )


class StaticAxisSplit(nn.Module):
    """`x.shape[-1] // 2` split on a static last axis; axis 1 is dynamic."""

    def forward(self, x):  # (B, S, F): S dynamic, F static
        f = x.shape[-1]
        half = f // 2
        a, b = torch.split(x, [half, f - half], dim=-1)
        return a + b


def test_split_on_static_axis_under_dynamic():
    _assert_dynamic_correct(
        StaticAxisSplit(),
        lambda s: torch.rand(1, s, 8),
        {"input_0": {1: "S"}},
        sizes=[12, 20, 5],
    )


class ReshapeSizeDerived(nn.Module):
    """Reshape whose dims are size-derived from a dynamic axis traced as 1."""

    def forward(self, x):  # (B, T, C): B dynamic (traced B=1)
        b, t, c = x.shape
        return x.reshape(b, t * c)


def test_reshape_size_derived_batch_one():
    _assert_dynamic_correct(
        ReshapeSizeDerived(),
        lambda b: torch.rand(b, 4, 3),
        {"input_0": {0: "B"}},
        sizes=[1, 3, 2],
    )


class TransposeThenConvReshape(nn.Module):
    """transpose moves the dynamic axis; a later conv width stays symbolic.

    Mirrors issue #18: `-1` in the reshape must resolve dynamically, which
    requires the transpose rule to move the dynamic axis correctly.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 3)
        self.linear = nn.Linear(80, 10)

    def forward(self, x):  # (B, S, 10): S dynamic
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)
        b, width = x.shape[0], x.shape[3]
        x = x.transpose(1, 3).reshape([b, width, -1])
        return self.linear(x)


def test_transpose_then_conv_reshape():
    _assert_dynamic_correct(
        TransposeThenConvReshape(),
        lambda s: torch.rand(1, s, 10),
        {"input_0": {1: "S"}},
        sizes=[1000, 1500],
    )


class TwoLayerAttention(nn.Module):
    """Two self-attention layers (view->transpose->matmul) over a dynamic seq.

    Exercises the matmul/bmm + transpose + view(literal-head-dims) rules across
    layers: `head_dim` (a literal reshape dim) must stay static so per-head
    reshapes fold, while the sequence axis stays symbolic.
    """

    def __init__(self, d=16, h=2):
        super().__init__()
        self.h, self.hd = h, d // h
        self.blocks = nn.ModuleList(
            nn.ModuleDict({k: nn.Linear(d, d) for k in "qkvo"})
            for _ in range(2)
        )

    def _attn(self, x, blk):
        b, s, _ = x.shape

        def heads(t):
            return t.view(b, s, self.h, self.hd).transpose(1, 2)

        q, k, v = heads(blk["q"](x)), heads(blk["k"](x)), heads(blk["v"](x))
        scores = (q @ k.transpose(-1, -2)) / (self.hd**0.5)
        out = scores.softmax(-1) @ v
        out = out.transpose(1, 2).reshape(b, s, self.h * self.hd)
        return blk["o"](out)

    def forward(self, x):  # (B, S, d): S dynamic
        for blk in self.blocks:
            x = x + self._attn(x, blk)
        return x


def test_two_layer_attention_dynamic_seq():
    _assert_dynamic_correct(
        TwoLayerAttention(),
        lambda s: torch.rand(1, s, 16),
        {"input_0": {1: "S"}},
        sizes=[8, 16, 4],
    )
