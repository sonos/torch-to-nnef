"""Regression for `ConvTranspose{1,2,3}d(output_padding=...)` export.

PyTorch's transposed-conv output formula is

    out = (in - 1) * stride - 2*pad + dilation*(kernel - 1) + output_padding + 1

NNEF's `deconv` does not expose `output_padding`. The naive mapping that
drops it produces an output `output_padding` smaller than PyTorch's
reference, and the mismatch propagates into any downstream shape-based
op (typically surfaced by a `reshape` whose static target no longer
matches the inferred upstream size).

The emitter encodes `output_padding` as asymmetric NNEF padding on the
"after" side of each spatial axis: `pytorch(pad, output_padding=op)`
becomes NNEF `padding=(pad, pad - op)`. These tests check the shape and
the numeric round-trip for the cases that fire in real models (notably
DFN3's `convt1`/`convt2` use `stride=2, kernel=3, padding=1,
output_padding=1`).
"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class ConvT2dOutputPadding1(nn.Module):
    """`stride=2` deconv with `output_padding=1`: doubles the spatial dim.

    Matches DFN3's `convt1` / `convt2` shape: kernel=3, padding=1,
    output_padding=1, stride=2. PyTorch yields `out_W = 2 * in_W`. The
    naive NNEF emit (without our asymmetric-padding correction) would
    give `out_W = 2 * in_W - 1`.
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, 1),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_t(x)


class ConvT2dDoubledThenReshape(nn.Module):
    """Same deconv pattern, followed by a static reshape that pins the size.

    A pure deconv test won't fail when the off-by-one stays purely
    inside dynamic-shape land; the downstream static reshape is what
    surfaces the bug end-to-end (matches DFN3's actual failure shape).
    """

    def __init__(self, channels: int = 4, in_w: int = 8) -> None:
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, 1),
            bias=False,
        )
        # PyTorch output_W = 2 * in_w = 16. Reshape that to a static
        # flat-vector target; if the emit dropped output_padding the
        # upstream produces 15 elements per row and tract rejects.
        self.target_w = 2 * in_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_t(x)
        return y.reshape(-1, self.target_w)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_conv_transpose2d_output_padding_basic(inference_target):
    """Bare deconv with output_padding=1: shape + numerics on tract."""
    module = ConvT2dOutputPadding1(channels=4).eval()
    x = torch.randn(1, 4, 1, 8)
    expected_w = 2 * 8
    with torch.no_grad():
        y = module(x)
    assert y.shape[-1] == expected_w, (y.shape, expected_w)
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=deepcopy(inference_target),
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_conv_transpose2d_output_padding_then_reshape(inference_target):
    """Deconv + downstream static reshape: end-to-end DFN3-style chain."""
    in_w = 8
    module = ConvT2dDoubledThenReshape(channels=4, in_w=in_w).eval()
    x = torch.randn(1, 4, 1, in_w)
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=deepcopy(inference_target),
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_conv_transpose1d_output_padding(inference_target):
    """1D variant for completeness."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_t = nn.ConvTranspose1d(
                4, 4, kernel_size=3, stride=2, padding=1, output_padding=1
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv_t(x)

    module = M().eval()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        y = module(x)
    assert y.shape[-1] == 16, y.shape
    check_model_io_test(
        model=module,
        test_input=(x,),
        input_names=["input"],
        output_names=["output"],
        inference_target=deepcopy(inference_target),
    )
