"""In-place scalar arithmetic in trace-time shape computations.

einops (and similar libraries) build a `reshape` target shape by
chaining `aten::mul_` on a running scalar Long tensor:

    %result.1 : Long = aten::mul(%element.1, %1)         # init
    %result   : Long = aten::mul_(%result.1, %element.3) # in-place
    %16       : Long = aten::mul_(%result,   %element.5) # in-place
    %17       : int  = aten::Int(%16)                    # to ListConstruct

Every `aten::mul_` returns the same underlying tensor, mutated. If t2n's
IR re-executes the chain via `call_op` (for shape inference) using
`torch.ops.aten.mul_`, the second call mutates an already-mutated
tensor and produces the wrong value, then every IR node in the chain
reads back the *final* shared value. The downstream `reshape` target
becomes `[final, final]` instead of `[per_step, final]`.

The fix in `update_call_op_arg_kwargs` reroutes `aten::mul_` /
`aten::div_` through their out-of-place equivalents so re-execution
is side-effect-free.

These tests pin the einops `rearrange` pattern (which exercises this
in-place chain via its own backend code) so a regression that drops
the rerouting is caught immediately.
"""

from __future__ import annotations

import pytest

einops = pytest.importorskip(
    "einops",
    reason=(
        "einops needed to reproduce the in-place mul_ chain shape-arith "
        "pattern; install with `pip install einops`."
    ),
)
import torch

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class _DprnnLikeBlock(torch.nn.Module):
    """Minimal repro of the DPRNN intra-chunk pattern.

    The combination of a parameterised submodule (`nn.Linear`,
    `nn.LayerNorm`) and an `einops.rearrange` that builds its reshape
    target via `aten::mul_` is what surfaces the bug. The pure-einops
    pattern alone goes through the size-fold pass and never reaches the
    `call_op` re-execution path, so the bug doesn't trigger; a model
    with submodules forces t2n down the recursive IR-construction code
    path where the cached `_traced_data` is replayed and the in-place
    mutation breaks the chain.
    """

    def __init__(self, num_feat: int = 8, hidden: int = 64) -> None:
        super().__init__()
        # Submodule parameters are load-bearing: they force t2n down the
        # recursive IR construction path that `call_op`-replays each op.
        # A pure einops graph with no module gets fully size-folded by
        # the harden pass and never exercises the bug.
        self.fc = torch.nn.Linear(hidden, hidden)
        self.ln = torch.nn.LayerNorm(hidden)
        self.num_feat = num_feat
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B=1, C=hidden, T=1, F=num_feat)
        # `(b f t) c` collapses *three* axes into one -- that produces
        # the multi-step `aten::mul_` chain that exposes the bug.
        # A single `mul_` is harmless (mutate once, read once); the
        # aliasing only corrupts results when multiple chained `mul_`
        # values are cached and later read back.
        h = einops.rearrange(x, "b c t f -> (b f t) c")
        h = self.ln(self.fc(h))
        h = einops.rearrange(
            h,
            "(b f t) c -> b c t f",
            b=x.shape[0],
            t=x.shape[2],
            f=self.num_feat,
        )
        return h


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_dprnn_like_einops_mul_inplace(inference_target):
    """DPRNN-style: einops rearrange around a parameterised submodule."""
    torch.manual_seed(0)
    x = torch.randn(1, 64, 1, 8)
    check_model_io_test(_DprnnLikeBlock(), x, inference_target)
