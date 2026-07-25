"""Round-trip test for the `t2n_extra::gated_delta_scan` custom op.

Qwen3.5's gated-delta-net (GDN) linear-attention recurrence, exported as a
`tract_core_scan` via the `gated_delta_scan` fragment. Returns `(y, s_final)`;
the q*scale / l2norm / GQA-repeat prep stays outside the op as plain ops.
"""

from __future__ import annotations

import pytest
import torch

if not hasattr(torch.library, "custom_op"):
    pytest.skip(
        "t2n_extra::gated_delta_scan needs torch.library.custom_op (>= 2.4)",
        allow_module_level=True,
    )

import tarfile
from pathlib import Path

# register the bundled t2n_extra handlers (picks up `gated_delta_scan`).
import torch_to_nnef.op.extras  # noqa: E402, F401
from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from torch_to_nnef.inference_target import TractNNEF


@torch.library.custom_op(
    "t2n_extra::gated_delta_scan",
    mutates_args=(),
    schema=(
        "(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor s0) "
        "-> (Tensor, Tensor)"
    ),
)
def _gated_delta_scan(q, k, v, g, beta, s0):
    """Pure-torch reference: the gated-delta recurrence over T (axis 2)."""
    state = s0
    ys = []
    for t in range(q.shape[2]):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        g_t = g[:, :, t].exp()[..., None, None]
        beta_t = beta[:, :, t][..., None]
        state = state * g_t
        kv = (state * k_t.unsqueeze(-1)).sum(-2)
        delta = (v_t - kv) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        ys.append((state * q_t.unsqueeze(-1)).sum(-2))
    return torch.stack(ys, dim=2), state


@_gated_delta_scan.register_fake
def _meta(q, k, v, g, beta, s0):
    b, h, t, _ = q.shape
    return q.new_empty((b, h, t, v.shape[-1])), s0.new_empty(s0.shape)


class _ScanMod(torch.nn.Module):
    def forward(self, q, k, v, g, beta, s0):
        return torch.ops.t2n_extra.gated_delta_scan(q, k, v, g, beta, s0)


def _make_inputs(B=1, H=2, T=5, hk=4, hv=4, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, H, T, hk),
        torch.randn(B, H, T, hk),
        torch.randn(B, H, T, hv),
        torch.randn(B, H, T) * 0.1,
        torch.rand(B, H, T),
        torch.zeros(B, H, hk, hv),
    )


def _skip_if_not_tract(inf):
    return isinstance(inf, TractNNEF)


_test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)
for seed in (0, 1, 2):
    _test_suite.add(
        _make_inputs(seed=seed),
        _ScanMod(),
        inference_conditions=_skip_if_not_tract,
    )


@pytest.mark.parametrize(
    "_id,test_input,model,inference_target",
    _test_suite.test_samples,
    ids=_test_suite.ids,
)
def test_gated_delta_scan_export(_id, test_input, model, inference_target):
    def _assert_fragment(it, export_path: Path):
        if not isinstance(it, TractNNEF):
            return
        with tarfile.open(export_path) as tf:
            graph = tf.extractfile("graph.nnef").read().decode("utf8")
            assert "gated_delta_scan" in graph

    check_model_io_test(
        model=model,
        test_input=test_input,
        inference_target=inference_target,
        callback_post_export=_assert_fragment,
    )
