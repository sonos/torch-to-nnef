"""Round-trip tests for the `t2n_extra::ssm_scan{,_y}` custom ops.

`ssm_scan` returns `(y, h_final)` and uses the `mamba_ssm_scan`
fragment. `ssm_scan_y` returns only `y` and uses the pulse-friendly
`mamba_ssm_scan_pulse` fragment (tract's Scan pulsifier rejects
`"last"` outputs, so we drop `h_final` for the pulse path).
"""

from __future__ import annotations

import pytest
import torch

if not hasattr(torch.library, "custom_op"):
    pytest.skip(
        "t2n_extra::ssm_scan tests need torch.library.custom_op (torch >= 2.4)",
        allow_module_level=True,
    )

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)


def _skip_if_not_tract(inf):
    from torch_to_nnef.inference_target import TractNNEF

    return isinstance(inf, TractNNEF)


@torch.library.custom_op(
    "t2n_extra::ssm_scan",
    mutates_args=(),
    schema=(
        "(Tensor discrete_A, Tensor deltaB_u, Tensor C, Tensor h_init) "
        "-> (Tensor, Tensor)"
    ),
)
def _ssm_scan(
    discrete_A: torch.Tensor,
    deltaB_u: torch.Tensor,
    C: torch.Tensor,
    h_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = h_init
    outs = []
    seq_len = discrete_A.shape[2]
    for t in range(seq_len):
        h = discrete_A[:, :, t, :] * h + deltaB_u[:, :, t, :]
        y = torch.matmul(h, C[:, t, :].unsqueeze(-1)).squeeze(-1)
        outs.append(y)
    return torch.stack(outs, dim=-1), h


@_ssm_scan.register_fake
def _ssm_scan_meta(discrete_A, deltaB_u, C, h_init):
    B, D, T, _N = discrete_A.shape
    y_shape = (B, D, T)
    h_shape = h_init.shape
    return (
        discrete_A.new_empty(y_shape),
        h_init.new_empty(h_shape),
    )


# Make sure registering the bundled handlers picks up `ssm_scan`.
import torch_to_nnef.op.extras  # noqa: E402, F401


@torch.library.custom_op(
    "t2n_extra::ssm_scan_y",
    mutates_args=(),
    schema=(
        "(Tensor discrete_A, Tensor deltaB_u, Tensor C, Tensor h_init) "
        "-> Tensor"
    ),
)
def _ssm_scan_y(
    discrete_A: torch.Tensor,
    deltaB_u: torch.Tensor,
    C: torch.Tensor,
    h_init: torch.Tensor,
) -> torch.Tensor:
    h = h_init
    outs = []
    seq_len = discrete_A.shape[2]
    for t in range(seq_len):
        h = discrete_A[:, :, t, :] * h + deltaB_u[:, :, t, :]
        y = torch.matmul(h, C[:, t, :].unsqueeze(-1)).squeeze(-1)
        outs.append(y)
    return torch.stack(outs, dim=-1)


@_ssm_scan_y.register_fake
def _ssm_scan_y_meta(discrete_A, deltaB_u, C, h_init):
    B, D, T, _N = discrete_A.shape
    return discrete_A.new_empty((B, D, T))


class _ScanFull(torch.nn.Module):
    def forward(self, A, Bu, C, h0):
        y, h = torch.ops.t2n_extra.ssm_scan(A, Bu, C, h0)
        return y, h


class _ScanY(torch.nn.Module):
    def forward(self, A, Bu, C, h0):
        return torch.ops.t2n_extra.ssm_scan_y(A, Bu, C, h0)


def _make_inputs(B=1, D=4, T=5, N=8, seed=0):
    torch.manual_seed(seed)
    A = torch.rand(B, D, T, N)
    Bu = torch.randn(B, D, T, N)
    C = torch.randn(B, T, N)
    h0 = torch.zeros(B, D, N)
    return (A, Bu, C, h0)


_test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)
for seed in (0, 1, 2):
    for module in (_ScanFull(), _ScanY()):
        _test_suite.add(
            _make_inputs(seed=seed),
            module,
            inference_conditions=_skip_if_not_tract,
        )


@pytest.mark.parametrize(
    "_id,test_input,model,inference_target",
    _test_suite.test_samples,
    ids=_test_suite.ids,
)
def test_ssm_scan_export(_id, test_input, model, inference_target):
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )
