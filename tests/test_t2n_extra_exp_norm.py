"""Round-trip tests for `t2n_extra::exp_{unit,mean}_norm`.

Both ops lower to `tract_extra_exp_*_norm` in NNEF. Tract's pulsifier
is wired (`OpPulsifier::register::<ExpUnitNorm>`), so this also
exercises the streaming path under `change_dynamic_axes`.
"""

from __future__ import annotations

import pytest
import torch

if not hasattr(torch.library, "custom_op"):
    pytest.skip(
        "t2n_extra::exp_*_norm tests need torch.library.custom_op "
        "(torch >= 2.4)",
        allow_module_level=True,
    )

import tarfile
from functools import partial
from pathlib import Path

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    change_dynamic_axes,
    check_model_io_test,
)
from torch_to_nnef.inference_target import TractNNEF


@torch.library.custom_op(
    "t2n_extra::exp_unit_norm",
    mutates_args=(),
    schema=(
        "(Tensor input, Tensor state_init, int axis, float alpha, "
        "float epsilon, bool complex) -> Tensor"
    ),
)
def _exp_unit_norm(
    input: torch.Tensor,
    state_init: torch.Tensor,
    axis: int,
    alpha: float,
    epsilon: float,
    complex: bool,  # noqa: A002 -- matches tract attr name
) -> torch.Tensor:
    """Reference EMA-unit-norm matching tract's `ExpUnitNorm` semantics."""
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
def _exp_mean_norm(
    input: torch.Tensor,
    state_init: torch.Tensor,
    axis: int,
    alpha: float,
    scaling_factor: float,
) -> torch.Tensor:
    """Reference EMA-mean-norm matching tract's `ExpUnitNorm{mean=true}`."""
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
def _exp_mean_norm_meta(input, state_init, axis, alpha, scaling_factor):
    return input.new_empty(input.shape)


# Make sure registering the bundled handlers picks up exp_*_norm.
import torch_to_nnef.op.extras  # noqa: E402, F401


def _skip_if_not_tract(inf):
    return isinstance(inf, TractNNEF)


# --- Real-input unit norm (mirrors SpecNorm without the complex flag) -------


class _UnitNormReal(torch.nn.Module):
    """`x.shape = [B, T, F]`, axis=T=1."""

    def __init__(self, alpha: float = 0.99, eps: float = 1e-12):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _t, f = x.shape
        state = torch.zeros(b, f, dtype=x.dtype, device=x.device)
        return torch.ops.t2n_extra.exp_unit_norm(
            x, state, 1, self.alpha, self.eps, False
        )


# --- Complex-input unit norm (full SpecNorm shape) --------------------------


class _UnitNormComplex(torch.nn.Module):
    """`x.shape = [B, T, F, 2]`, axis=T=1, complex=True."""

    def __init__(self, alpha: float = 0.99, eps: float = 1e-12):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _t, f, _ = x.shape
        state = torch.zeros(b, f, dtype=x.dtype, device=x.device)
        return torch.ops.t2n_extra.exp_unit_norm(
            x, state, 1, self.alpha, self.eps, True
        )


# --- Mean norm (mirrors ErbNorm: centre then divide by 40) ------------------


class _MeanNorm(torch.nn.Module):
    """`x.shape = [B, T, F]`, axis=T=1, scaling_factor=40.0."""

    def __init__(self, alpha: float = 0.99, scaling_factor: float = 40.0):
        super().__init__()
        self.alpha = alpha
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _t, f = x.shape
        state = torch.zeros(b, f, dtype=x.dtype, device=x.device)
        return torch.ops.t2n_extra.exp_mean_norm(
            x, state, 1, self.alpha, self.scaling_factor
        )


def _input_real(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 16, 8)


def _input_complex(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 16, 8, 2)


_test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)
# Static-shape cases.
_test_suite.add(
    _input_real(), _UnitNormReal(), inference_conditions=_skip_if_not_tract
)
_test_suite.add(
    _input_complex(),
    _UnitNormComplex(),
    inference_conditions=_skip_if_not_tract,
)
_test_suite.add(
    _input_real(), _MeanNorm(), inference_conditions=_skip_if_not_tract
)
# Streaming-axis variant (axis=T=1 marked STREAM). Pulsifier overrides
# `skip` to the runtime delay; the export bakes `skip=0` and tract
# rewrites at PulsedModel-build time.
_test_suite.add(
    _input_real(),
    _UnitNormReal(),
    inference_conditions=_skip_if_not_tract,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes={"input_0": {1: "S"}}
    ),
)
_test_suite.add(
    _input_complex(),
    _UnitNormComplex(),
    inference_conditions=_skip_if_not_tract,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes={"input_0": {1: "S"}}
    ),
)
_test_suite.add(
    _input_real(),
    _MeanNorm(),
    inference_conditions=_skip_if_not_tract,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes={"input_0": {1: "S"}}
    ),
)


@pytest.mark.parametrize(
    "_id,test_input,model,inference_target",
    _test_suite.test_samples,
    ids=_test_suite.ids,
)
def test_exp_norm_export(_id, test_input, model, inference_target):
    def _assert_op_in_graph(it, export_path: Path):
        if not isinstance(it, TractNNEF):
            return
        expected = (
            "tract_extra_exp_mean_norm"
            if isinstance(model, _MeanNorm)
            else "tract_extra_exp_unit_norm"
        )
        with tarfile.open(export_path) as tf:
            graph = tf.extractfile("graph.nnef").read().decode("utf8")
            assert expected in graph

    check_model_io_test(
        model=model,
        test_input=test_input,
        inference_target=inference_target,
        callback_post_export=_assert_op_in_graph,
    )
