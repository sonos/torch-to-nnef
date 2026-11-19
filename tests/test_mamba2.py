"""Mamba-2 recurrence and biased causal conv export to tract ops."""

import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCli
from torch_to_nnef.op.custom_extractors import (
    CausalConvUpdateReified,
    Mamba2RecurrentReified,
)
from torch_to_nnef.op.custom_extractors.base import CUSTOMOP_KIND
from torch_to_nnef.op.custom_extractors.gdn import (
    CAUSAL_CONV_BIAS_MIN_TRACT_VERSION,
)
from torch_to_nnef.op.custom_extractors.mamba2 import (
    MAMBA2_MIN_TRACT_VERSION,
)
from torch_to_nnef.torch_graph.ir_graph import module_tracer_into_ir_graph
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


@pytest.fixture
def export_only_target(monkeypatch):
    """Exercise emission/version checks without downloading future releases."""
    monkeypatch.setattr(
        TractCli,
        "download",
        lambda version: SimpleNamespace(
            version=version, tract_path=Path("unused-export-only-tract")
        ),
    )
    return lambda version: TractNNEF(version, check_io=False)


def _graph_text(archive: Path) -> str:
    with tarfile.open(archive) as tar:
        member = tar.extractfile("graph.nnef")
        assert member is not None
        return member.read().decode()


class Mamba2Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Mamba2RecurrentReified()

    def forward(self, x, b, c, log_decay, state):
        return self.core(x, b, c, log_decay, state)


def _mamba2_inputs(seq=7, heads=4, head_dim=8, groups=2, n=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, seq, heads, head_dim)
    dt = torch.rand(1, seq, heads) * 0.5 + 0.01
    a = -torch.rand(heads) - 0.5
    b = torch.randn(1, seq, groups, n)
    c = torch.randn(1, seq, groups, n)
    state = torch.randn(1, heads, head_dim, n) * 0.1
    return x, dt, a, b, c, state


def test_mamba2_reified_matches_chunked_scan():
    nemotron_h = pytest.importorskip(
        "transformers.models.nemotron_h.modeling_nemotron_h",
        reason="transformers too old for nemotron_h",
    )
    x, dt, a, b, c, state = _mamba2_inputs()
    expected, expected_state = nemotron_h.mamba2_chunk_scan(
        x,
        dt,
        a,
        b,
        c,
        chunk_size=4,
        initial_states=state,
        return_final_states=True,
    )
    y, new_state = Mamba2RecurrentReified()(
        x * dt[..., None], b, c, dt * a, state
    )
    torch.testing.assert_close(y, expected.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(new_state, expected_state, rtol=1e-4, atol=1e-4)


def _mamba2_args():
    x, dt, a, b, c, state = _mamba2_inputs()
    return (x * dt[..., None], b, c, dt * a, state)


def test_mamba2_emission(export_only_target, tmp_path):
    path = tmp_path / "mamba2.nnef.tgz"
    export_model_to_nnef(
        Mamba2Core(),
        _mamba2_args(),
        path,
        export_only_target(MAMBA2_MIN_TRACT_VERSION),
        input_names=["x", "b", "c", "log_decay", "state"],
        output_names=["y", "new_state"],
    )
    graph = _graph_text(path)
    assert "tract_transformers_mamba2_recurrent(x, b, c, log_decay, state)" in (
        graph
    ), graph


def test_mamba2_rejects_older_tract(export_only_target, tmp_path):
    with pytest.raises(T2NErrorNotImplemented, match="requires tract >= "):
        export_model_to_nnef(
            Mamba2Core(),
            _mamba2_args(),
            tmp_path / "mamba2.nnef.tgz",
            export_only_target("0.23.10"),
        )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_mamba2_tract(inference_target):
    if inference_target.version < MAMBA2_MIN_TRACT_VERSION:
        pytest.skip(f"Mamba-2 requires tract >= {MAMBA2_MIN_TRACT_VERSION}")
    check_model_io_test(
        model=Mamba2Core(),
        test_input=_mamba2_args(),
        input_names=["x", "b", "c", "log_decay", "state"],
        output_names=["y", "new_state"],
        inference_target=inference_target,
    )


class BiasedConv(nn.Module):
    def __init__(self, channels=6, kernel=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, kernel) * 0.3)
        self.bias = nn.Parameter(torch.randn(channels) * 0.3)
        self.conv = CausalConvUpdateReified()

    def forward(self, hidden, state):
        return self.conv(hidden, state, self.weight, self.bias)


def _conv_args(seq=3, channels=6, kernel=4):
    torch.manual_seed(1)
    return (torch.randn(1, channels, seq), torch.randn(1, channels, kernel))


def test_causal_conv_bias_eager():
    model = BiasedConv()
    hidden, state = _conv_args()
    out, new_state = model(hidden, state)
    full = torch.cat([state, hidden], dim=-1)
    expected = torch.nn.functional.conv1d(
        full, model.weight[:, None], model.bias, groups=6
    )[:, :, -3:]
    torch.testing.assert_close(out, torch.nn.functional.silu(expected))
    torch.testing.assert_close(new_state, full[:, :, -4:])


def test_causal_conv_bias_emission(export_only_target, tmp_path):
    path = tmp_path / "conv.nnef.tgz"
    export_model_to_nnef(
        BiasedConv(),
        _conv_args(),
        path,
        export_only_target(CAUSAL_CONV_BIAS_MIN_TRACT_VERSION),
    )
    graph = _graph_text(path)
    assert "tract_transformers_causal_conv1d_update(" in graph, graph
    assert "bias = " in graph, graph


class KeywordConv(BiasedConv):
    def forward(self, hidden, state):
        return self.conv(
            hidden_states=hidden,
            conv_state=state,
            weight=self.weight,
            bias=self.bias,
        )


def test_causal_conv_keyword_call(export_only_target, tmp_path):
    model = KeywordConv()
    hidden, state = _conv_args()
    for got, expected in zip(
        model(hidden, state),
        BiasedConv.forward(model, hidden, state),
        strict=True,
    ):
        torch.testing.assert_close(got, expected)
    unbiased = model.conv(
        hidden_states=hidden, conv_state=state, weight=model.weight
    )
    assert len(unbiased) == 2
    path = tmp_path / "conv.nnef.tgz"
    export_model_to_nnef(
        model,
        (hidden, state),
        path,
        export_only_target(CAUSAL_CONV_BIAS_MIN_TRACT_VERSION),
    )
    assert "bias = " in _graph_text(path)


def test_causal_conv_bias_rejects_older_tract(export_only_target, tmp_path):
    with pytest.raises(T2NErrorNotImplemented, match="requires tract >= "):
        export_model_to_nnef(
            BiasedConv(),
            _conv_args(),
            tmp_path / "conv.nnef.tgz",
            export_only_target("0.23.10"),
        )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_causal_conv_bias_tract(inference_target):
    if inference_target.version < CAUSAL_CONV_BIAS_MIN_TRACT_VERSION:
        pytest.skip(
            f"conv bias requires tract >= {CAUSAL_CONV_BIAS_MIN_TRACT_VERSION}"
        )
    check_model_io_test(
        model=BiasedConv(),
        test_input=_conv_args(),
        inference_target=inference_target,
    )


def test_causal_conv_node_replays_like_the_module():
    """A custom op node replays its module like the extractor traced it.

    Its inputs come parameters first, as shape inference passes them.
    """
    model = BiasedConv()
    hidden, state = _conv_args()
    graph = module_tracer_into_ir_graph(
        TorchModuleTracer(model, args=(hidden, state))
    )
    (node,) = [op for op in graph.op_nodes if op.kind.startswith(CUSTOMOP_KIND)]
    out, new_state = node.call_op()
    assert out.shape == hidden.shape
    assert new_state.shape == state.shape
