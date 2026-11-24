"""Default opaque tracing must compose with ordinary weights and state."""

import pytest
import torch
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.compress import DEFAULT_COMPRESSION
from torch_to_nnef.tensor.opaque import NEW_OPAQUE_TRACING_STRATEGY


class MixedWeights(nn.Module):
    def __init__(self, with_state):
        super().__init__()
        self.proj = nn.Linear(32, 32, bias=False)
        self.conv = nn.Conv1d(32, 32, 3, groups=32, bias=True)
        self.with_state = with_state

    def forward(self, x, history=None):
        x = self.proj(x).transpose(1, 2)
        if self.with_state:
            x = torch.cat((history, x), dim=-1)
        return self.conv(x).transpose(1, 2)


@pytest.mark.skipif(
    not NEW_OPAQUE_TRACING_STRATEGY, reason="requires default opaque tracing"
)
@pytest.mark.parametrize("with_state", [False, True])
def test_quantized_linear_then_float_conv(tmp_path, with_state):
    torch.manual_seed(7)
    model = DEFAULT_COMPRESSION["min_max_q4_0"](MixedWeights(with_state).eval())
    args = (
        (torch.randn(1, 5, 32), torch.randn(1, 32, 2))
        if with_state
        else (torch.randn(1, 5, 32),)
    )
    with torch.no_grad():
        expected = model(*args)
        export_model_to_nnef(
            model,
            args,
            tmp_path / "model.nnef.tgz",
            inference_target=TractNNEF("0.23.10", check_io=False),
        )
        actual = model(*args)
        assert getattr(actual, "fake_mode", None) is None
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not NEW_OPAQUE_TRACING_STRATEGY, reason="requires default opaque tracing"
)
@pytest.mark.parametrize("with_state", [False, True])
def test_mixed_weight_dynamic_runtime_and_packed_storage(tmp_path, with_state):
    import os
    import subprocess
    import tarfile
    from pathlib import Path

    import numpy as np

    from torch_to_nnef.inference_target.tract import TractCli
    from torch_to_nnef.nnef_io.tensor import DatBinHeader

    binary = os.environ.get("TRACT_BIN")
    if not binary:
        pytest.skip("set TRACT_BIN for dynamic NNEF runtime checks")
    torch.manual_seed(7)
    model = DEFAULT_COMPRESSION["min_max_q4_0"](MixedWeights(with_state).eval())
    cli = TractCli(Path(binary))
    axes = {"x": {1: "S"}}
    names = ["x", "history"] if with_state else ["x"]
    cases = [
        (torch.randn(1, s, 32), torch.randn(1, 32, 2))
        if with_state
        else (torch.randn(1, s, 32),)
        for s in (5, 3, 7)
    ]
    path = tmp_path / "model.nnef.tgz"
    with torch.no_grad():
        expected = [model(*args).numpy() for args in cases]
        export_model_to_nnef(
            model,
            cases[0],
            path,
            inference_target=TractNNEF(
                cli.version,
                specific_tract_binary_path=cli.tract_path,
                dynamic_axes=axes,
                check_io=False,
            ),
            input_names=names,
            output_names=["y"],
        )
    with tarfile.open(path) as archive:
        member = next(
            m
            for m in archive.getmembers()
            if m.name.endswith("proj.weight.dat")
        )
        weight_path = tmp_path / "weight.dat"
        weight_path.write_bytes(archive.extractfile(member).read())
        assert (
            DatBinHeader.from_dat(weight_path).torch_dtype_or_custom
            == DatBinHeader.TractCustomTypes.Q40
        )
    for args, reference in zip(cases, expected, strict=True):
        input_path, output_path = (
            tmp_path / "inputs.npz",
            tmp_path / "outputs.npz",
        )
        np.savez(
            input_path,
            **dict(zip(names, [x.numpy() for x in args], strict=True)),
        )
        subprocess.run(
            cli.run_save_outputs_cmd_str(path, input_path, output_path),
            check=True,
            capture_output=True,
        )
        with np.load(output_path) as result:
            np.testing.assert_allclose(
                result["y"], reference, atol=2e-3, rtol=2e-3
            )


def _raise_boundary_failure():
    from torch_to_nnef.tensor.named import OpaqueTraceBoundary

    raise OpaqueTraceBoundary("fake activation reached a real named operand")


@pytest.mark.parametrize("retry_error", [False, True])
def test_retry_restores_inputs_buffers_rng_and_keeps_metadata_refs(
    monkeypatch, retry_error
):
    from torch_to_nnef.tensor import opaque
    from torch_to_nnef.tensor.opaque import set_opaque_tensor_in_params_as_ref
    from torch_to_nnef.torch_graph.opaque_trace import trace_with_opaque_retry

    if not NEW_OPAQUE_TRACING_STRATEGY:
        pytest.skip("requires default opaque tracing")
    model = DEFAULT_COMPRESSION["min_max_q4_0"](MixedWeights(True).eval())
    model.register_buffer("counter", torch.tensor(1.0))
    set_opaque_tensor_in_params_as_ref(model)
    assert model.proj.weight.meta_tensor.device.type == "meta"
    x = torch.ones(1, 5, 32)
    original_buffer = model.counter
    rng = torch.random.get_rng_state().clone()
    calls = []

    def trace(module, args, **kwargs):
        calls.append(opaque._MATERIALIZED_TRACE.get())
        if len(calls) == 1:
            args[0].add_(10)
            module.counter.add_(5)
            module.counter = torch.tensor(99.0)
            torch.rand(3)
            _raise_boundary_failure()
        assert torch.equal(args[0], torch.ones_like(x))
        assert module.counter is original_buffer
        assert module.counter.item() == 1
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert module.proj.weight.meta_tensor.device.type == "meta"
        if retry_error:
            args[0].add_(20)
            module.counter.add_(20)
            torch.rand(3)
            raise RuntimeError("concrete trace failed")
        return "traced"

    monkeypatch.setattr(torch.jit, "trace", trace)
    if retry_error:
        with pytest.raises(RuntimeError, match="concrete trace failed"):
            trace_with_opaque_retry(model, (x,))
        assert torch.equal(x, torch.ones_like(x))
        assert model.counter.item() == 1
        assert torch.equal(torch.random.get_rng_state(), rng)
    else:
        assert trace_with_opaque_retry(model, (x,)) == "traced"
    assert calls == [False, True]
    assert not opaque._MATERIALIZED_TRACE.get()


def test_unrelated_trace_failure_is_not_retried(monkeypatch):
    from torch_to_nnef.tensor.opaque import set_opaque_tensor_in_params_as_ref
    from torch_to_nnef.torch_graph.opaque_trace import trace_with_opaque_retry

    model = DEFAULT_COMPRESSION["min_max_q4_0"](MixedWeights(False).eval())
    set_opaque_tensor_in_params_as_ref(model)
    calls = []

    def trace(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("unrelated model failure")

    monkeypatch.setattr(torch.jit, "trace", trace)
    with pytest.raises(RuntimeError, match="unrelated model failure"):
        trace_with_opaque_retry(model, (torch.ones(1, 5, 32),))
    assert len(calls) == 1
