"""Default opaque tracing must compose with ordinary weights and state."""

import pytest
import torch
from torch import nn

from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX
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
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_mixed_weight_dynamic_runtime_and_packed_storage(
    tmp_path, with_state, inference_target
):
    import subprocess
    import tarfile

    import numpy as np

    from torch_to_nnef.nnef_io.tensor import DatBinHeader

    torch.manual_seed(7)
    model = DEFAULT_COMPRESSION["min_max_q4_0"](MixedWeights(with_state).eval())
    cli = inference_target.tract_cli
    axes = {"x": {1: "S"}}
    names = ["x", "history"] if with_state else ["x"]
    cases = [
        (torch.randn(1, s, 32), torch.randn(1, 32, 2))
        if with_state
        else (torch.randn(1, s, 32),)
        for s in (5, 3, 7)
    ]
    path = tmp_path / "model.nnef.tgz"
    target = inference_target.with_dynamic_axes(axes)
    target.check_io = False
    with torch.no_grad():
        expected = [model(*args).numpy() for args in cases]
        export_model_to_nnef(
            model,
            cases[0],
            path,
            inference_target=target,
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
    from torch_to_nnef.exceptions import T2NErrorOpaqueTraceBoundary

    raise T2NErrorOpaqueTraceBoundary(
        "fake activation reached a real named operand"
    )


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


@pytest.mark.skipif(
    not NEW_OPAQUE_TRACING_STRATEGY, reason="requires default opaque tracing"
)
def test_retry_after_inplace_input_transpose():
    from torch_to_nnef.tensor.named import apply_name_to_tensor_in_module
    from torch_to_nnef.tensor.opaque import set_opaque_tensor_in_params_as_ref
    from torch_to_nnef.torch_graph.opaque_trace import trace_with_opaque_retry

    class TransposedInput(MixedWeights):
        def forward(self, x):
            x.transpose_(1, 2)
            return super().forward(x)

    torch.manual_seed(7)
    model = DEFAULT_COMPRESSION["min_max_q4_0"](TransposedInput(False).eval())
    x = torch.randn(1, 32, 5)
    with torch.no_grad():
        expected = model(x.clone())
        apply_name_to_tensor_in_module(model)
        set_opaque_tensor_in_params_as_ref(model)
        traced = trace_with_opaque_retry(
            model, (x.clone(),), check_trace=False, strict=False
        )
        torch.testing.assert_close(traced(x.clone()), expected)


@pytest.mark.parametrize(
    "mutation", ["transpose", "offset", "resize", "storage"]
)
def test_checkpoint_restores_original_tensor_metadata_and_aliases(mutation):
    from torch_to_nnef.torch_graph.opaque_trace import _TraceCheckpoint

    base = torch.arange(30.0)
    x = base[3:15].view(3, 4)
    buffer = base[15:27].view(3, 4)
    model = nn.Module()
    model.register_buffer("state", buffer)
    checkpoint = _TraceCheckpoint(model, (x,))
    originals = [t.clone() for t in (x, buffer)]
    for tensor in (x, buffer):
        if mutation == "transpose":
            tensor.transpose_(0, 1)
        elif mutation == "offset":
            tensor.as_strided_((3, 4), (4, 1), 0)
        elif mutation == "resize":
            tensor.resize_(2, 6)
        else:
            tensor.set_(torch.zeros(3, 4))
    checkpoint.restore()
    assert model.state is buffer
    for tensor, saved, offset in zip(
        (x, buffer), originals, (3, 15), strict=True
    ):
        assert tensor.shape == (3, 4)
        assert tensor.stride() == (4, 1)
        assert tensor.storage_offset() == offset
        assert tensor.data_ptr() == base[offset:].data_ptr()
        torch.testing.assert_close(tensor, saved)
    # Restoration preserves shared storage, not just matching values.
    x[0, 0] = -1
    assert base[3] == -1


def test_checkpoint_restores_stride_even_when_values_are_equal():
    from torch_to_nnef.torch_graph.opaque_trace import _TraceCheckpoint

    x = torch.zeros(3, 3).t()
    checkpoint = _TraceCheckpoint(nn.Module(), (x,))
    x.transpose_(0, 1)
    checkpoint.restore()
    assert x.stride() == (1, 3)
