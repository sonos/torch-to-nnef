import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    check_model_io_test,
    skipif_limited_offload_support,
    skipif_unsupported_qtensor,
)
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.nnef_io.writer import Writer, write_nnef_tensor
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.opaque import (
    OFFLOAD_STATE_KEY,
    OpaqueTensor,
    SupportsOffloadState,
)
from torch_to_nnef.tensor.quant.qtract import (
    QTensorTractScaleOnly,
    fp_to_tract_q4_0_with_min_max_calibration,
)


@skipif_limited_offload_support
def test_int_opaque_tensor_is_materialized_for_meta_trace():
    # Integer opaque params (codebooks / index tables) carry values tracing
    # must read, so a "meta" trace request must still materialize real data
    # instead of a valueless meta placeholder. Float weights stay meta (that
    # is the RAM optimization); only whole-number dtypes are materialized.
    int_weight = torch.arange(24).reshape(4, 6)
    assert not int_weight.dtype.is_floating_point
    with tempfile.TemporaryDirectory() as td:
        offloaded = OffloadedTensor.from_original_tensor(
            int_weight, "codebook", offload_dir=Path(td)
        )
        traced = offloaded._to_trace_tensor("meta")
        assert traced.device.type != "meta"
        assert torch.equal(traced, int_weight)


@skipif_limited_offload_support
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_offload_tensor_export_with_tract_and_conv2d(inference_target):
    with torch.no_grad():
        k = 3
        y = 8
        in_size = 128
        multiplier = 2
        chan_size = k * multiplier
        test_input = torch.arange(in_size * k * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, k, chan_size)
        ker = (k, k)
        model = torch.nn.Conv2d(in_size, y, kernel_size=ker).eval()

        original_weight = (
            torch.arange(in_size * y * k * k).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        with tempfile.TemporaryDirectory() as td:
            offloaded_value = OffloadedTensor.from_original_tensor(
                original_weight, "my_offloaded_weight", offload_dir=Path(td)
            )
            # offloaded_value = original_weight
            model.weight = torch.nn.Parameter(
                offloaded_value, requires_grad=False
            )
            check_model_io_test(
                model=model,
                test_input=test_input,
                inference_target=inference_target,
            )


INFERENCE_TARGET = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
INFERENCE_TARGET.check_io_tolerance = TractCheckTolerance.VERY


@skipif_unsupported_qtensor
def test_offload_qtensor_export():
    with torch.no_grad():
        k = 3
        y = 8
        in_size = 32
        multiplier = 2
        chan_size = k * multiplier
        test_input = torch.arange(in_size * k * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, k, chan_size)
        ker = (k, k)
        model = torch.nn.Conv2d(in_size, y, kernel_size=ker).eval()

        n_elms = in_size * y * k * k
        original_weight = (
            torch.arange(n_elms).reshape(y, -1, *ker).float()
            % (n_elms // 8)
            / n_elms
        )

        assert original_weight.shape == model.weight.shape
        with tempfile.TemporaryDirectory() as td:
            q_tensor = fp_to_tract_q4_0_with_min_max_calibration(
                original_weight
            )
            offloaded_value = OffloadedTensor.from_original_tensor(
                q_tensor, "my_offloaded_weight", offload_dir=Path(td)
            )
            # offloaded_value = original_weight
            model.weight = torch.nn.Parameter(
                offloaded_value, requires_grad=False
            )
            check_model_io_test(
                model=model,
                test_input=test_input,
                inference_target=INFERENCE_TARGET,
            )


@skipif_unsupported_qtensor
def test_offload_qtensor_uses_compact_serialized_state():
    with tempfile.TemporaryDirectory() as td:
        original_weight = torch.randn(8192, 32, dtype=torch.float16)
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        offloaded_value = OffloadedTensor.from_original_tensor(
            q_tensor, "my_offloaded_qweight", offload_dir=Path(td)
        )

        fp_payload_size = (
            original_weight.numel() * original_weight.element_size()
        )
        assert offloaded_value.offload_path.stat().st_size < fp_payload_size
        reloaded = offloaded_value.reload()
        assert type(reloaded) is type(q_tensor)
        assert torch.allclose(reloaded.decompress(), q_tensor.decompress())


@skipif_unsupported_qtensor
def test_offload_qtensor_write_uses_compact_state_without_reload(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        original_weight = torch.randn(4, 32)
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        offloaded_value = OffloadedTensor.from_original_tensor(
            q_tensor, "my_offloaded_qweight", offload_dir=td_path
        )

        ref_dir = td_path / "ref"
        out_dir = td_path / "out"
        ref_dir.mkdir()
        out_dir.mkdir()
        q_tensor.write_in_file(ref_dir, "weight", INFERENCE_TARGET)

        def fail_if_reloaded(*args, **kwargs):
            raise AssertionError("offloaded qtensor writer should not reload")

        def fail_if_full_content_is_built(*args, **kwargs):
            raise AssertionError("offloaded qtensor writer should stream")

        monkeypatch.setattr(OffloadedTensor, "reload", fail_if_reloaded)
        monkeypatch.setattr(
            QTensorTractScaleOnly,
            "_build_binary_dat_content",
            fail_if_full_content_is_built,
        )

        assert offloaded_value.write_qtensor_in_file(
            out_dir, "weight", INFERENCE_TARGET
        )
        assert (out_dir / "weight.dat").read_bytes() == (
            ref_dir / "weight.dat"
        ).read_bytes()


@skipif_unsupported_qtensor
def test_writer_uses_compact_offloaded_qtensor_path(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        original_weight = torch.randn(4, 32)
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        offloaded_value = OffloadedTensor.from_original_tensor(
            q_tensor, "my_offloaded_qweight", offload_dir=td_path
        )

        ref_dir = td_path / "ref"
        out_dir = td_path / "out"
        ref_dir.mkdir()
        out_dir.mkdir()
        q_tensor.write_in_file(ref_dir, "weight", INFERENCE_TARGET)

        def fail_if_reloaded(*args, **kwargs):
            raise AssertionError("writer should use compact offloaded state")

        monkeypatch.setattr(OffloadedTensor, "reload", fail_if_reloaded)

        graph = SimpleNamespace(
            operations=[
                SimpleNamespace(
                    type="variable",
                    attribs={
                        "custom_datatype": "quant_tensor",
                        "label": "weight",
                    },
                    output=SimpleNamespace(qtensor=offloaded_value),
                )
            ]
        )

        Writer(inference_target=INFERENCE_TARGET)._write_tensors_from_operators(
            graph, out_dir
        )

        assert (out_dir / "weight.dat").read_bytes() == (
            ref_dir / "weight.dat"
        ).read_bytes()


@skipif_limited_offload_support
def test_writer_uses_plain_offloaded_tensor_path(monkeypatch, tmp_path):
    original_weight = torch.randn(4, 32, dtype=torch.float16)
    offloaded_value = OffloadedTensor.from_original_tensor(
        original_weight, "my_offloaded_weight", offload_dir=tmp_path
    )

    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    out_dir.mkdir()
    write_nnef_tensor(
        original_weight.detach().numpy(),
        ref_dir / "weight.dat",
        quantized=False,
    )

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("writer should stream plain offloaded tensor")

    monkeypatch.setattr(OffloadedTensor, "reload", fail_if_reloaded)

    graph = SimpleNamespace(
        operations=[
            SimpleNamespace(
                type="variable",
                attribs={"label": "weight"},
                output=SimpleNamespace(
                    data=offloaded_value,
                    quant=None,
                ),
            )
        ]
    )

    Writer(inference_target=INFERENCE_TARGET)._write_tensors_from_operators(
        graph, out_dir
    )

    assert (out_dir / "weight.dat").read_bytes() == (
        ref_dir / "weight.dat"
    ).read_bytes()


class _ToyOpaque(OpaqueTensor, SupportsOffloadState):
    # A second SupportsOffloadState implementation, unrelated to QTensor, used
    # to prove the offload layer dispatches on the serialized state alone. It
    # deliberately has NO ``to_device`` method: the reload path must not
    # require one (device placement is owned by ``from_offload_state``).
    OFFLOAD_STATE_TAG = "tests.toy_opaque.v1"
    last_target_device = None

    @staticmethod
    def __new__(cls, payload):
        obj = torch.Tensor._make_subclass(cls, payload.to(torch.float32), False)
        obj.payload = payload
        return obj

    def _to_base_tensor(self):
        return self.payload.clone()

    def to_offload_state(self):
        return {OFFLOAD_STATE_KEY: self.OFFLOAD_STATE_TAG, "p": self.payload}

    @classmethod
    def from_offload_state(cls, state, target_device=None):
        assert cls.is_offload_state(state), state
        cls.last_target_device = target_device
        payload = state["p"]
        if target_device is not None:
            payload = payload.to(target_device)
        return cls(payload)

    @classmethod
    def write_offload_state_in_file(
        cls, state, dirpath, label, inference_target=None
    ):
        (Path(dirpath) / f"{label}.dat").write_bytes(
            state["p"].numpy().tobytes()
        )


class _NonOpaqueState(torch.Tensor, SupportsOffloadState):
    # A SupportsOffloadState that is NOT an OpaqueTensor. reload() only rebuilds
    # compact state through its OpaqueTensor branch, so _save must NOT compact
    # this one (it would be unreloadable). Used to lock the save/reload gate.
    OFFLOAD_STATE_TAG = "tests.non_opaque_state.v1"
    to_offload_state_called = False

    @staticmethod
    def __new__(cls, base):
        return torch.Tensor._make_subclass(cls, base, False)

    def to_offload_state(self):
        type(self).to_offload_state_called = True
        return {OFFLOAD_STATE_KEY: self.OFFLOAD_STATE_TAG}

    @classmethod
    def from_offload_state(cls, state, target_device=None):
        raise AssertionError("non-opaque state must be pickled plainly")

    @classmethod
    def write_offload_state_in_file(
        cls, state, dirpath, label, inference_target=None
    ):
        raise AssertionError("non-opaque state has no compact writer")


def test_save_does_not_compact_non_opaque_offload_state(tmp_path):
    _NonOpaqueState.to_offload_state_called = False
    tensor = _NonOpaqueState(torch.zeros(2, 2))
    OffloadedTensor._save(tensor, tmp_path, "x")

    assert not _NonOpaqueState.to_offload_state_called
    path = OffloadedTensor._offload_path(tmp_path, "x", tensor.dtype)
    raw = torch.load(path, weights_only=False)
    assert not (isinstance(raw, dict) and OFFLOAD_STATE_KEY in raw)


@skipif_limited_offload_support
def test_offload_dispatches_generic_offload_state(tmp_path):
    # The offload layer must dispatch on the serialized state, not on any
    # concrete tensor class. _ToyOpaque round-trips through save/reload/write
    # without OffloadedTensor knowing anything about it.
    assert not hasattr(_ToyOpaque, "to_device")
    _ToyOpaque.last_target_device = None
    payload = torch.arange(6).reshape(2, 3)
    offloaded = OffloadedTensor.from_original_tensor(
        _ToyOpaque(payload), "toy", offload_dir=tmp_path
    )
    # _save dispatched to to_offload_state: the pickle is the compact dict,
    # not the tensor subclass.
    raw = torch.load(offloaded.offload_path, weights_only=False)
    assert isinstance(raw, dict)
    assert raw[OFFLOAD_STATE_KEY] == _ToyOpaque.OFFLOAD_STATE_TAG
    # reload dispatched to from_offload_state via the registry and delegated
    # device placement to it (no to_device call on the reconstructed object).
    reloaded = offloaded.reload()
    assert type(reloaded) is _ToyOpaque
    assert torch.equal(reloaded.payload, payload)
    assert _ToyOpaque.last_target_device == offloaded.target_device
    # write dispatched generically too.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert offloaded.write_qtensor_in_file(out_dir, "toy", None)
    assert (out_dir / "toy.dat").read_bytes() == payload.numpy().tobytes()


def test_offload_state_subclass_without_own_tag_is_not_reregistered():
    # Subclassing a registered SupportsOffloadState without redeclaring the tag
    # must not raise (nor shadow the ancestor in the registry).
    from torch_to_nnef.tensor.opaque import resolve_offload_state

    class _ToySub(_ToyOpaque):
        pass

    assert _ToySub.OFFLOAD_STATE_TAG == _ToyOpaque.OFFLOAD_STATE_TAG
    resolved = resolve_offload_state(
        {OFFLOAD_STATE_KEY: _ToyOpaque.OFFLOAD_STATE_TAG}
    )
    assert resolved is _ToyOpaque


@skipif_limited_offload_support
def test_offload_change_dtype():
    with tempfile.TemporaryDirectory() as td:
        offloaded_value = OffloadedTensor.from_original_tensor(
            torch.rand(10, 10, dtype=torch.float32),
            "my_offloaded_weight",
            offload_dir=Path(td),
        )
        offloaded_value = offloaded_value.to(torch.float16)
        assert offloaded_value.dtype == torch.float16
        assert offloaded_value.reload().dtype == torch.float16


@skipif_limited_offload_support
def test_offload_set_updates_payload():
    with tempfile.TemporaryDirectory() as td:
        offloaded_value = OffloadedTensor.from_original_tensor(
            torch.ones(2, 3),
            "my_offloaded_weight",
            offload_dir=Path(td),
        )
        new_value = torch.arange(6).reshape(2, 3).float()
        offloaded_value.set_(new_value)
        assert torch.equal(offloaded_value.reload(), new_value)


@skipif_unsupported_qtensor
def test_offload_set_updates_nested_qtensor_payload():
    with tempfile.TemporaryDirectory() as td:
        original_weight = torch.arange(64).reshape(2, 32).float()
        offloaded_value = OffloadedTensor.from_original_tensor(
            original_weight,
            "my_offloaded_weight",
            offload_dir=Path(td),
        )
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        offloaded_q_tensor = OffloadedTensor.from_original_tensor(
            q_tensor,
            "my_offloaded_qweight",
            offload_dir=Path(td),
        )

        offloaded_value.set_(offloaded_q_tensor)

        reloaded = offloaded_value.reload()
        assert type(reloaded) is type(q_tensor)
        assert torch.allclose(reloaded.decompress(), q_tensor.decompress())
