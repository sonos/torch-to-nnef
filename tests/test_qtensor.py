import operator
import platform
import tarfile
import time
import typing as T
from copy import deepcopy
from datetime import datetime
from functools import partial, reduce
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from torch_to_nnef.compress import (
    DEFAULT_COMPRESSION_REGISTRY,
    dynamic_load_registry,
    quantize_weights_min_max_Q4_K,
    quantize_weights_min_max_Q8_1,
)
from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorMisuse,
    T2NErrorNotImplemented,
    T2NErrorTestFailed,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import TractCheckTolerance, TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.opaque import set_opaque_tensor_in_params_as_ref
from torch_to_nnef.tensor.quant import (
    QTensor,
    QTensorTractQ4_K,
    QTensorTractQ8_1,
    QTensorTractScaleOnly,
    U8Compressor,
    fp_to_tract_q4_0_with_min_max_calibration,
    fp_to_tract_q4_k,
    fp_to_tract_q8_1_with_min_max_calibration,
    qscale_per_group_f16_min_max_calibration,
)
from torch_to_nnef.torch_graph.ir_graph import module_tracer_into_ir_graph
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.utils import SemanticVersion, cd, torch_version

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TRACT_INFERENCES_TO_TESTS_EXACT,
    check_model_io_test,
    skipif_no_meta_opaque_tracing,
    skipif_unsupported_qtensor,
)


def check_tensor_in_nnef_archive(
    inference_target: InferenceTarget,
    path: Path,
    labels: T.Union[T.List[str], T.Dict[str, T.Dict[str, T.Any]]],
):
    """Helper to check .dat variable are dumped with expected dtype in NNEF."""
    assert isinstance(inference_target, InferenceTarget)
    if not isinstance(inference_target, TractNNEF):
        return
    assert path.exists()
    exdir = path.parent / "extract"
    exdir.mkdir(parents=True, exist_ok=True)
    graph_filename = "graph.nnef"
    with cd(exdir):
        # Extract graph.nnef from either .tar or .tgz
        with tarfile.open(path, "r:*") as tf:
            member = None
            for m in tf.getmembers():
                if m.name.endswith(graph_filename):
                    member = m
                    break
            if member is None:
                raise T2NErrorTestFailed("graph.nnef not found in archive")
            tf.extract(member, exdir)
        graph_filepath = exdir / graph_filename
        graph_content = graph_filepath.read_text()
        found_labels = set()
        probe1 = "variable<scalar>(label = "
        probe2 = "variable(label = "
        for line in graph_content.split("\n"):
            for probe in [probe1, probe2]:
                if probe in line:
                    col_ix = line.index(probe)
                    start_ix = col_ix + len(probe)
                    line_label = line[start_ix + 1 :].split("'", maxsplit=1)[0]
                    for lab in labels:
                        if line_label == lab:
                            assert lab not in found_labels, (
                                "duplicate weight label"
                            )
                            found_labels.add(lab)
        remaining_labels = set(labels).difference(found_labels)
        if remaining_labels:
            raise T2NErrorTestFailed(
                "Some tensor where not found in exported NNEF archive: "
                f"{remaining_labels}"
            )
        if isinstance(labels, dict):
            for label_name, label_opt_checks in labels.items():
                expected_dtype = label_opt_checks.get("dtype")
                if expected_dtype is not None:
                    dat_filename = f"{label_name}.dat"
                    with tarfile.open(path, "r:*") as tf:
                        member = None
                        for m in tf.getmembers():
                            if m.name.endswith(dat_filename):
                                member = m
                                break
                        if member is None:
                            raise T2NErrorTestFailed(
                                f"{dat_filename} not found in archive"
                            )
                        tf.extract(member, exdir)
                    bin_header = DatBinHeader.from_dat(exdir / dat_filename)
                    if bin_header.torch_dtype_or_custom != expected_dtype:
                        raise T2NErrorTestFailed(
                            "wrong dtype in NNEF archive "
                            f"{label_name}: {bin_header.torch_dtype_or_custom}"
                            f" but expected {expected_dtype}"
                        )


def _legacy_q40_dat_content(
    qtensor: QTensorTractScaleOnly, post_tract_21_11: bool
) -> bytes:
    tensor_flat = qtensor.decompress_to_u8().clone().flatten()
    if post_tract_21_11:
        tensor_per_group = tensor_flat.reshape(-1, 2, 16)
        tensor_per_group[:, 1, :] <<= 4
        tensor_per_group = tensor_per_group.sum(dim=1).numpy().astype(np.uint8)
    else:
        tensor_per_group = tensor_flat.reshape(-1, 16, 2)
        tensor_per_group[:, :, 1] <<= 4
        tensor_per_group = tensor_per_group.sum(dim=2).numpy().astype(np.uint8)

    b_scales = qtensor.qscheme.scale.numpy().view(np.byte)
    b_vals = tensor_per_group.view(np.byte)
    return np.hstack([b_scales, b_vals]).tobytes()


@skipif_unsupported_qtensor
@pytest.mark.parametrize("post_tract_21_11", [False, True])
def test_q40_dat_content_uses_bitpacking_without_sum(
    monkeypatch, post_tract_21_11
):
    qtensor = fp_to_tract_q4_0_with_min_max_calibration(
        torch.arange(6 * 32).reshape(6, 32).float()
    )
    u8_before = qtensor.u8_blob.clone()
    expected = _legacy_q40_dat_content(qtensor, post_tract_21_11)

    def fail_if_sum_is_used(*args, **kwargs):
        raise AssertionError("Q40 writer should pack nibbles without sum")

    monkeypatch.setattr(torch.Tensor, "sum", fail_if_sum_is_used)

    inference_target = TractNNEF(
        "0.21.11" if post_tract_21_11 else "0.21.10", check_io=False
    )
    assert qtensor._build_binary_dat_content(inference_target) == expected
    assert torch.equal(qtensor.u8_blob, u8_before)


@skipif_unsupported_qtensor
def test_q40_write_streams_without_building_full_content(monkeypatch, tmp_path):
    qtensor = fp_to_tract_q4_0_with_min_max_calibration(
        torch.arange(6 * 32).reshape(6, 32).float()
    )
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    out_dir.mkdir()
    qtensor.write_in_file(ref_dir, "weight", TractNNEF.latest())

    def fail_if_full_content_is_built(*args, **kwargs):
        raise AssertionError("Q40 write should stream content")

    monkeypatch.setattr(
        QTensorTractScaleOnly,
        "_build_binary_dat_content",
        fail_if_full_content_is_built,
    )

    qtensor.write_in_file(out_dir, "weight", TractNNEF.latest())
    assert (out_dir / "weight.dat").read_bytes() == (
        ref_dir / "weight.dat"
    ).read_bytes()


@skipif_unsupported_qtensor
def test_quantize_with_tract_q4_0_and_manipulate_tensor():
    original_weight = torch.arange(64).reshape(2, 32).float()
    q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
    _ = q_tensor.u8_blob  # check access works
    new_q_tensor = q_tensor.to(torch.float32)
    _ = new_q_tensor.u8_blob  # check access works

    class Test(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(new_q_tensor, requires_grad=False)

        def forward(self, x):
            return x @ self.weight

    mod = Test()
    assert mod.weight.u8_blob.dtype == torch.uint8
    # mod.weight.data.u8_blob

    inp_tensor = torch.rand(4, 2)
    out = mod(inp_tensor)
    assert type(out) is type(inp_tensor)  # avoid propagation of qtype


@pytest.fixture
def forbid_qtensor_materialization(monkeypatch):
    """Fail the test if any quantized weight is decompressed during trace."""

    def fail_if_materialized(self):
        raise AssertionError("quantized weights must stay opaque during trace")

    monkeypatch.setattr(
        QTensorTractScaleOnly, "_to_base_tensor", fail_if_materialized
    )


@skipif_unsupported_qtensor
@skipif_no_meta_opaque_tracing
def test_quantized_opaque_trace_does_not_materialize_base_tensor(
    forbid_qtensor_materialization,
):
    model = nn.Linear(32, 8, bias=False).eval()
    q_tensor = fp_to_tract_q4_0_with_min_max_calibration(model.weight)
    model.weight = nn.Parameter(q_tensor, requires_grad=False)

    set_opaque_tensor_in_params_as_ref(model)
    module_tracer_into_ir_graph(
        TorchModuleTracer(model, args=(torch.randn(1, 32),))
    )


@skipif_unsupported_qtensor
@skipif_no_meta_opaque_tracing
def test_quantized_opaque_view_on_ref_does_not_materialize(
    forbid_qtensor_materialization,
):
    # `view` called directly on the opaque param ref (before any slicing)
    # must stay opaque instead of decompressing the full packed weight.
    class ViewThenSelect(nn.Module):
        def __init__(self):
            super().__init__()
            weight = fp_to_tract_q4_0_with_min_max_calibration(
                torch.randn(3, 4, 32)
            )
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            reshaped = self.weight.view(3, 4, 32)
            chunks = x.split([1, 1, 1], dim=0)
            outputs = [
                nn.functional.linear(chunks[e], reshaped[e]) for e in range(3)
            ]
            return torch.cat(outputs, dim=0)

    model = ViewThenSelect().eval()
    set_opaque_tensor_in_params_as_ref(model)
    module_tracer_into_ir_graph(
        TorchModuleTracer(model, args=(torch.randn(3, 32),))
    )


@skipif_unsupported_qtensor
@skipif_no_meta_opaque_tracing
def test_quantized_opaque_shape_ops_on_ref_do_not_materialize(
    forbid_qtensor_materialization,
):
    # Common weight-manipulation idioms (`.t()`, `.reshape()`, ...) applied
    # directly to the opaque param ref must stay opaque instead of
    # decompressing the full packed weight during trace.
    class ReshapeTranspose(nn.Module):
        def __init__(self):
            super().__init__()
            weight = fp_to_tract_q4_0_with_min_max_calibration(
                torch.randn(8, 32)
            )
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            # weight is (out=8, in=32); `.t()` -> (32, 8) so x @ w maps
            # (batch, 32) -> (batch, 8).
            w = self.weight.reshape(8, 32).t()
            return x @ w

    model = ReshapeTranspose().eval()
    set_opaque_tensor_in_params_as_ref(model)
    module_tracer_into_ir_graph(
        TorchModuleTracer(model, args=(torch.randn(1, 32),))
    )


@skipif_unsupported_qtensor
def test_opaque_trace_tensor_uses_fake_target_device_for_meta_request():
    q_tensor = fp_to_tract_q4_0_with_min_max_calibration(torch.randn(8, 32))

    traced = q_tensor._to_trace_tensor("meta")
    assert traced.device == q_tensor.device
    assert traced.shape == q_tensor.shape
    assert getattr(traced, "fake_mode", None) is not None

    # a real device must expose decompressed values, not uninitialized memory,
    # otherwise constant-index gathers bake garbage into the exported graph.
    cpu = q_tensor._to_trace_tensor("cpu")
    assert cpu.device.type == "cpu"
    assert torch.equal(cpu, q_tensor._to_base_tensor().to("cpu"))


@skipif_unsupported_qtensor
@skipif_no_meta_opaque_tracing
def test_opaque_fake_trace_allows_same_device_arithmetic(
    forbid_qtensor_materialization,
):
    # Fake target-device placeholders keep opaque weights non-materialized
    # while still allowing same-device arithmetic during trace.
    class ScaledViewedWeight(nn.Module):
        def __init__(self):
            super().__init__()
            weight = fp_to_tract_q4_0_with_min_max_calibration(
                torch.randn(8, 32)
            )
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.register_buffer("scale", torch.ones(8, 32))

        def forward(self, x):
            w = self.weight.view(8, 32) * self.scale
            return x @ w.t()

    model = ScaledViewedWeight().eval()
    set_opaque_tensor_in_params_as_ref(model)
    module_tracer_into_ir_graph(
        TorchModuleTracer(model, args=(torch.randn(1, 32),))
    )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_basic(inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = 0.0
        model.weight[0:5, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.decompress()
        diff = (original_weight - deq_weights).abs()
        assert diff.sum() == 0

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.sum() == 0
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_controled(inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        test_input = torch.zeros(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        original_weight = model.weight
        fp_res = model(test_input)

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.decompress()
        diff = (original_weight - deq_weights).abs()
        assert diff.mean() < 0.01, diff.mean()

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        q_res = model(test_input)
        abs_diff = (q_res - fp_res).abs()
        assert abs_diff.mean() < 0.01, diff.mean()
        if "arm" in platform.uname().machine.lower():
            inference_target = deepcopy(inference_target)
            inference_target.check_io_tolerance = TractCheckTolerance.SUPER

        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_rounding2(inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        nd = 32
        test_input = torch.rand(nd).float().reshape(1, nd)
        model = nn.Linear(nd, 2, bias=False).eval()
        model.weight[:, :] = 0.0
        offset = 2.0
        model.weight[0, 0:5] = torch.tensor(
            [
                1.0 - offset,
                2.0 - offset,
                3.0 - offset,
                4.0 - offset,
                5.0 - offset,
            ]
        )
        original_weight = model.weight

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
        deq_weights = q_tensor.decompress()
        diff = (original_weight - deq_weights).abs()
        assert diff.mean() < 0.006, diff.mean()

        model.weight = nn.Parameter(q_tensor, requires_grad=False)

        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_arange(inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        if (
            "arm" in platform.uname().machine.lower()
            or torch_version() < "2.0.0"
        ):
            inference_target = deepcopy(inference_target)
            inference_target.check_io_tolerance = TractCheckTolerance.SUPER

        test_input = torch.arange(960).float().reshape(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = torch.arange(16 * 96).float().reshape(16, 96)
        original_weight = model.weight

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        # can safely check io since all values controled
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


class DummyU8Compressor(U8Compressor):
    def __init__(self):
        self.compress_times = []
        self.decompress_times = []

    def compress(self, u8_tensor) -> torch.Tensor:
        time.sleep(0.01)
        self.compress_times.append(datetime.now())
        return u8_tensor

    def decompress(self, u8_tensor) -> torch.Tensor:
        time.sleep(0.01)
        self.decompress_times.append(datetime.now())
        return u8_tensor


@skipif_unsupported_qtensor
def test_u8_compressors():
    fp_tensor = torch.rand(2, 32)
    with torch.no_grad():
        q_scheme = qscale_per_group_f16_min_max_calibration(
            fp_tensor, n_bits=4, group_size=32, percentile=1
        )
        dummy_compressor1 = DummyU8Compressor()
        dummy_compressor2 = DummyU8Compressor()
        qtensor = QTensorTractScaleOnly(
            fp_tensor,
            qscheme=q_scheme,
            dequant_to_dtype=fp_tensor.dtype,
            u8_compressors=[dummy_compressor1, dummy_compressor2],
        )
        # compressed
        assert len(dummy_compressor1.compress_times) == 1
        assert len(dummy_compressor2.compress_times) == 1
        assert (
            dummy_compressor1.compress_times[0]
            < dummy_compressor2.compress_times[0]
        )
        # decompressed
        assert len(dummy_compressor1.decompress_times) == 1
        assert len(dummy_compressor2.decompress_times) == 1
        assert (
            dummy_compressor1.decompress_times[0]
            > dummy_compressor2.decompress_times[0]
        )
        # test
        torch.matmul(torch.rand(1, 2), qtensor)
        assert len(dummy_compressor1.compress_times) == 1
        assert len(dummy_compressor2.compress_times) == 1
        # decompressed
        assert len(dummy_compressor1.decompress_times) == 2
        assert len(dummy_compressor2.decompress_times) == 2
        assert (
            dummy_compressor1.decompress_times[1]
            > dummy_compressor2.decompress_times[1]
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [
        TractNNEF("0.21.10", check_io=False),
        TractNNEF("0.21.11", check_io=False),
    ],
)
def test_q40_offload_state_write_applies_u8_compressors(
    tmp_path, inference_target
):
    fp_tensor = torch.rand(2, 32)
    q_scheme = qscale_per_group_f16_min_max_calibration(
        fp_tensor, n_bits=4, group_size=32, percentile=1
    )
    qtensor = QTensorTractScaleOnly(
        fp_tensor,
        qscheme=q_scheme,
        dequant_to_dtype=fp_tensor.dtype,
        u8_compressors=[DummyU8Compressor()],
    )
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    out_dir.mkdir()

    qtensor.write_in_file(ref_dir, "weight", inference_target)
    QTensorTractScaleOnly.write_offload_state_in_file(
        qtensor.to_offload_state(),
        out_dir,
        "weight",
        inference_target=inference_target,
    )

    assert (out_dir / "weight.dat").read_bytes() == (
        ref_dir / "weight.dat"
    ).read_bytes()


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_assign_to(inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        test_input = torch.arange(960).float().reshape(10, 96)
        test_input[0, :] = 1
        model = nn.Linear(96, 16, bias=False).eval()
        model.weight[:, :] = torch.arange(16 * 96).float().reshape(16, 96)
        original_weight = model.weight

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device


TRACT_INFERENCES_TO_TESTS_APPROX_CONV = [
    _ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version >= "0.21.12"
]


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version >= "0.21.11"],
)
def test_quantize_with_tract_q4_0_embedding(inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        x = 6
        y = 32
        model = nn.Embedding(x, y).eval()
        original_weight = (torch.arange(x * y).reshape(x, y).float() * 2).half()

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


def _expected_q81_type(inference_target):
    if inference_target.version >= "0.23.10":
        return DatBinHeader.TractCustomTypes.Q81
    return DatBinHeader.TractCustomTypes.Q81_LEGACY


@skipif_unsupported_qtensor
def test_q81_dat_content_layout(tmp_path):
    fp_tensor = torch.arange(-64, 64).reshape(2, 64).float()
    qtensor = fp_to_tract_q8_1_with_min_max_calibration(fp_tensor)
    qtensor.write_in_file(tmp_path, "weight", TractNNEF.latest())

    with (tmp_path / "weight.dat").open("rb") as fh:
        header = DatBinHeader.from_dat_file(fh)
        content = fh.read()
    assert header.torch_dtype_or_custom == _expected_q81_type(
        TractNNEF.latest()
    )
    assert header.dims[: header.rank] == [2, 64]
    assert header.data_size_bytes == len(content) == 4 * 36

    blocks = np.frombuffer(content, dtype=np.uint8).reshape(4, 36)
    scale = blocks[:, :2].copy().view(np.float16).reshape(4, 1)
    block_sum = blocks[:, 2:4].copy().view(np.float16).reshape(4)
    i8_vals = blocks[:, 4:].copy().view(np.int8)
    assert np.allclose(scale.flatten(), qtensor.qscheme.scale.numpy().flatten())
    dequant = i8_vals.astype(np.float32) * scale.astype(np.float32)
    # like tract Q8_1 gather: dequantized in target dtype on all platforms
    assert np.array_equal(dequant.reshape(2, 64), qtensor.decompress().numpy())
    assert np.allclose(block_sum, dequant.sum(axis=1), rtol=1e-3)
    assert i8_vals.min() >= -127


@skipif_unsupported_qtensor
@pytest.mark.parametrize("percentile", [1.0, 0.9])
def test_q81_values_in_restricted_range(percentile):
    fp_tensor = torch.randn(64, 64)
    fp_tensor[:, 0] = -10.0  # outliers clamped if percentile < 1
    fp_tensor[:, 1] = 10.0
    qtensor = fp_to_tract_q8_1_with_min_max_calibration(
        fp_tensor, percentile=percentile
    )
    u8_vals = qtensor.decompress_to_u8()
    # u8 offset by 128: int8 in [-127, 127]
    assert u8_vals.min() >= 1 and u8_vals.max() <= 255
    deq = qtensor.decompress()
    assert torch.equal(deq[:, 0], -deq[:, 1])


@skipif_unsupported_qtensor
def test_q81_and_q40_are_distinct_formats():
    q81 = fp_to_tract_q8_1_with_min_max_calibration(torch.randn(2, 32))
    q40 = fp_to_tract_q4_0_with_min_max_calibration(torch.randn(2, 32))
    assert not isinstance(q81, QTensorTractScaleOnly)
    assert not isinstance(q40, QTensorTractQ8_1)


@skipif_unsupported_qtensor
def test_q81_write_streams_without_building_full_content(monkeypatch, tmp_path):
    qtensor = fp_to_tract_q8_1_with_min_max_calibration(
        torch.randn(6, 64), percentile=1.0
    )
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    out_dir.mkdir()
    qtensor.write_in_file(ref_dir, "weight", TractNNEF.latest())
    QTensorTractQ8_1.write_offload_state_in_file(
        qtensor.to_offload_state(),
        out_dir,
        "weight",
        inference_target=TractNNEF.latest(),
    )

    def fail_if_full_content_is_built(*args, **kwargs):
        raise AssertionError("Q81 write should stream content")

    monkeypatch.setattr(
        QTensorTractQ8_1,
        "_build_binary_dat_content",
        fail_if_full_content_is_built,
    )
    stream_dir = tmp_path / "stream"
    stream_dir.mkdir()
    qtensor.write_in_file(stream_dir, "weight", TractNNEF.latest())
    expected = (ref_dir / "weight.dat").read_bytes()
    assert (out_dir / "weight.dat").read_bytes() == expected
    assert (stream_dir / "weight.dat").read_bytes() == expected


@skipif_unsupported_qtensor
def test_q81_unsupported_before_tract_0_23(tmp_path):
    qtensor = fp_to_tract_q8_1_with_min_max_calibration(torch.randn(2, 32))
    with pytest.raises(T2NErrorNotImplemented):
        qtensor.write_in_file(
            tmp_path, "weight", TractNNEF("0.22.1", check_io=False)
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target,dtype",
    [
        (i, dt)
        for i in TRACT_INFERENCES_TO_TESTS_EXACT
        if i.version >= "0.23.0"
        for dt in [torch.float32, torch.float16]
    ],
)
def test_quantize_with_tract_q8_1_embedding(inference_target, dtype):
    """Q8_1 is only supported by tract on embedding/gathering."""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        x = 6
        y = 64
        model = nn.Embedding(x, y).eval().to(dtype)
        original_weight = torch.randn(x, y).to(dtype)

        q_tensor = fp_to_tract_q8_1_with_min_max_calibration(original_weight)
        deq_weight = q_tensor.qscheme.dequantize(
            q_tensor.decompress_to_u8(), target_dtype=torch.float32
        )
        # restricted range: no clamping, error within half a step
        # (codes computed in f32 even for f16 weights)
        diff = (original_weight.float() - deq_weight).abs().reshape(-1, 32)
        scale = q_tensor.qscheme.scale.abs().float()
        assert (diff <= scale * 0.501).all(), (diff / scale).max()

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {"dtype": _expected_q81_type(inference_target)}
                },
            ),
        )


@skipif_unsupported_qtensor
def test_compress_min_max_q8_1_embeddings_only_quantize_embeddings():
    model = nn.Sequential(nn.Embedding(10, 64), nn.Linear(64, 32))
    tied = nn.Sequential(nn.Embedding(10, 64), nn.Linear(64, 10, bias=False))
    tied[1].weight = tied[0].weight
    registry = dynamic_load_registry(DEFAULT_COMPRESSION_REGISTRY)

    registry["min_max_q8_1_embeddings"](model)
    assert isinstance(model[0].weight, QTensorTractQ8_1)
    assert not isinstance(model[1].weight, QTensor)

    # weight shared with a Linear must stay float (no tract Q8_1 matmul)
    registry["min_max_q8_1_embeddings"](tied)
    assert not isinstance(tied[0].weight, QTensor)


@skipif_unsupported_qtensor
def test_compress_min_max_q8_1_rejects_non_embeddings():
    model = nn.Sequential(nn.Embedding(10, 64), nn.Linear(64, 32))
    with pytest.raises(T2NErrorMisuse):
        quantize_weights_min_max_Q8_1(
            model, to_quantize_module_classes=(nn.Linear, nn.Embedding)
        )
    assert not isinstance(model[0].weight, QTensor)


@skipif_unsupported_qtensor
def test_q81_group_sum_clamped_to_f16_range(tmp_path, caplog):
    qtensor = fp_to_tract_q8_1_with_min_max_calibration(
        torch.full((2, 32), 3000.0)
    )
    with caplog.at_level("WARNING"):
        qtensor.write_in_file(tmp_path, "weight", TractNNEF.latest())
    assert "group sum exceed f16 range" in caplog.text
    with (tmp_path / "weight.dat").open("rb") as fh:
        DatBinHeader.from_dat_file(fh)
        blocks = np.frombuffer(fh.read(), dtype=np.uint8).reshape(2, 36)
    group_sum = blocks[:, 2:4].copy().view(np.float16)
    assert np.isfinite(group_sum).all(), group_sum


def _ggml_ramp(length: int, zeros: int) -> np.ndarray:
    """Ramp of tract `ggml_k` tests over [-2, 2] with leading zeros."""
    return np.array(
        [
            0.0 if i < zeros else (((i * 17 + 3) % 41) - 20.0) / 10.0
            for i in range(length)
        ],
        dtype=np.float32,
    )


def _check_ggml_reference(
    qtensor_cls, data: np.ndarray, expected_hex: str, values
):
    """Check quant bytes and dequant f32 bits against llama.cpp ones."""
    blocks = qtensor_cls.quantize_blocks(data.reshape(1, -1))
    assert blocks.tobytes().hex() == expected_hex
    ref_blocks = np.frombuffer(bytes.fromhex(expected_hex), dtype=np.uint8)
    deq = qtensor_cls.dequantize_blocks(ref_blocks.reshape(1, -1))
    for ix, bits in values:
        assert int(deq[0, ix].view(np.uint32)) == bits, ix


def _check_ggml_dat_content_layout(tmp_path, qtensor, item_type):
    """Check a ggml block `.dat` is the header + raw ggml blocks."""
    qtensor_cls = type(qtensor)
    qtensor.write_in_file(tmp_path, "weight", TractNNEF.latest())
    with (tmp_path / "weight.dat").open("rb") as fh:
        header = DatBinHeader.from_dat_file(fh)
        content = fh.read()
    n_blocks = qtensor.numel() // qtensor_cls.BLOCK_LEN
    assert header.torch_dtype_or_custom == item_type
    assert header.item_type_vendor == DatBinHeader.TRACT_ITEM_TYPE_VENDOR
    assert header.dims[: header.rank] == list(qtensor.shape)
    assert header.bits_per_item == 2**32 - 1
    assert header.data_size_bytes == len(content)
    assert len(content) == n_blocks * qtensor_cls.BLOCK_BYTES
    blocks = np.frombuffer(content, dtype=np.uint8).reshape(
        n_blocks, qtensor_cls.BLOCK_BYTES
    )
    assert np.array_equal(blocks, qtensor.decompress_to_u8().numpy())
    deq = qtensor_cls.dequantize_blocks(blocks).reshape(qtensor.shape)
    assert np.array_equal(deq, qtensor.decompress().float().numpy())


def _tract_target(version: str) -> TractNNEF:
    """Latest tract target relabeled with `version` (prereleases allowed)."""
    target = deepcopy(TractNNEF.latest())
    target.version = SemanticVersion.from_str(version)
    return target


def _check_ggml_offload_state_write(tmp_path, qtensor, target):
    """Offload state reload and direct write match the in-memory qtensor."""
    qtensor_cls = type(qtensor)
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    ref_dir.mkdir()
    out_dir.mkdir()
    qtensor.write_in_file(ref_dir, "weight", target)
    state = qtensor.to_offload_state()
    assert qtensor_cls.is_offload_state(state)
    qtensor_cls.write_offload_state_in_file(
        state, out_dir, "weight", inference_target=target
    )
    expected = (ref_dir / "weight.dat").read_bytes()
    assert (out_dir / "weight.dat").read_bytes() == expected
    reloaded = qtensor_cls.from_offload_state(state)
    assert type(reloaded) is qtensor_cls
    assert torch.equal(reloaded.decompress(), qtensor.decompress())
    # reloaded qtensor writes the same bytes
    reload_dir = tmp_path / "reload"
    reload_dir.mkdir()
    reloaded.write_in_file(reload_dir, "weight", target)
    assert (reload_dir / "weight.dat").read_bytes() == expected
    # same label written twice with same content is accepted
    qtensor.write_in_file(ref_dir, "weight", target)


def _check_ggml_rejects_old_tract(tmp_path, caplog, qtensor, version):
    """Both write paths warn with an upgrade hint and write no data."""
    qtensor_cls = type(qtensor)
    target = _tract_target(version)
    match = (
        rf"{qtensor_cls.FORMAT_NAME} export requires tract >= "
        + qtensor_cls.MIN_TRACT_VERSION.replace(".", r"\.")
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(T2NErrorNotImplemented, match=match):
            qtensor.write_in_file(tmp_path, "weight", target)
        with pytest.raises(T2NErrorNotImplemented, match=match):
            qtensor_cls.write_offload_state_in_file(
                qtensor.to_offload_state(), tmp_path, "offloaded", target
            )
    hint = (
        f"{qtensor_cls.FORMAT_NAME} export requires tract >= "
        f"{qtensor_cls.MIN_TRACT_VERSION}; target is {version}. "
        f"Upgrade tract before exporting {qtensor_cls.FORMAT_NAME} weights."
    )
    assert caplog.text.count(hint) == 2, caplog.text
    assert not list(tmp_path.glob("*.dat"))


def _check_ggml_decompress_matches_reference(qtensor_cls, fp_tensor):
    """Check decompress is the numpy reference dequant in source dtype."""
    qtensor = qtensor_cls.from_fp(fp_tensor)
    assert qtensor.shape == fp_tensor.shape
    assert qtensor.numel() == fp_tensor.numel()
    n_blocks = fp_tensor.numel() // qtensor_cls.BLOCK_LEN
    u8_blocks = qtensor.decompress_to_u8().numpy()
    assert u8_blocks.shape == (n_blocks, qtensor_cls.BLOCK_BYTES)
    fp32_blocks = fp_tensor.float().numpy().reshape((n_blocks, -1))
    assert np.array_equal(u8_blocks, qtensor_cls.quantize_blocks(fp32_blocks))
    expected = (
        torch.from_numpy(qtensor_cls.dequantize_blocks(u8_blocks))
        .reshape(fp_tensor.shape)
        .to(fp_tensor.dtype)
    )
    deq = qtensor.decompress()
    assert deq.dtype == fp_tensor.dtype
    assert torch.equal(deq, expected)


def _ggml_invalid_weights(block_len: int):
    """Weights any ggml block format must reject (shape, dtype, values)."""
    return [
        pytest.param(torch.ones(block_len), id="rank1"),
        pytest.param(torch.ones(1, 1, 1, 1, block_len), id="rank5"),
        pytest.param(torch.ones(2, block_len // 2), id="incomplete_block"),
        pytest.param(
            torch.ones(2, block_len, dtype=torch.int32), id="non_floating"
        ),
        pytest.param(torch.empty(0, block_len), id="empty"),
        pytest.param(torch.full((2, block_len), float("nan")), id="nan"),
        pytest.param(torch.full((2, block_len), float("inf")), id="inf"),
    ]


# llama.cpp quantize_row_q4_K_ref / dequantize_row_q4_K vectors
# (from tract v0.23.10 linalg/src/frame/block_quant/ggml_k.rs tests)
Q4_K_GGML_REFS = [
    (
        0,
        "4e1c2028fefefffcfbfefdfeddbdfede51c73e950b72d84fa61c83ea50c72d94fb"
        "62d83fa51c73e950b72d84fa61c83ee950b72d94fa61d83ea50c73e940b62d84fa"
        "61c73ea50b72d94fb61c83fa51c762d94fa61d73ea51b72e95fb62c83fa60c73ea"
        "40b72e84fb51c83f950c73d940fb62c83f950c73d940b71d84ea51c83e950c62d9"
        "4fa61d84ea51b72e95fb62d9",
        [(0, 0xBFD20380), (32, 0xBF31E300), (255, 0x3FB61280)],
    ),
    (
        32,
        "4e1c2028c0fefffcc0fefdfeddbdfede50c030900070d040a01080e050c02090f0"
        "60d030a01070e050b02080f060c030e950b72d94fa61d83ea50c73e940b62d84fa"
        "61c73ea50b72d94fb61c83fa51c762d94fa61d73ea51b72e95fb62c83fa60c73ea"
        "40b72e84fb51c83f950c73d940fb62c83f950c73d940b71d84ea51c83e950c62d9"
        "4fa61d84ea51b72e95fb62d9",
        [(0, 0), (32, 0xBF31E300), (255, 0x3FB61280)],
    ),
    (256, "00" * 144, [(0, 0)]),
]


@skipif_unsupported_qtensor
@pytest.mark.parametrize("zeros,expected_hex,values", Q4_K_GGML_REFS)
def test_q4k_matches_ggml_reference(zeros, expected_hex, values):
    _check_ggml_reference(
        QTensorTractQ4_K, _ggml_ramp(256, zeros), expected_hex, values
    )


@skipif_unsupported_qtensor
def test_q4k_dat_content_layout(tmp_path):
    qtensor = fp_to_tract_q4_k(torch.randn(3, 512))
    assert qtensor.u8_blob.shape == (6, 144)
    _check_ggml_dat_content_layout(
        tmp_path, qtensor, DatBinHeader.TractCustomTypes.Q4K
    )


@skipif_unsupported_qtensor
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q4k_dequant_round_trip_error_bounded(seed):
    # same bound as tract q4_k_roundtrip
    ramp = torch.from_numpy(_ggml_ramp(512, 0)).reshape(2, 256)
    qtensor = fp_to_tract_q4_k(ramp)
    assert (qtensor.decompress() - ramp).abs().max() < 0.35
    zeros = fp_to_tract_q4_k(torch.zeros(2, 256))
    assert torch.equal(zeros.decompress(), torch.zeros(2, 256))
    gen = torch.Generator().manual_seed(seed)
    # rows of very different magnitudes
    fp_tensor = torch.randn(16, 1024, generator=gen) * 10 ** torch.linspace(
        -3, 3, 16
    ).reshape(16, 1)
    deq = fp_to_tract_q4_k(fp_tensor).decompress()
    # Bound per super-block, not per sub-block: the 6bit quantized
    # sub-block scales and mins (relative to the super-block d and dmin)
    # can exceed one sub-block step (measured up to 1.06). Per super-block
    # the error stays below 0.81 step (range / 15) over 300 seeds,
    # 2 steps leaves a robust margin.
    sb_fp = fp_tensor.reshape(-1, 256)
    sb_range = sb_fp.max(dim=1).values - sb_fp.min(dim=1).values
    sb_err = (deq - fp_tensor).abs().reshape(-1, 256).max(dim=1).values
    assert (sb_err <= 2 * sb_range / 15).all(), (sb_err / sb_range).max()
    # relative error of a 4.5bpw format (measured below 0.075)
    assert (deq - fp_tensor).norm() / fp_tensor.norm() < 0.1


@skipif_unsupported_qtensor
@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256), (2, 2, 2, 256)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_q4k_decompress_matches_reference(shape, dtype):
    gen = torch.Generator().manual_seed(0)
    fp_tensor = torch.randn(shape, generator=gen).to(dtype)
    _check_ggml_decompress_matches_reference(QTensorTractQ4_K, fp_tensor)


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "weight",
    _ggml_invalid_weights(256)
    + [
        # f16 overflow of d (positive range) and dmin (negative offset)
        pytest.param(torch.full((2, 256), 1e9), id="d_overflow"),
        pytest.param(torch.full((2, 256), -1e9), id="dmin_overflow"),
        # nonzero super-block whose d and dmin round to 0 in f16
        pytest.param(torch.full((2, 256), 1e-10), id="underflow"),
    ],
)
def test_q4k_rejects_invalid_weights(weight):
    with pytest.raises(T2NErrorImpossibleQuantization):
        fp_to_tract_q4_k(weight)


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "version", ["0.22.1", "0.23.9", "0.23.10-pre", "0.23.10-rc.1"]
)
def test_q4k_rejects_old_tract(tmp_path, caplog, version):
    _check_ggml_rejects_old_tract(
        tmp_path, caplog, fp_to_tract_q4_k(torch.randn(2, 256)), version
    )


@skipif_unsupported_qtensor
@pytest.mark.parametrize("version", ["0.23.10", "0.23.11", "0.23.11-pre"])
def test_q4k_offload_state_write(tmp_path, version):
    _check_ggml_offload_state_write(
        tmp_path, fp_to_tract_q4_k(torch.randn(4, 512)), _tract_target(version)
    )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [
        i
        for i in TRACT_INFERENCES_TO_TESTS_EXACT
        if i.version >= QTensorTractQ4_K.MIN_TRACT_VERSION
    ],
)
# f16 CPU embedding (index_select) works on torch 1.x: like Q4_0/Q8_1 tests
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantize_with_tract_q4_k_embedding(inference_target, dtype):
    """Q4_K is only supported by tract on embedding/gathering."""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        model = nn.Embedding(6, 512).eval().to(dtype)
        q_tensor = fp_to_tract_q4_k(torch.randn(6, 512).to(dtype))
        assert q_tensor.decompress().dtype == dtype
        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={"weight": {"dtype": DatBinHeader.TractCustomTypes.Q4K}},
            ),
        )


@skipif_unsupported_qtensor
def test_compress_min_max_q4_k_embeddings_only_quantize_embeddings():
    model = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32))
    tied = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 10, bias=False))
    tied[1].weight = tied[0].weight
    registry = dynamic_load_registry(DEFAULT_COMPRESSION_REGISTRY)

    registry["min_max_q4_k_embeddings"](model)
    assert isinstance(model[0].weight, QTensorTractQ4_K)
    assert not isinstance(model[1].weight, QTensor)

    # weight shared with a Linear must stay float (no tract Q4_K matmul)
    registry["min_max_q4_k_embeddings"](tied)
    assert not isinstance(tied[0].weight, QTensor)

    with pytest.raises(T2NErrorMisuse):
        quantize_weights_min_max_Q4_K(
            nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32)),
            to_quantize_module_classes=(nn.Linear, nn.Embedding),
        )


# conv1d: linear (aka kernel=1)
# conv1d with various kernel size (3, 9)
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "kernel_size,inference_target",
    [(k, i) for i in TRACT_INFERENCES_TO_TESTS_APPROX_CONV for k in [1, 3, 9]],
)
def test_quantize_with_tract_q4_0_conv_base(kernel_size, inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        test_input = torch.arange(32 * kernel_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, 32, kernel_size)
        x = 32
        y = 2
        ker = (kernel_size,)
        model = nn.Conv1d(x, y, kernel_size=ker).eval()

        original_weight = (
            torch.arange(x * y * reduce(operator.mul, ker))
            .reshape(y, x, *ker)
            .float()
        )

        assert original_weight.shape == model.weight.shape
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            unit_test_naming=test_quantize_with_tract_q4_0_conv_base.__name__
            + f"_kernel{kernel_size}",
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


# conv1d with various in-channels size (32, 64, 128)
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "in_size,inference_target",
    [
        (in_size, i)
        for i in TRACT_INFERENCES_TO_TESTS_APPROX_CONV
        for in_size in [64, 128]
    ],
)
def test_quantize_with_tract_q4_0_conv_insize(in_size, inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        k = 3
        test_input = torch.arange(in_size * k).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, k)
        y = 2
        ker = (k,)
        model = nn.Conv1d(in_size, y, kernel_size=ker).eval()

        original_weight = (
            torch.arange(in_size * y * reduce(operator.mul, ker))
            .reshape(y, in_size, *ker)
            .float()
        )
        assert original_weight.shape == model.weight.shape

        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            unit_test_naming=test_quantize_with_tract_q4_0_conv_insize.__name__
            + f"{in_size}",
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


# conv1d with stride
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "stride,inference_target",
    [
        (stride, i)
        for i in TRACT_INFERENCES_TO_TESTS_APPROX_CONV
        for stride in [2, 3]
    ],
)
def test_quantize_with_tract_q4_0_conv_stride(stride, inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        k = 3
        in_size = 32
        n_iter_stride = 3
        chan_size = k * stride * n_iter_stride
        test_input = torch.arange(in_size * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, chan_size)
        y = 2
        ker = (k,)
        model = nn.Conv1d(in_size, y, kernel_size=ker, stride=(stride,)).eval()

        original_weight = (
            torch.arange(in_size * y * k).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            unit_test_naming=test_quantize_with_tract_q4_0_conv_stride.__name__
            + f"{stride}",
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


# conv1d with dilation
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "dilation,inference_target",
    [
        (dilation, i)
        for i in TRACT_INFERENCES_TO_TESTS_APPROX_CONV
        for dilation in [2, 4, 8]
    ],
)
def test_quantize_with_tract_q4_0_conv_dilation(dilation, inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        k = 3
        in_size = 32
        multiplier = 3
        chan_size = k * dilation * multiplier
        test_input = torch.arange(in_size * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, chan_size)
        y = 2
        ker = (k,)
        model = nn.Conv1d(
            in_size, y, kernel_size=ker, dilation=(dilation,)
        ).eval()

        original_weight = (
            torch.arange(in_size * y * k).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            unit_test_naming=test_quantize_with_tract_q4_0_conv_dilation.__name__
            + f"{dilation}",
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


# conv1d with groups
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "groups,inference_target",
    [
        (groups, i)
        for i in TRACT_INFERENCES_TO_TESTS_APPROX_CONV
        for groups in [2, 4]
    ],
)
def test_quantize_with_tract_q4_0_conv_groups(groups, inference_target):
    """Basic quantization values."""
    with torch.no_grad():
        k = 3
        y = 8
        in_size = 128
        multiplier = 4  # should be as big as biggest 'groups'
        chan_size = k * multiplier // groups
        test_input = torch.arange(in_size * chan_size).float()
        test_input[:3] = 3
        test_input = test_input.reshape(1, in_size, chan_size)
        ker = (k,)
        model = nn.Conv1d(in_size, y, kernel_size=ker, groups=groups).eval()

        original_weight = (
            torch.arange(in_size * y * k // groups).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            unit_test_naming=test_quantize_with_tract_q4_0_conv_groups.__name__
            + f"{groups}",
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )


# conv2d vanilla
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target", TRACT_INFERENCES_TO_TESTS_APPROX_CONV
)
def test_quantize_with_tract_q4_0_conv2d(inference_target):
    """Basic quantization values."""
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
        model = nn.Conv2d(in_size, y, kernel_size=ker).eval()

        original_weight = (
            torch.arange(in_size * y * k * k).reshape(y, -1, *ker).float()
        )

        assert original_weight.shape == model.weight.shape
        q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)

        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        model.to(torch.device("cpu", 0))  # goal to assign new device
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {
                        "dtype": DatBinHeader.TractCustomTypes.Q40
                        if inference_target.version >= "0.21.11"
                        else DatBinHeader.TractCustomTypes.Q40_LEGACY
                    }
                },
            ),
        )
