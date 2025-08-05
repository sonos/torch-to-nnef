import typing as T
from copy import deepcopy
from functools import partial, reduce
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime
import operator

import numpy as np
import torch
from torch import nn
import pytest

from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant import (
    U8Compressor,
    qscale_per_group_f16_min_max_calibration,
    QTensorTractScaleOnly,
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.inference_target.tract import TractCheckTolerance, TractNNEF
from torch_to_nnef.tensor.quant.base import QTensor
from torch_to_nnef.utils import cd

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TRACT_INFERENCES_TO_TESTS_EXACT,
    check_model_io_test,
    skipif_unsupported_qtensor,
)


@skipif_unsupported_qtensor
def test_quantize_with_tract_q4_0_and_manipulate_tensor():
    original_weight = torch.arange(64).reshape(2, 32).float()
    q_tensor = fp_to_tract_q4_0_with_min_max_calibration(original_weight)
    q_tensor.u8_blob  # check access works
    new_q_tensor = q_tensor.to(torch.float32)
    new_q_tensor.u8_blob  # check access works

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


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_basic(inference_target):
    """basic quantization values"""
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
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_controled(inference_target):
    """basic quantization values"""
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
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_rounding2(inference_target):
    """basic quantization values"""
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
        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_arange(inference_target):
    """basic quantization values"""
    with torch.no_grad():
        if "arm" in platform.uname().machine.lower():
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
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version > "0.21.6"],
)
def test_quantize_with_tract_q4_0_assign_to(inference_target):
    """basic quantization values"""
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


def check_tensor_in_nnef_archive(
    inference_target: InferenceTarget,
    path: Path,
    labels: T.Union[T.List[str], T.Dict[str, T.Dict[str, T.Any]]],
):
    assert isinstance(inference_target, InferenceTarget)
    if not isinstance(inference_target, TractNNEF):
        return
    assert path.exists()
    exdir = path.parent / "extract"
    exdir.mkdir(parents=True, exist_ok=True)
    graph_filename = "graph.nnef"
    with cd(exdir):
        subprocess.check_call(["tar", "-xzf", str(path), graph_filename])
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
            raise ValueError(
                f"Some tensor where not found in exported NNEF archive: {remaining_labels}"
            )
        if isinstance(labels, dict):
            for label_name, label_opt_checks in labels.items():
                expected_dtype = label_opt_checks.get("dtype")
                if expected_dtype is not None:
                    dat_filename = f"{label_name}.dat"
                    subprocess.check_call(
                        ["tar", "-xzf", str(path), dat_filename]
                    )
                    bin_header = DatBinHeader.from_dat(dat_filename)
                    if bin_header.torch_dtype_or_custom != expected_dtype:
                        raise ValueError(
                            "wrong dtype in NNEF archive "
                            f"{label_name}: {bin_header.torch_dtype_or_custom} but expected {expected_dtype}"
                        )


@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_EXACT if _.version >= "0.21.11"],
)
def test_quantize_with_tract_q4_0_embedding(inference_target):
    """basic quantization values"""
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
        )


# conv1d: linear (aka kernel=1)
# conv1d with various kernel size (3, 9)
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "kernel_size,inference_target",
    [(k, i) for i in TRACT_INFERENCES_TO_TESTS_APPROX_CONV for k in [1, 3, 9]],
)
def test_quantize_with_tract_q4_0_conv_base(kernel_size, inference_target):
    """basic quantization values"""
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
    """basic quantization values"""
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
    """basic quantization values"""
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
    """basic quantization values"""
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
    """basic quantization values"""
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
        )


# conv2d vanilla
@skipif_unsupported_qtensor
@pytest.mark.parametrize(
    "inference_target", TRACT_INFERENCES_TO_TESTS_APPROX_CONV
)
def test_quantize_with_tract_q4_0_conv2d(inference_target):
    """basic quantization values"""
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
        )
