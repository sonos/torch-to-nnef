"""E2E Test of the export function"""

import logging as log
import tempfile
import typing as T
from dataclasses import dataclass
from pathlib import Path

import nnef
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tests.utils import INFERENCE_TARGETS_TO_TESTS, skipif_unsupported_qtensor
from torch_to_nnef.exceptions import TorchToNNEFInvalidArgument
from torch_to_nnef.export import (
    export_model_to_nnef,
    export_tensors_from_disk_to_nnef,
    export_tensors_to_nnef,
)
from torch_to_nnef.inference_target.tract import build_io
from torch_to_nnef.tensor.quant import (
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.utils import torch_version


class MyDumbNN(nn.Module):
    def forward(self, x):
        return x * 2


class MultiTensorsIO(nn.Module):
    def forward(self, x, y):
        return x * y * 2, x / y


class MultiStructInputs(nn.Module):
    def forward(self, x, y: T.Tuple[torch.Tensor, torch.Tensor]):
        return x * y[0] * 2 * y[1]


class MultiStructOutputs(nn.Module):
    def forward(
        self, x
    ) -> T.Tuple[torch.Tensor, T.Tuple[torch.Tensor, torch.Tensor]]:
        return x * 2, (x * 3, x * 4)


@dataclass()
class FakeConfig:
    scale: float = 2.1


class MultiObjInputs(nn.Module):
    def forward(self, x, fake_config: FakeConfig):
        return x * 2 * fake_config.scale


class MultiDictInputs(nn.Module):
    def forward(self, x, y):
        res = x * 2 * y["a"] * y["b"]
        return res


class MultiDeepObjInputs(nn.Module):
    def forward(self, x, y):
        res = x * 2 * y["a"] * y["b"].scale
        return res


class MultiDictOutputs(nn.Module):
    def forward(self, x):
        return {
            "x2": x * 2,
            "x3": x * 3,
        }


class MultiInputsPrimitives(nn.Module):
    def forward(self, x, is_action_a: bool, n_loop: int):
        if is_action_a:
            for _ in range(n_loop):
                x = x**2
        else:
            x = x**2
        return x


def test_export_base():
    test_input = torch.rand(1, 2)
    model = MyDumbNN()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        model = model.eval()

        input_names, output_names = build_io(
            model, test_input, io_npz_path=io_npz_path
        )
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=input_names,
            output_names=output_names,
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def _test_export_io_names(input_names, output_names):
    test_input = torch.rand(1, 2)
    model = MultiTensorsIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input, test_input),
            file_path_export=export_path,
            input_names=input_names,
            output_names=output_names,
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_export_no_name_collision():
    _test_export_io_names(["a", "b"], ["c", "d"])


def test_export_inputs_name_collision():
    with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
        _test_export_io_names(["a", "a"], ["c", "d"])
    assert "Each str in input_names" in str(e_info.value)


def test_export_outputs_name_collision():
    with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
        _test_export_io_names(["a", "b"], ["c", "c"])
    assert "Each str in output_names" in str(e_info.value)


def test_export_io_name_collision():
    with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
        _test_export_io_names(["a", "b"], ["a", "c"])
    assert "input_names and output_names must be different" in str(e_info.value)


def test_export_tuple_inp_types():
    test_input = torch.rand(1, 2)
    model = MultiStructInputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input, (test_input, test_input)),
            file_path_export=export_path,
            input_names=["a", "tup"],
            output_names=["b"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_export_tuple_out_types():
    test_input = torch.rand(1, 2)
    model = MultiStructOutputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input,),
            file_path_export=export_path,
            input_names=["a"],
            output_names=["b", "tup"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_export_obj_inp_types():
    test_input = torch.rand(1, 2)
    model = MultiObjInputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input, FakeConfig()),
            file_path_export=export_path,
            input_names=["a", "conf"],
            output_names=["b"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_multi_deep_obj_inputs():
    test_input = torch.rand(1, 2)
    model = MultiDeepObjInputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input, {"a": torch.rand(1, 2), "b": FakeConfig()}),
            file_path_export=export_path,
            input_names=["a", "dic"],
            output_names=["b"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_multi_dict_inputs():
    test_input = torch.rand(1, 2)
    model = MultiDictInputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input, {"a": torch.rand(1, 2), "b": torch.rand(1, 2)}),
            file_path_export=export_path,
            input_names=["a", "dic"],
            output_names=["b"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_multi_dict_outputs():
    test_input = torch.rand(1, 2)
    model = MultiDictOutputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=["a"],
            output_names=["dic"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_primitives():
    test_input = torch.rand(1, 2)
    model = MultiInputsPrimitives()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        export_model_to_nnef(
            model=model,
            args=(test_input, True, 2),
            file_path_export=export_path,
            input_names=["a", "b", "c"],
            output_names=["d"],
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


def test_without_names():
    test_input = torch.rand(1, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = MultiInputsPrimitives()
        model = model.eval()

        export_model_to_nnef(
            model=model,
            args=(test_input, True, 2),
            file_path_export=export_path,
            log_level=log.INFO,
            inference_target=INFERENCE_TARGETS_TO_TESTS[0],
        )


_SAMPLES = {
    "alpha": torch.rand((10, 20)),
    "beta": torch.rand((12, 24)).bool(),
    "gamma": torch.rand((12, 24)).to(torch.float16),
}


def _test_export_tensors_base(fn):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fn(td, _SAMPLES)
        for path in td.iterdir():
            if not path.name.endswith(".dat"):
                continue
            with path.open("rb") as fh:
                ntensor = nnef.read_tensor(fh)
            variable_name = path.with_suffix("").name
            assert np.allclose(_SAMPLES[variable_name].numpy(), ntensor)


def test_export_tensors_to_nnef_basic():
    def fn(td, samples):
        export_tensors_to_nnef(samples, output_dir=td)

    _test_export_tensors_base(fn)


@skipif_unsupported_qtensor
def test_export_tensors_to_nnef_qtensor():
    samples = {
        "qalpha": fp_to_tract_q4_0_with_min_max_calibration(torch.rand((2, 32)))
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        export_tensors_to_nnef(samples, output_dir=td)
        for path in td.iterdir():
            try:
                with path.open("rb") as fh:
                    _ = nnef.read_tensor(fh)
            except ValueError as exp:
                assert "unsupported combination of item type" in exp.args[0]

            variable_name = path.with_suffix("").name
            assert variable_name in samples


def test_export_tensors_to_nnef_from_disk():
    tensor_name_to_export = ["alpha", "beta"]

    def fn_check_found_tensors(to_export):
        not_found = set(tensor_name_to_export).difference(to_export.keys())
        if len(not_found) > 0:
            raise ValueError(f"missing keys in provided file: {not_found}")
        return True

    def fn(td, samples):
        serial_fp = td / "s.pt"
        torch.save(samples, serial_fp)
        export_tensors_from_disk_to_nnef(
            serial_fp,
            output_dir=td,
            filter_key=lambda x: x in tensor_name_to_export,
            fn_check_found_tensors=fn_check_found_tensors,
        )

    _test_export_tensors_base(fn)


def test_export_tensors_to_nnef_from_safetensors():
    tensor_name_to_export = ["alpha", "beta"]

    def fn_check_found_tensors(to_export):
        not_found = set(tensor_name_to_export).difference(to_export.keys())
        if len(not_found) > 0:
            raise ValueError(f"missing keys in provided file: {not_found}")
        return True

    def fn(td, samples):
        serial_fp = td / "model.safetensors"
        save_file(samples, serial_fp)
        export_tensors_from_disk_to_nnef(
            serial_fp,
            output_dir=td,
            filter_key=lambda x: x in tensor_name_to_export,
            fn_check_found_tensors=fn_check_found_tensors,
        )

    _test_export_tensors_base(fn)
