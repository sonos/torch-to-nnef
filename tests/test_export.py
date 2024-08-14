"""E2E Test of the export function"""

import logging as log
import tempfile
import typing as T
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFInvalidArgument
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.tract import build_io


class MyDumbNN(nn.Module):
    def forward(self, x):
        return x * 2


class MultiTensorsIO(nn.Module):
    def forward(self, x, y):
        return x * y * 2, x / y


class MultiStructInput(nn.Module):
    def forward(self, x, y: T.Tuple[torch.Tensor, torch.Tensor]):
        return x * y[0] * 2


class MultiStructOutput(nn.Module):
    def forward(
        self, x
    ) -> T.Tuple[torch.Tensor, T.Tuple[torch.Tensor, torch.Tensor]]:
        return x * 2, (x * 3, x * 4)


def test_export_without_dot_nnef():
    """Test simple export"""
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
            check_same_io_as_tract=True,
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
            check_same_io_as_tract=True,
        )


def test_export_no_name_collision():
    """Test simple export"""
    _test_export_io_names(["a", "b"], ["c", "d"])


def test_export_inputs_name_collision():
    """Test simple export"""
    with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
        _test_export_io_names(["a", "a"], ["c", "d"])
    assert "Each str in input_names" in str(e_info.value)


def test_export_outputs_name_collision():
    """Test simple export"""
    with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
        _test_export_io_names(["a", "b"], ["c", "c"])
    assert "Each str in output_names" in str(e_info.value)


def test_export_io_name_collision():
    """Test simple export"""
    with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
        _test_export_io_names(["a", "b"], ["a", "c"])
    assert "input_names and output_names must be different" in str(e_info.value)


def test_export_wrong_inp_types():
    test_input = torch.rand(1, 2)
    model = MultiStructInput()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
            export_model_to_nnef(
                model=model,
                args=(test_input, (test_input, test_input)),
                file_path_export=export_path,
                input_names=["a", "tup"],
                output_names=["b"],
                log_level=log.INFO,
                check_same_io_as_tract=True,
            )
        assert (
            "Provided args[1] is of type <class 'tuple'> "
            "but only torch.Tensor is supported." in str(e_info.value)
        )


def test_export_wrong_out_types():
    test_input = torch.rand(1, 2)
    model = MultiStructOutput()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        model = model.eval()
        with pytest.raises(TorchToNNEFInvalidArgument) as e_info:
            export_model_to_nnef(
                model=model,
                args=(test_input,),
                file_path_export=export_path,
                input_names=["a"],
                output_names=["b", "tup"],
                log_level=log.INFO,
                check_same_io_as_tract=True,
            )
        assert (
            "Obtained model outputs[1] is of type <class 'tuple'> but only torch.Tensor is supported."
            in str(e_info.value)
        )
