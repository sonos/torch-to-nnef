""" E2E Test of the export function """

import logging as log
import tempfile
from pathlib import Path

import torch
from torch import nn

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.tract import build_io


class MyDumbNN(nn.Module):
    def forward(self, x):
        return x * 2


def test_export_without_dot_nnef():
    """Test simple export"""
    test_input = torch.rand(1, 2)
    model = MyDumbNN()
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model"
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
