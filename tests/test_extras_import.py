from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_to_nnef import KhronosNNEF, TractNNEF, export_model_to_nnef


class _UnitRelu(nn.Module):
    def forward(self, x):
        return torch.ops.t2n_extra.unit_relu(x)


@pytest.mark.parametrize(
    "inference_target",
    [
        KhronosNNEF.latest(),
        TractNNEF(TractNNEF.latest_version(), check_io=False),
    ],
)
def test_load_extra_op_modules_import(inference_target, tmp_path):
    model = _UnitRelu().eval()
    x = torch.randn(2, 3)
    out = export_model_to_nnef(
        model,
        args=(x,),
        file_path_export=tmp_path / "mod.nnef",
        inference_target=inference_target,
        compression_level=0,
        input_names=["x"],
        output_names=["y"],
        load_extra_op_modules=["tests.plugins.unit_handlers"],
    )
    assert out.exists()


@pytest.mark.parametrize(
    "inference_target",
    [
        KhronosNNEF.latest(),
        TractNNEF(TractNNEF.latest_version(), check_io=False),
    ],
)
def test_env_auto_import(inference_target, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "TORCH_TO_NNEF_EXTRA_MODULES", "tests.plugins.unit_handlers"
    )
    model = _UnitRelu().eval()
    x = torch.randn(2, 3)
    out = export_model_to_nnef(
        model,
        args=(x,),
        file_path_export=tmp_path / "mod.nnef",
        inference_target=inference_target,
        compression_level=0,
        input_names=["x"],
        output_names=["y"],
    )
    assert out.exists()
