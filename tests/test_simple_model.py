"""Tests simple models."""

from pathlib import Path
import subprocess
import tempfile
import typing
import pytest
import torch
from torch import nn
from torch_to_nnef.export import export_model_to_nnef


INPUT_AND_MODELS = [
    (torch.rand(1, 10, 100), nn.Sequential(nn.Conv1d(10, 20, 3))),
]


def tract_call(nnef_path: Path, shape: typing.Tuple[int, ...], dtype="f32"):
    shape_str = ','.join(map(str, shape))
    return subprocess.check_call(
        f"tract {nnef_path} -i '{shape_str},{dtype}' -f nnef -O dump --profile",
        shell=True,
    )


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_model_export(test_input, model):
    """Test simple models"""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        export_model_to_nnef(
            model=model,
            args=test_input,
            base_path=export_path,
            input_names=["input"],
            output_names=["output"],
        )

        print(
            export_path,
            tract_call(
                export_path.with_suffix(".nnef.tgz"),
                shape=test_input.shape,
            ),
        )
