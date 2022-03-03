"""Tests simple models."""

from pathlib import Path
import subprocess
import tempfile
import pytest

import numpy as np
import torch
from torch import nn
from torch_to_nnef.export import export_model_to_nnef


INPUT_AND_MODELS = [
    (torch.rand(1, 10, 100), nn.Sequential(nn.Conv1d(10, 20, 3))),
    (torch.rand(1, 10, 100), nn.Sequential(nn.ReLU())),
]

# profile
# f"tract {nnef_path} -i '{shape_str},{dtype}' -f nnef -O dump --profile",


def tract_assert_io(nnef_path: Path, io_npz_path: Path):
    cmd = f"tract {nnef_path} --input-bundle {io_npz_path} -O run --assert-output-bundle {io_npz_path}"
    try:
        subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(cmd)
        import ipdb

        ipdb.set_trace()
        return False


def test_should_fail_since_false_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_input = torch.rand(1, 10, 100)
        model = nn.Sequential(nn.Conv1d(10, 20, 3))
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        test_output = model(test_input)
        export_model_to_nnef(
            model=model,
            args=test_input,
            base_path=export_path,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )

        np.savez(
            io_npz_path,
            input=test_input.detach().numpy(),
            output=test_output.detach().numpy()
            + 1,  # <-- here we artificially add 1
        )
        assert not tract_assert_io(
            export_path.with_suffix(".nnef.tgz"), io_npz_path
        ), f"SHOULD fail tract io check with {model}"


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_model_export(test_input, model):
    """Test simple models"""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        test_output = model(test_input)
        export_model_to_nnef(
            model=model,
            args=test_input,
            base_path=export_path,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )

        np.savez(
            io_npz_path,
            input=test_input.detach().numpy(),
            output=test_output.detach().numpy(),
        )
        assert tract_assert_io(
            export_path.with_suffix(".nnef.tgz"), io_npz_path
        ), f"failed tract io check with {model}"
