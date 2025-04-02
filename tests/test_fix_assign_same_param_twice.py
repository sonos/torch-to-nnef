from copy import deepcopy
import tempfile
from pathlib import Path
import subprocess

import torch

from torch_to_nnef.utils import cd
from torch_to_nnef.compress import dynamic_load_registry

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


# Check no duplicates of weights
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(1, 32, bias=False)
        self.lin1 = torch.nn.Linear(32, 32, bias=False)
        self.lin2 = torch.nn.Linear(32, 32, bias=False)

    def forward(self, x):
        x = self.lin0(x)
        x = torch.relu(x)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        return x


def check_no_dup_dat(inference_target, path):
    assert path.exists()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with cd(td):
            subprocess.check_output(["tar", "-xzf", str(path)])
            dats = [_ for _ in td.iterdir() if ".dat" in _.suffixes]
            if len(dats) != 2:
                names = [_.name for _ in dats]
                raise ValueError(f"too much .dat produced: {names}")


def test_issue_dup_if_shared_tensor_export():
    """Test issue with duplicate tensor."""
    latest_tract_inference = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
    latest_tract_inference.dynamic_axes = {
        "input_0": {0: "S"},
    }
    mod = MyModule()
    mod.lin1.weight = mod.lin2.weight
    check_model_io_test(
        model=mod,
        test_input=torch.rand(10, 1),
        inference_target=latest_tract_inference,
        callback=check_no_dup_dat,
    )


def test_issue_dup_compress_if_shared_tensor_export():
    """Test issue with duplicate tensor."""
    latest_tract_inference = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
    latest_tract_inference.check_io = False
    latest_tract_inference.dynamic_axes = {
        "input_0": {1: "S"},
    }
    registry = dynamic_load_registry(
        "torch_to_nnef.compress.DEFAULT_COMPRESSION"
    )
    for k, fn in registry.items():
        mod = MyModule()
        mod.lin1.weight = mod.lin2.weight
        mod = fn(mod)
        check_model_io_test(
            model=mod,
            test_input=torch.rand(10, 1),
            inference_target=latest_tract_inference,
            callback=check_no_dup_dat,
        )
