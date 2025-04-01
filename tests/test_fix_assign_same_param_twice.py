from copy import deepcopy
import tempfile
from pathlib import Path
import subprocess

import torch

from torch_to_nnef.utils import cd

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


# Check no duplicates of weights
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, 1, bias=False)
        self.conv1 = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
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
        "input_0": {2: "S"},
    }
    mod = MyModule()
    mod.conv1.weight = mod.conv2.weight
    check_model_io_test(
        model=mod,
        test_input=torch.rand(1, 10, 1000),
        inference_target=latest_tract_inference,
        callback=check_no_dup_dat,
    )
