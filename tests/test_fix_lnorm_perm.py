from copy import deepcopy

import torch

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


# add unit test for https://github.com/{project}/issues/18
# export was fine but tract failed to find that -1 is 80 in reality
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 3)
        self.lnorm = torch.nn.LayerNorm(10)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 1, 0)
        return self.lnorm(x)


def check_align_with_reserialized(inference_target, path):
    assert path.exists()
    reser_path = path.parent / "reserialize_model.nnef.tgz"
    inference_target.tract_cli.run(
        [
            str(path),
            "--nnef-tract-core",
            "dump",
            "--quiet",
            "--nnef",
            str(reser_path),
        ]
    )

    exp_out = path.parent / "expected_out.npz"
    inference_target.tract_cli.run(
        [
            str(path),
            "--nnef-tract-core",
            "-O",
            "run",
            "--input-from-npz",
            str(path.parent / "io.npz"),
            "--save-outputs-npz",
            str(exp_out),
        ]
    )

    inference_target.tract_cli.run(
        [
            str(reser_path),
            "--nnef-tract-core",
            "-O",
            "run",
            "--input-from-npz",
            str(path.parent / "io.npz"),
            "--assert-output-bundle",
            str(exp_out),
        ]
    )


def test_issue_lnorm_export():
    """Test issue tract with Permute+LayerNorm then deser->ser->deser."""
    latest_tract_inference = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
    latest_tract_inference.dynamic_axes = {
        "input_0": {2: "S"},
    }
    check_model_io_test(
        model=MyModule(),
        test_input=torch.rand(1, 10, 1000),
        inference_target=latest_tract_inference,
        callback=check_align_with_reserialized,
    )
