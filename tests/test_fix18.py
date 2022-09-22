import pytest
import torch

from torch_to_nnef.tract import tract_version
from torch_to_nnef.utils import SemanticVersion

from .utils import _test_check_model_io


# add unit test for https://github.com/{project}/issues/18
# export was fine but tract failed to find that -1 is 80 in reality
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 3)
        self.linear = torch.nn.Linear(80, 10)

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)
        batch_size = x.shape[0]
        width = x.shape[3]
        x = x.transpose(1, 3).reshape([batch_size, width, -1])
        return self.linear(x)


@pytest.mark.skipif(
    tract_version() < SemanticVersion.from_str("0.18.0"),
    reason="tract version installed too old",
)
def test_issue18_export():
    """Test issue 18.

    Should work starting with tract 0.18.0

    """
    _test_check_model_io(
        model=MyModule(),
        test_input=torch.rand(1, 1000, 10),
        dynamic_axes={"input_0": {1: "S"}, "output_0": {1: "S"}},
    )
