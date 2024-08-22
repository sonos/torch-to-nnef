import torch

from tests.utils import TRACT_INFERENCES_TO_TESTS, check_model_io_test


class MyModule(torch.nn.Module):
    CONST = 2

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(
            (torch.arange(self.CONST)).reshape(self.CONST, 1, 1).float()
        )

    def forward(self, x):
        return x * self.param


def test_issue_tract_mul_export():
    """Test issue mul not behaving as expected

    Should work starting with tract 0.21.6

    """
    full_shape = (2, MyModule.CONST, 17, 2)
    size = 1
    for fdim in full_shape:
        size *= fdim
    inp = torch.arange(size).reshape(full_shape).float()
    check_model_io_test(
        model=MyModule(),
        test_input=inp,
        inference_target=TRACT_INFERENCES_TO_TESTS[0],
    )
