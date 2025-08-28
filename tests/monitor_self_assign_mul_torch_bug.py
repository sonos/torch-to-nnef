"""Monitor failure of torch JIT regarding self assign mul."""

import typing

import torch

from tests.utils import check_model_io_test


class NetWrapper(torch.nn.Module):
    def __init__(
        self,
        alpha: typing.Optional[float] = None,
    ):
        super().__init__()
        self.alpha = alpha

    def forward(self, inp):
        if self.alpha is not None:
            inp *= self.alpha
        return inp


if __name__ == "__main__":
    check_model_io_test(
        model=NetWrapper(3.0), test_input=torch.arange(4).reshape(2, 2).float()
    )
