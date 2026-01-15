import torch
from torch import nn

from torch_to_nnef import TractNNEF

from .utils import check_model_io_test

SEQLEN = 10
BATCH = 4
INPUTS = 3


class Cat0(nn.Module):
    """Concatenate a zero-sized tensor on axis 1.

    This currently causes Tract to reject the model,
    but semantically should be a no-op.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(
            [x.size(0), 0, x.size(2)],
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        return torch.cat((x, c), dim=1)


class Cat0Cond(nn.Module):
    """Conditionally concatenate zeros on axis 1.

    lg=0 is an explicit identity.
    """

    def __init__(self, lg: int):
        super().__init__()
        self.lg = lg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(
            [x.size(0), self.lg, x.size(2)],
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        if self.lg > 0:
            x = torch.cat((x, c), dim=1)
        return x


def _run(model: nn.Module, x: torch.Tensor) -> None:
    """Canonical torch-to-nnef + Tract IO check."""
    tract_target = TractNNEF(
        version="0.21.13",
        check_io=True,
        dynamic_axes={"input": {0: "B", 1: "S"}},
    )

    check_model_io_test(
        model=model,
        test_input=x,
        input_names=["input"],
        output_names=["output"],
        inference_target=tract_target,
        # nnef_variable_naming_scheme="numeric",
        # compression_level=0,
    )


def test_concat_positive_length_is_ok():
    """Concatenating a non-zero-length tensor must work."""
    torch.manual_seed(0)
    x = torch.rand(BATCH, SEQLEN, INPUTS)
    model = Cat0Cond(lg=1).eval()

    _run(model, x)


def test_concat_zero_length_tensor_is_noop():
    """Concat with a zero-sized tensor should behave as identity."""
    torch.manual_seed(0)
    x = torch.rand(BATCH, SEQLEN, INPUTS)
    model = Cat0().eval()

    _run(model, x)


# NOTE: Explicit identity (lg=0) in a full model IO is prevented
# by default in torch to nnef
# but will be possible in 0.22 when PR: https://github.com/sonos/torch-to-nnef/pull/5/changes#diff-c936c200fdea32069d48b31cc2e1d38c09aa959906f76e89c685c6185bb2dba6R59
# is merged (to support nemo).
# This is typically a bad idea to move a tensor arround
# when nothing is changed, has it may lead to performance degradation,
# in inference engine that could move the data instead of
# just reusing the same buffer.
