import torch
from torch import nn


class MyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom op; the handler maps it to NNEF `relu`.
        return torch.ops.t2n_extra.my_relu(x)
