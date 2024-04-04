import logging as log
from pathlib import Path

import torch
from nnef_tools.io.nnef.writer import tempfile
from torch import nn
from torch.nn import functional as f
from torch.nn import init

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.tract import build_io


class SqueezeExcitationBlock1d(nn.Module):
    def __init__(
        self, nb_input_channels: int, compression_ratio: int = 8, dim: int = -2
    ):
        super().__init__()
        # Make sure the dimension is given in a backwards counting manner
        assert dim in [
            -1,
            -2,
        ], "The feature dim can only be among the last 2 dimensions"
        self.__remove_dim = -2 if dim == -1 else -1
        # Compute the number of channels after compression
        reduced_channels = max(1, int(nb_input_channels / compression_ratio))
        if reduced_channels == 1:
            print(
                "!! Warning !! Compressing channels down to 1 in "
                "squeeze & excitation block"
            )
        # Initialize the 2 linear layers of the squeeze operation
        self.linear_down = nn.Conv2d(nb_input_channels, reduced_channels, 1)
        self.linear_up = nn.Conv2d(reduced_channels, nb_input_channels, 1)
        # Initialize the weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        """
        GlobalPooling -> bottleneck (FC-ReLU-FC) -> sigmoid -> multiply with FMs
        """
        # Compute kernel to perform average pooling along the selected dim only
        pooling_kernel = [x.size(2), x.size(3)]
        pooling_kernel[self.__remove_dim] = 1
        # Average pool globally along 1 dim only
        x_branch = f.avg_pool2d(x, pooling_kernel, stride=1)
        # Squeeze operation
        x_branch = torch.sigmoid(
            self.linear_up(torch.relu(self.linear_down(x_branch)))
        )
        # Gating and return
        return torch.mul(x, x_branch)

    def _initialize_weights(self):
        init.kaiming_normal_(self.linear_down.weight)
        init.constant_(self.linear_down.bias, 1)
        init.xavier_normal_(self.linear_up.weight)
        init.constant_(self.linear_up.bias, 2)


def test_export():
    """Test simple export"""
    test_input = torch.rand(1, 2, 10, 20)
    model = SqueezeExcitationBlock1d(nb_input_channels=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        model = model.eval()

        input_names, output_names = build_io(
            model, test_input, io_npz_path=io_npz_path
        )
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=input_names,
            output_names=output_names,
            log_level=log.INFO,
            check_same_io_as_tract=True,
        )
