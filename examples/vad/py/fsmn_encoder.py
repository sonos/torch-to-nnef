"""Standalone FSMN encoder for FSMN-VAD.

Copied from FunASR (funasr/models/fsmn_vad_streaming/encoder.py) and stripped of
the funasr.register dependency so it can be imported without pulling the full
FunASR stack. Only the non-streaming forward path is kept; streaming cache dicts
are replaced by an optional explicit cache argument to keep tracing friendly.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class AffineTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class RectifiedLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class FSMNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, rorder=None, lstride=1, rstride=1):
        super().__init__()
        self.dim = input_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.conv_left = nn.Conv2d(
            self.dim, self.dim, [lorder, 1], dilation=[lstride, 1], groups=self.dim, bias=False
        )
        if self.rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim, self.dim, [rorder, 1], dilation=[rstride, 1], groups=self.dim, bias=False
            )
        else:
            self.conv_right = None

    def forward(self, x, cache: Optional[torch.Tensor] = None):
        x = torch.unsqueeze(x, 1)
        x_per = x.permute(0, 3, 2, 1)
        if cache is not None:
            y_left = torch.cat((cache, x_per), dim=2)
        else:
            y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])
        y_left = self.conv_left(y_left)
        out = x_per + y_left
        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            y_right = self.conv_right(y_right)
            out = out + y_right
        out_per = out.permute(0, 3, 2, 1)
        return out_per.squeeze(1)


class BasicBlock(nn.Module):
    def __init__(self, linear_dim, proj_dim, lorder, rorder, lstride, rstride, stack_layer):
        super().__init__()
        self.stack_layer = stack_layer
        self.linear = LinearTransform(linear_dim, proj_dim)
        self.fsmn_block = FSMNBlock(proj_dim, proj_dim, lorder, rorder, lstride, rstride)
        self.affine = AffineTransform(proj_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.fsmn_block(x1, None)
        x3 = self.affine(x2)
        return self.relu(x3)


class FsmnStack(nn.Sequential):
    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class FSMN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_affine_dim: int,
        fsmn_layers: int,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride: int,
        rstride: int,
        output_affine_dim: int,
        output_dim: int,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(
            *[
                BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i)
                for i in range(fsmn_layers)
            ]
        )
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_linear1(x)
        x = self.in_linear2(x)
        x = self.relu(x)
        x = self.fsmn(x)
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x
