"""A patch to ShiftedWindowAttention so that jit pass it correctly.

Indeed in torchvision 0.13.1 : ShiftedWindowAttention
call shifted_window_attention in forward pass
but part of the fn is assignation to mask as such:
```
attn_mask = x.new_zeros((pad_H, pad_W))
h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
count = 0
for h in h_slices:
    for w in w_slices:
        attn_mask[h[0] : h[1], w[0] : w[1]] = count
        count += 1
```
This kind of assignation ca not be transated to jit correctly.

Thanksfully: window_size and shift_size are fixed per ShiftedWindowAttention
layer.
So the mask computation can happen before hand at layer initialisation.
Doing so original mask compute for more memory (which may have been the origin
of this decision, since at training time GPU mem are often limited.)

"""

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MISSING_SWIN = False
try:
    from torchvision.models.swin_transformer import SwinTransformerBlock
except ImportError:
    MISSING_SWIN = True
    print("swin_transformer not found in torchvision.")


def shifted_window_attention(
    inp: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    num_windows: Optional[int] = None,
    input_shape: List[int] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        inp (Tensor[N, H, W, C]): The inp tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input_shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(inp, (0, 0, 0, pad_r, 0, pad_b))  # issue here
    _, pad_H, pad_W, _ = x.shape

    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    x = x.view(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        B * num_windows, window_size[0] * window_size[1], C
    )  # B*nW, Ws*Ws, C
    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(
        x.size(0), x.size(1), 3, num_heads, C // num_heads
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # Add relative position bias
    attn = attn + relative_position_bias
    # return attn # This work upt to this point
    if sum(shift_size) > 0:
        if attn_mask is None:
            raise ValueError("missing attn_mask with shift_size")
        # generate attention mask
        attn = attn.view(
            x.size(0) // num_windows,
            num_windows,
            num_heads,
            x.size(1),
            x.size(1),
        )
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    if attention_dropout > 0:
        # WARNING this will NOT be desactivated at eval time
        attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    if dropout > 0:
        # WARNING this will NOT be desactivated at eval time
        x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


class ExportableShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += (
            self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(
            -1
        )  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.attn_mask = None
        self.num_windows = None
        self.input_shape = None

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def pre_compute_attn_mask(self, inp):
        # generate attention mask
        # B, H, W, C
        _, H, W, _ = inp.shape
        pad_r = (
            self.window_size[1] - W % self.window_size[1]
        ) % self.window_size[1]
        pad_b = (
            self.window_size[0] - H % self.window_size[0]
        ) % self.window_size[0]
        x = F.pad(inp, (0, 0, 0, pad_r, 0, pad_b))  # issue here
        _, pad_H, pad_W, _ = x.shape
        # no more need for x after that

        num_windows = (pad_H // self.window_size[0]) * (
            pad_W // self.window_size[1]
        )
        attn_mask = None
        if self.attn_mask is None and sum(self.shift_size) > 0:
            attn_mask = torch.zeros((pad_H, pad_W))
            h_slices = (
                (0, -self.window_size[0]),
                (-self.window_size[0], -self.shift_size[0]),
                (-self.shift_size[0], None),
            )
            w_slices = (
                (0, -self.window_size[1]),
                (-self.window_size[1], -self.shift_size[1]),
                (-self.shift_size[1], None),
            )
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(
                pad_H // self.window_size[0],
                self.window_size[0],
                pad_W // self.window_size[1],
                self.window_size[1],
            )
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
                num_windows, self.window_size[0] * self.window_size[1]
            )
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        return num_windows, attn_mask

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as inp, i.e. [B, H, W, C]
        """
        if self.num_windows is None:
            self.num_windows, self.attn_mask = self.pre_compute_attn_mask(x)
        if self.input_shape is None:
            self.input_shape = x.shape

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index
        ]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = (
            relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        )

        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            attn_mask=self.attn_mask,
            num_windows=self.num_windows,
            input_shape=self.input_shape,
        )


if not MISSING_SWIN:

    class ExportableSwinTransformerBlock(SwinTransformerBlock):
        """Important to overwrite as well due attn_layer default"""

        # pylint: disable-next=useless-parent-delegation
        def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: List[int],
            shift_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.0,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_layer: Callable[
                ..., nn.Module
            ] = ExportableShiftedWindowAttention,
        ):
            super().__init__(
                dim,
                num_heads,
                window_size,
                shift_size,
                mlp_ratio,
                dropout,
                attention_dropout,
                stochastic_depth_prob,
                norm_layer,
                attn_layer,
            )
