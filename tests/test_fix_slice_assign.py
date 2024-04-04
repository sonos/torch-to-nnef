import math

import torch
from torch import nn

from .utils import check_model_io_test


class RelPosEncXL(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

        inv_freq = torch.exp(
            torch.arange(0, self.emb_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.emb_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        with torch.no_grad():
            tot_pe = torch.zeros((2, seq_len, self.emb_dim), dtype=x.dtype).to(
                x
            )
            pe_past = tot_pe[0]
            pe_future = tot_pe[1]
            positions = (
                torch.arange(0, seq_len, dtype=x.dtype, device=x.device)
                .to(x)
                .unsqueeze(-1)
            )
            rev_positions = (
                torch.arange(
                    seq_len - 1, -1, step=-1, dtype=x.dtype, device=x.device
                )
                .to(x)
                .unsqueeze(-1)
            )

            pe_past[:, 0::2] = torch.sin(rev_positions * self.inv_freq)
            pe_past[:, 1::2] = torch.cos(rev_positions * self.inv_freq)
            pe_future[:, 0::2] = torch.sin(positions * self.inv_freq)
            pe_future[:, 1::2] = torch.cos(-positions * self.inv_freq)

            pe_past = pe_past.unsqueeze(0)
            pe_future = pe_future[1:].unsqueeze(0)
            pe = torch.cat([pe_past, pe_future], dim=1)
            # pe is now 1, 2*seq_len, embed_dim
            return pe


class AssignSliceIssue(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        with torch.no_grad():
            pe_future = torch.zeros((seq_len, self.emb_dim), dtype=x.dtype)
            positions = (
                torch.arange(0, seq_len, dtype=x.dtype, device=x.device)
                .to(x)
                .unsqueeze(-1)
            )
            pe_future[:, 0::2] = torch.sin(positions)
            return pe_future


def test_export_assign_slice():
    """Test simple models"""
    check_model_io_test(
        model=AssignSliceIssue(3),
        test_input=torch.arange(10).float().reshape(2, 5),
        # without dyn axes assume constant shape inputs so
        # positions and pe_future are static value tensors
        # dynamic_axes={"input_0": {1: "S"}},
    )
