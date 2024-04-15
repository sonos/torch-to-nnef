import math

import torch
from torch import nn

from .utils import check_model_io_test


class RelPosEncXL(nn.Module):
    """Original SpeechBrain Positional encoding"""

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


class FixedRelPosEncXL(nn.Module):
    """Equivalent Export friendly to SpeechBrain Positional encoding"""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        inv_freq = torch.exp(
            torch.arange(0, self.emb_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.emb_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.inv_freq_shape_0 = self.inv_freq.shape[0]

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        with torch.no_grad():
            positions = (
                torch.arange(0, seq_len, dtype=x.dtype, device=x.device)
                .to(x)
                .unsqueeze(1)
                .repeat(1, self.inv_freq_shape_0)  # tract friendly
            )
            rev_positions = (
                torch.arange(
                    seq_len - 1, -1, step=-1, dtype=x.dtype, device=x.device
                )
                .to(x)
                .unsqueeze(1)
                .repeat(1, self.inv_freq_shape_0)  # tract friendly
            )

            rev_inv_freq = rev_positions * self.inv_freq
            pos_inv_freq = positions * self.inv_freq

            pe_past = (
                torch.stack(
                    (torch.sin(rev_inv_freq), torch.cos(rev_inv_freq)), dim=1
                )
                .permute(0, 2, 1)
                .reshape(1, seq_len, -1)
            )
            pe_future = (
                torch.stack(
                    (torch.sin(pos_inv_freq), torch.cos(-pos_inv_freq)), dim=1
                )
                .permute(0, 2, 1)
                .reshape(1, seq_len, -1)
            )

            pe = torch.cat([pe_past, pe_future[:, 1:]], dim=1)
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


# def test_export_assign_slice():
#     """Test simple models"""
#     check_model_io_test(
#         model=AssignSliceIssue(3),
#         test_input=torch.arange(10).float().reshape(2, 5),
#         # without dyn axes assume constant shape inputs so
#         # positions and pe_future are static value tensors
#         # dynamic_axes={"input_0": {1: "S"}},
#     )


def test_export_assign_slice():
    """Test simple models"""
    check_model_io_test(
        model=FixedRelPosEncXL(3),
        test_input=torch.arange(10).float().reshape(2, 5),
        # without dyn axes assume constant shape inputs so
        # positions and pe_future are static value tensors
        dynamic_axes={"input_0": {1: "S"}},
    )


def test_export_check_equivalent_pos_encoding():
    original_encoder = RelPosEncXL(100)
    export_friendly_encoder = FixedRelPosEncXL(100)
    test_inputs = torch.arange(10).float().reshape(2, 5)

    ref_res = original_encoder(test_inputs)
    new_res = export_friendly_encoder(test_inputs)
    assert (ref_res == new_res).all()
