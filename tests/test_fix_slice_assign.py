import math

import torch
from torch import nn

from torch_to_nnef.tract import tract_version

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


class FastFixedRelPosEncXL(nn.Module):
    """Equivalent Export friendly to SpeechBrain Positional encoding

    Attempt to make implementation faster
    """

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
            positions = torch.arange(0, seq_len, dtype=x.dtype, device=x.device)
            rev_positions = torch.arange(
                seq_len - 1, -1, step=-1, dtype=x.dtype, device=x.device
            )
            rep_jpos = (
                torch.vstack([rev_positions, positions])
                .to(x)
                .unsqueeze(2)
                .repeat(1, 1, self.inv_freq_shape_0)
            )  # tract friendly

            jpos_inv_freq = rep_jpos * self.inv_freq
            jpos_inv_freq_sin = torch.sin(jpos_inv_freq)
            jpos_inv_freq_cos = torch.cos(
                jpos_inv_freq.permute(2, 1, 0) * torch.tensor([1, -1])
            ).permute(2, 1, 0)

            results = (
                torch.stack([jpos_inv_freq_sin, jpos_inv_freq_cos])
                .permute(0, 2, 1)
                .reshape(2, seq_len, -1)
            )
            return results


class AssignSliceIssue(nn.Module):
    """2024-04-15 -> no handling YET: tensor slice assign mutation

    This is not handled by torch ONNX export as well

    This assignation issue is the core of RelPosEncXL impossibility to export.

    This could be solved if tommorow we introduce a dedicated tract operator
    of kind (NOT EXIST TODAY) `tract_core_slice_assign`:

    ```nnef
        slice_infos = [(axis, begin, end, step), ...];
        a_bis = tract_core_slice_assign(a, slice_infos, b);
    ```

    Given the provided jit torch graph observed is of the form:
    ```torch_internals

    a = tensor( ... )
    b = tensor (... )
    c = slice(a, c1, c2, c3, c4)
    d = slice(c, d1, d2, d3, d4)
    e = copy_(d, b, false)

    ```

    """

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


if tract_version() >= "0.21.2":  # prior bug in tract rank range

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


# def test_fast_export_check_equivalent_pos_encoding():
#     original_encoder = RelPosEncXL(100)
#     export_friendly_encoder = FastFixedRelPosEncXL(100)
#     test_inputs = torch.arange(10).float().reshape(2, 5)
#
#     ref_res = original_encoder(test_inputs)
#     print(ref_res.shape)
#     new_res = export_friendly_encoder(test_inputs)
#     assert (ref_res == new_res).all()
