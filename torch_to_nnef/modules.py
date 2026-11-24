"""Traceable modules shared by architecture-specific export adapters."""

import torch
from torch import nn


class SelectedGatedExperts(nn.Module):
    """Evaluate selected experts without data-dependent Python dispatch.

    Weights use linear-filter layout: gate/up [E, 2H, D], down [E, D, H].
    ``activation(gate, up)`` retains the model's activation, including custom
    softcaps. Gather materializes selected weights before floating-point
    matmuls; this is a portable fallback, not a fused MoE kernel.
    """

    def __init__(self, gate_up_proj, down_proj, activation):
        super().__init__()
        if (
            gate_up_proj.ndim != 3
            or down_proj.ndim != 3
            or gate_up_proj.shape[0] != down_proj.shape[0]
            or gate_up_proj.shape[1] != 2 * down_proj.shape[2]
            or gate_up_proj.shape[2] != down_proj.shape[1]
        ):
            raise ValueError("expected gate/up [E, 2H, D] and down [E, D, H]")
        self.hidden_dim = gate_up_proj.shape[2]
        self.intermediate_dim = down_proj.shape[2]
        # Tract's quantized gather reads rows of a 2-D table. Flatten each
        # expert's matrix into one row, then reshape only the selected rows.
        # These parameter views share storage with the upstream weights.
        self.gate_up_proj = nn.Parameter(
            gate_up_proj.detach().flatten(1), requires_grad=False
        )
        self.down_proj = nn.Parameter(
            down_proj.detach().flatten(1), requires_grad=False
        )
        self.activation = activation

    def forward(self, hidden_states, top_k_index, top_k_weights):
        weights = self.gate_up_proj[top_k_index].reshape(
            *top_k_index.shape, 2 * self.intermediate_dim, self.hidden_dim
        )
        gate_up = torch.matmul(
            weights, hidden_states[:, None, :, None]
        ).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.activation(gate, up)
        out = torch.matmul(
            self.down_proj[top_k_index].reshape(
                *top_k_index.shape, self.hidden_dim, self.intermediate_dim
            ),
            activated.unsqueeze(-1),
        ).squeeze(-1)
        return (
            (out.float() * top_k_weights.unsqueeze(-1))
            .sum(1)
            .to(hidden_states.dtype)
        )
