"""Per-token streaming wrapper for HF MambaForCausalLM.

The deploy shape:

    (input_id, conv_states, ssm_states)
        -> (logits, conv_states', ssm_states')

where conv_states is `(L, B=1, D, K)` and ssm_states is `(L, B, D, N)`.
We re-implement the per-token decoding body of `MambaMixer.slow_forward`
inline so it traces cleanly without needing HF's `Cache` machinery.

The per-token math mirrors the `is_decoding=True` branch of
`transformers.models.mamba.modeling_mamba.MambaMixer.slow_forward`.
"""

from __future__ import annotations

import torch
from torch import nn


def _streaming_mixer(
    mixer,
    hidden_states: torch.Tensor,  # (B, D_model)
    conv_state: torch.Tensor,  # (B, D, K)
    ssm_state: torch.Tensor,  # (B, D, N)
):
    """One mixer step for a single token (mirrors slow_forward's decoding path)."""  # noqa: E501
    state_size = mixer.ssm_state_size

    projected = mixer.in_proj(hidden_states)
    x, gate = projected.chunk(2, dim=-1)

    new_conv = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
    conv_weight = mixer.conv1d.weight
    x_conv = (new_conv * conv_weight[:, 0, :]).sum(dim=-1)
    if mixer.use_conv_bias:
        x_conv = x_conv + mixer.conv1d.bias
    x_conv = mixer.act(x_conv)

    ssm_params = mixer.x_proj(x_conv)
    time_step, B_in, C_in = torch.split(
        ssm_params, [mixer.time_step_rank, state_size, state_size], dim=-1
    )
    discrete_time_step = nn.functional.softplus(mixer.dt_proj(time_step))
    A = -torch.exp(mixer.A_log.float())
    discrete_A = torch.exp(A.unsqueeze(0) * discrete_time_step.unsqueeze(-1))
    discrete_B = discrete_time_step.unsqueeze(-1) * B_in.unsqueeze(1)
    deltaB_u = discrete_B * x_conv.unsqueeze(-1)

    new_ssm = discrete_A * ssm_state + deltaB_u
    scan_out = torch.matmul(new_ssm, C_in.unsqueeze(-1)).squeeze(-1)
    scan_out = scan_out + mixer.D * x_conv
    scan_out = scan_out * mixer.act(gate)

    contextualized = mixer.out_proj(scan_out)
    return contextualized, new_conv, new_ssm


class StreamingMamba(nn.Module):
    """Per-token streaming wrapper.

    Signature: ``(token_id, conv_states, ssm_states) -> (logits, conv', ssm')``.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner
        self.num_layers = inner.config.num_hidden_layers
        self.d_model = inner.config.hidden_size
        self.intermediate = inner.backbone.layers[0].mixer.intermediate_size
        self.conv_kernel = inner.backbone.layers[0].mixer.conv_kernel_size
        self.state_size = inner.backbone.layers[0].mixer.ssm_state_size
        self.vocab_size = inner.config.vocab_size

    def forward(
        self,
        input_id: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
    ):
        backbone = self.inner.backbone
        h = backbone.embeddings(input_id)
        if h.dim() == 3:
            h = h[:, 0, :]

        new_conv_list = []
        new_ssm_list = []
        for i, block in enumerate(backbone.layers):
            normed = block.norm(h)
            mixer_out, new_conv_i, new_ssm_i = _streaming_mixer(
                block.mixer, normed, conv_states[i], ssm_states[i]
            )
            h = h + mixer_out
            new_conv_list.append(new_conv_i)
            new_ssm_list.append(new_ssm_i)

        h = backbone.norm_f(h)
        logits = self.inner.lm_head(h)
        new_conv = torch.stack(new_conv_list, dim=0)
        new_ssm = torch.stack(new_ssm_list, dim=0)
        return logits, new_conv, new_ssm
